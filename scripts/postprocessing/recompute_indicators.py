#!/usr/bin/env python3
"""Reproducible post-processing of saved multi-objective fronts.

This script DOES NOT run any optimization algorithm and DOES NOT overwrite
source result files. Run it from the project root:
    python postprocessing_step2_recompute.py
or pass the project root as its only argument.

Outputs are written to postprocessing_reports/step2_recomputed/.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

N_SAMPLES_DEFAULT = 20000
REF_POINT = 1.1

# Existing weighted-Dijkstra decoders are treated as one family in the
# current manuscript, to reduce the known overlap among MOGWO/MOFA/MOABC.
DIJKSTRA_FAMILY = {"MOGWO", "MOFA", "MOABC"}
SYNTH_MOEAD_GROUP = {"MOEA/D", "RL-MOEA/D", "RL-MOEA/D-random"}
SYNTH_OMF_GROUP = {"OMF-statique", "OMF-aleatoire", "OMF-RL"}
LYON_MOEAD_GROUP = {"MOEA/D", "RL-MOEA/D", "RL-MOEA/D-random"}

FAMILIES = {
    "NSGA-II": {"NSGA-II"},
    "SPEA2": {"SPEA2"},
    "MOEA/D": {"MOEA/D"},
    "MOGWO": DIJKSTRA_FAMILY,
    "MOFA": DIJKSTRA_FAMILY,
    "MOABC": DIJKSTRA_FAMILY,
    "RL-MOEA/D": {"RL-MOEA/D"},
    "RL-MOEA/D-random": {"RL-MOEA/D-random"},
    "OMF-statique": {"OMF-statique"},
    "OMF-aleatoire": {"OMF-aleatoire"},
    "OMF-RL": {"OMF-RL"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalide dans {path}, ligne {line_number}: {exc}")
    if not rows:
        raise ValueError(f"Aucun résultat trouvé dans {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]], dataset_label: str) -> None:
    required = {"instance", "algo", "rep", "objectives", "n_evals", "front_size", "cpu_seconds"}
    seen = set()
    for idx, r in enumerate(rows, start=1):
        missing = required - set(r)
        if missing:
            raise ValueError(f"{dataset_label}: ligne {idx}, champs manquants: {sorted(missing)}")
        key = (r["instance"], r["algo"], r["rep"])
        if key in seen:
            raise ValueError(f"{dataset_label}: doublon de clé {key}")
        seen.add(key)
        if not isinstance(r["objectives"], list):
            raise ValueError(f"{dataset_label}: objectives non-liste pour {key}")
        for p in r["objectives"]:
            if not isinstance(p, (list, tuple)) or len(p) != 4:
                raise ValueError(f"{dataset_label}: vecteur objectif invalide pour {key}: {p}")
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in p):
                raise ValueError(f"{dataset_label}: valeur objectif non finie pour {key}: {p}")


def unique_points(points: Iterable[Iterable[float]]) -> list[tuple[float, ...]]:
    # Dict preserves insertion order and deduplicates exact objective vectors.
    return list(dict.fromkeys(tuple(float(x) for x in p) for p in points))


def fast_non_dominated_indices(points: list[tuple[float, ...]], block_size: int = 256) -> list[int]:
    """Vectorized Pareto filtering; avoids slow nested Python loops."""
    arr = np.asarray(points, dtype=float)
    n = len(arr)
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        targets = arr[start:end]
        no_worse = np.all(arr[:, None, :] <= targets[None, :, :], axis=2)
        strictly_better = np.any(arr[:, None, :] < targets[None, :, :], axis=2)
        dominated[start:end] = np.any(no_worse & strictly_better, axis=0)
    return np.flatnonzero(~dominated).tolist()


def non_dominated(points: list[tuple[float, ...]], nd_func) -> list[tuple[float, ...]]:
    points = unique_points(points)
    if not points:
        return []
    idx = nd_func(points)
    return [points[i] for i in idx]


def calculate_bounds(rows: list[dict[str, Any]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    points = [tuple(float(x) for x in p) for r in rows for p in r["objectives"]]
    if not points:
        raise ValueError("Aucun vecteur objectif disponible pour le calcul des bornes")
    m = len(points[0])
    ideal = tuple(min(p[k] for p in points) for k in range(m))
    nadir = tuple(max(p[k] for p in points) for k in range(m))
    return ideal, nadir


def normalize(points: Iterable[Iterable[float]], ideal: tuple[float, ...],
              nadir: tuple[float, ...]) -> list[tuple[float, ...]]:
    output = []
    for p in points:
        row = []
        for k, val in enumerate(p):
            span = nadir[k] - ideal[k]
            row.append(0.0 if span == 0 else (float(val) - ideal[k]) / span)
        output.append(tuple(row))
    return output


def euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def calculate_gd_igd(candidate: list[tuple[float, ...]], reference: list[tuple[float, ...]],
                      ideal: tuple[float, ...], nadir: tuple[float, ...]) -> tuple[float, float]:
    if not candidate or not reference:
        return float("inf"), float("inf")
    cand_n = np.asarray(normalize(candidate, ideal, nadir), dtype=float)
    ref_n = np.asarray(normalize(reference, ideal, nadir), dtype=float)
    # Vectorized distances: GD is candidate -> reference; IGD is reference -> candidate.
    delta = cand_n[:, None, :] - ref_n[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    gd = float(np.mean(np.min(distances, axis=1)))
    igd = float(np.mean(np.min(distances, axis=0)))
    return gd, igd


def estimate_hv_common_samples(candidate_normalized: list[tuple[float, ...]],
                                samples: np.ndarray,
                                reference_value: float = REF_POINT,
                                sample_block: int = 1024) -> float:
    """Fast deterministic Monte-Carlo HV using shared sample locations.

    This estimates the same dominated-box fraction as model.pareto.hypervolume,
    but vectorizes the comparisons for much faster post-processing.
    """
    if not candidate_normalized:
        return 0.0
    cand = np.asarray(candidate_normalized, dtype=float)
    dominated_count = 0
    for start in range(0, len(samples), sample_block):
        block = samples[start:start + sample_block]
        # A sample is dominated when any candidate point is <= it in all objectives.
        covered = np.any(np.all(cand[None, :, :] <= block[:, None, :], axis=2), axis=1)
        dominated_count += int(np.count_nonzero(covered))
    box_volume = reference_value ** cand.shape[1]
    return (dominated_count / len(samples)) * box_volume


def stable_seed(dataset: str, instance: str) -> int:
    digest = hashlib.sha256(f"{dataset}|{instance}|HV-v2-common-samples".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def get_reference_groups(dataset: str, algos: set[str]) -> dict[str, tuple[set[str], str]]:
    """Return policy name -> (excluded algorithms, description).

    The three primary conditions within each host branch share one empirical
    reference front that excludes all three compared conditions and the
    known highly overlapping weighted-Dijkstra family. Other algorithms keep
    the existing leave-family-out reference rule, for descriptive reporting.
    """
    groups: dict[str, tuple[set[str], str]] = {}
    if dataset == "synthetic":
        if SYNTH_MOEAD_GROUP <= algos:
            groups["primary_moead"] = (
                SYNTH_MOEAD_GROUP | DIJKSTRA_FAMILY,
                "common reference for MOEA/D, RL-MOEA/D, and RL-MOEA/D-random; excludes the compared group and weighted-Dijkstra family",
            )
        if SYNTH_OMF_GROUP <= algos:
            groups["primary_omf"] = (
                SYNTH_OMF_GROUP | DIJKSTRA_FAMILY,
                "common reference for OMF-static, OMF-random, and OMF-RL; excludes the compared group and weighted-Dijkstra family",
            )
    elif dataset == "lyon":
        if LYON_MOEAD_GROUP <= algos:
            groups["primary_moead"] = (
                LYON_MOEAD_GROUP | DIJKSTRA_FAMILY,
                "common reference for MOEA/D, RL-MOEA/D, and RL-MOEA/D-random; excludes the compared group and weighted-Dijkstra family",
            )
    # Register leave-family-out references for remaining descriptive rows.
    for algo in sorted(algos):
        if dataset == "synthetic" and (algo in SYNTH_MOEAD_GROUP or algo in SYNTH_OMF_GROUP):
            continue
        if dataset == "lyon" and algo in LYON_MOEAD_GROUP:
            continue
        excluded = set(FAMILIES.get(algo, {algo}))
        key = f"legacy_family_exclusion__{algo}"
        groups[key] = (excluded, f"leave-family-out reference for descriptive row {algo}")
    return groups


def process_dataset(rows: list[dict[str, Any]], dataset: str, output_dir: Path,
                    nd_func, n_samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_instance[str(r["instance"])].append(r)
    all_algos = {str(r["algo"]) for r in rows}
    policy_definitions = get_reference_groups(dataset, all_algos)
    output_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for instance in sorted(by_instance):
        inst_rows = by_instance[instance]
        ideal, nadir = calculate_bounds(inst_rows)  # shared by all algorithms for this instance
        mc_seed = stable_seed(dataset, instance)
        mc_rng = np.random.default_rng(mc_seed)
        mc_samples = mc_rng.uniform(0.0, REF_POINT, size=(n_samples, 4))
        all_points_by_algo: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        for r in inst_rows:
            all_points_by_algo[str(r["algo"])].extend(tuple(float(x) for x in p) for p in r["objectives"])

        # A common empirical reference front is calculated for each explicitly defined policy.
        ref_by_policy: dict[str, list[tuple[float, ...]]] = {}
        ref_cache_by_exclusions: dict[frozenset[str], list[tuple[float, ...]]] = {}
        for policy, (excluded, description) in policy_definitions.items():
            exclusion_key = frozenset(excluded)
            if exclusion_key not in ref_cache_by_exclusions:
                pool = []
                for algo, points in all_points_by_algo.items():
                    if algo not in excluded:
                        pool.extend(points)
                ref = non_dominated(pool, nd_func)
                if not ref:
                    raise ValueError(f"Référence vide pour {dataset}/{instance}/{policy}; exclus={excluded}")
                ref_cache_by_exclusions[exclusion_key] = ref
            ref = ref_cache_by_exclusions[exclusion_key]
            ref_by_policy[policy] = ref
            manifest.append({
                "dataset": dataset, "instance": instance, "reference_policy": policy,
                "excluded_algorithms": sorted(excluded), "reference_front_size": len(ref),
                "ideal_bounds": list(ideal), "nadir_bounds": list(nadir),
                "reference_point_normalized": [REF_POINT] * 4,
                "monte_carlo_samples": n_samples,
                "monte_carlo_seed": stable_seed(dataset, instance),
                "policy_description": description,
            })

        for r in inst_rows:
            algo = str(r["algo"])
            if dataset == "synthetic" and algo in SYNTH_MOEAD_GROUP:
                policy = "primary_moead"
            elif dataset == "synthetic" and algo in SYNTH_OMF_GROUP:
                policy = "primary_omf"
            elif dataset == "lyon" and algo in LYON_MOEAD_GROUP:
                policy = "primary_moead"
            else:
                policy = f"legacy_family_exclusion__{algo}"
            if policy not in ref_by_policy:
                raise ValueError(f"Référence absente: {dataset}/{instance}/{algo}/{policy}")

            candidate = non_dominated([tuple(float(x) for x in p) for p in r["objectives"]], nd_func)
            reference = ref_by_policy[policy]
            candidate_n = normalize(candidate, ideal, nadir)
            seed = mc_seed
            # Every method on this instance is evaluated against exactly the
            # same pre-generated Monte-Carlo sample locations.
            hv = estimate_hv_common_samples(candidate_n, mc_samples)
            gd, igd = calculate_gd_igd(candidate, reference, ideal, nadir)

            out = {k: v for k, v in r.items() if k != "objectives"}
            out.update({
                "dataset": dataset,
                "hv_recomputed": hv,
                "igd_recomputed": igd,
                "gd_recomputed": gd,
                "front_size_raw": len(r["objectives"]),
                "front_size_nondominated_unique": len(candidate),
                "reference_policy": policy,
                "reference_front_size": len(reference),
                "normalization_ideal": list(ideal),
                "normalization_nadir": list(nadir),
                "hv_mc_seed": seed,
                "hv_mc_samples": n_samples,
            })
            output_rows.append(out)

    return output_rows, manifest


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], dataset: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        if r["dataset"] != dataset:
            continue
        if dataset == "synthetic":
            group_label = r.get("category", "unknown")
        else:
            group_label = r["instance"]  # retain route-level summaries for Lyon
        groups[(group_label, r["algo"], r["reference_policy"])].append(r)

    fields = ["dataset", "instance_or_category", "algorithm", "reference_policy", "n_runs",
              "mean_hv", "sd_hv", "mean_igd", "sd_igd", "mean_gd", "sd_gd",
              "mean_n_evals", "mean_front_size", "mean_unique_objective_points", "mean_elapsed_seconds"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (label, algo, policy), rs in sorted(groups.items()):
            hv_m, hv_sd = mean_sd([r["hv_recomputed"] for r in rs])
            igd_m, igd_sd = mean_sd([r["igd_recomputed"] for r in rs])
            gd_m, gd_sd = mean_sd([r["gd_recomputed"] for r in rs])
            eval_m, _ = mean_sd([r["n_evals"] for r in rs])
            front_m, _ = mean_sd([r["front_size"] for r in rs])
            unique_pts_m, _ = mean_sd([r.get("front_size_nondominated_unique", r["front_size"]) for r in rs])
            elapsed_m, _ = mean_sd([r["cpu_seconds"] for r in rs])
            writer.writerow({
                "dataset": dataset, "instance_or_category": label, "algorithm": algo,
                "reference_policy": policy, "n_runs": len(rs),
                "mean_hv": hv_m, "sd_hv": hv_sd,
                "mean_igd": igd_m, "sd_igd": igd_sd,
                "mean_gd": gd_m, "sd_gd": gd_sd,
                "mean_n_evals": eval_m, "mean_front_size": front_m,
                "mean_unique_objective_points": unique_pts_m,
                "mean_elapsed_seconds": elapsed_m,
            })


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=".", help="Racine du projet (défaut: répertoire courant)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT, help="Nombre d'échantillons Monte-Carlo pour HV")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    outdir = root / "postprocessing_reports" / "step2_recomputed"
    outdir.mkdir(parents=True, exist_ok=True)

    # No project module is modified or executed; this script uses a local,
    # vectorized implementation of Pareto filtering and Monte-Carlo HV.

    synth_path = root / "results" / "phase4_raw_runs.jsonl"
    lyon_path = root / "data" / "lyon" / "case_study_raw_runs.jsonl"
    if not synth_path.exists() or not lyon_path.exists():
        raise SystemExit("Fichiers bruts introuvables. Lancez le script depuis la racine du projet ou passez celle-ci en argument.")

    datasets = []
    for label, path in [("synthetic", synth_path), ("lyon", lyon_path)]:
        rows = read_jsonl(path)
        validate_rows(rows, label)
        datasets.append((label, rows))

    all_out = []
    all_manifest = []
    for label, rows in datasets:
        processed, manifest = process_dataset(rows, label, outdir, fast_non_dominated_indices, args.n_samples)
        outname = "synthetic" if label == "synthetic" else "lyon"
        write_jsonl(outdir / f"{outname}_indicators_recomputed.jsonl", processed)
        write_summary(outdir / f"{outname}_summary_recomputed.csv", processed, label)
        all_out.extend(processed)
        all_manifest.extend(manifest)
        print(f"{label}: {len(rows)} exécutions post-traitées (source inchangée)")

    with (outdir / "reference_and_normalization_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "method": "step2 shared per-instance normalization; deterministic Monte-Carlo HV with common sample locations; shared reference front within each primary comparison group",
            "n_samples": args.n_samples,
            "normalized_hv_reference_point": [REF_POINT] * 4,
            "note": "Primary RL-vs-static-vs-random comparisons use a common empirical reference front per branch. Other algorithms retain leave-family-out fronts for descriptive reporting; do not compare IGD/GD across different reference_policy labels.",
            "source_files_read_only": ["results/phase4_raw_runs.jsonl", "data/lyon/case_study_raw_runs.jsonl"],
            "references": all_manifest,
        }, f, ensure_ascii=False, indent=2)

    # Human-readable audit trail.
    lines = [
        "POST-PROCESSING STEP 2 — RECOMPUTED INDICATORS",
        "",
        f"Project root: {root.name} (resolved at runtime)",
        f"Monte-Carlo samples per run: {args.n_samples}",
        "Normalization: shared ideal/nadir per instance, computed from all stored objective fronts across all algorithms and repetitions.",
        "Hypervolume: normalized objectives; reference point (1.1, 1.1, 1.1, 1.1); fixed deterministic seed per dataset/instance; identical sample locations across algorithms within an instance.",
        "Reference fronts: common within each primary comparison group (MOEA/D trio or OMF trio), excluding the compared group and weighted-Dijkstra family; leave-family-out references retained for descriptive reference algorithms.",
        "GD/IGD: computed using the shared per-instance normalization bounds and the selected reference front.",
        "Source files are read-only; no optimization runs were executed.",
        "",
        f"Synthetic output rows: {sum(1 for r in all_out if r['dataset']=='synthetic')}",
        f"Lyon output rows: {sum(1 for r in all_out if r['dataset']=='lyon')}",
        "",
        "IMPORTANT INTERPRETATION NOTE:",
        "- Primary static/RL/random comparisons use a common reference front within each host-algorithm family.",
        "- The MOEA/D-group and OMF-group reference fronts differ; do not rank the two groups against each other by IGD/GD.",
        "- Descriptive baseline algorithms use leave-family-out reference fronts and should not be compared with primary-group IGD/GD unless the reference_policy is the same.",
        "- These are empirical approximation fronts, not known true Pareto fronts.",
    ]
    (outdir / "step2_recompute_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sorties écrites dans: {outdir}")
    print("Aucun résultat brut ni fichier d'indicateurs existant n'a été modifié.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
