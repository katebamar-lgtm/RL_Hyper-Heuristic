#!/usr/bin/env python3
"""Reproduce the statistical tables and key numerical claims in the paper.

This script only reads postprocessing_reports/step2_recomputed/*.jsonl and
writes CSV/TXT reports under postprocessing_reports/step3_statistics/.
It never runs an optimization algorithm and never modifies source result files.

Usage from the project root:
    python scripts/postprocessing/analyze_paper_results.py .

Requires pandas, NumPy and SciPy. Indicator field names intentionally match
the final protocol: hv_recomputed, igd_recomputed, gd_recomputed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, studentized_range

METRICS = ("hv_recomputed", "igd_recomputed", "gd_recomputed")
PRIMARY_COMPARISONS = [
    ("RL-MOEA/D", "MOEA/D", "RL-MOEA/D vs static MOEA/D"),
    ("RL-MOEA/D", "RL-MOEA/D-random", "RL-MOEA/D vs random MOEA/D"),
    ("OMF-RL", "OMF-statique", "OMF-RL vs static OMF"),
    ("OMF-RL", "OMF-aleatoire", "OMF-RL vs random OMF"),
]
LYON_COMPARISONS = [
    ("RL-MOEA/D", "MOEA/D", "RL-MOEA/D vs static MOEA/D"),
    ("RL-MOEA/D", "RL-MOEA/D-random", "RL-MOEA/D vs random MOEA/D"),
    ("RL-MOEA/D-random", "MOEA/D", "random MOEA/D vs static MOEA/D"),
]
ALGORITHMS = [
    "MOGWO", "MOABC", "MOFA", "NSGA-II", "SPEA2", "RL-MOEA/D",
    "RL-MOEA/D-random", "MOEA/D", "OMF-statique", "OMF-aleatoire", "OMF-RL",
]
PRIMARY_ALGORITHMS = [
    "MOEA/D", "RL-MOEA/D", "RL-MOEA/D-random",
    "OMF-statique", "OMF-aleatoire", "OMF-RL",
]
ROUTE_LABELS = {
    "Perreux Les Franchises": "Perreux–Les Franchises",
    "Belleville St Jean Bourg": "Belleville–Saint-Jean-Bourg",
    "Thizy-Les-B. THI. Jodard": "Thizy-les-Bourgs–Jodard",
}


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return np.asarray([], dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def load_jsonl(path: Path, dataset: str, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/postprocessing/recompute_indicators.py first."
        )
    df = pd.read_json(path, lines=True)
    required = {"algo", "instance", "rep", "seed", "dataset", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required fields {sorted(missing)}")
    df = df[df["dataset"].astype(str).str.lower() == dataset].copy()
    if len(df) != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} {dataset} rows, found {len(df)}")
    if df.duplicated(["instance", "algo", "rep"]).any():
        raise ValueError(f"{path.name}: duplicate (instance, algo, rep) keys")
    if df[list(METRICS)].isna().any().any():
        raise ValueError(f"{path.name}: missing indicator values")
    if not np.isfinite(df[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: non-finite indicator values")
    return df


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Wilcoxon; match the paper's n=9 instance-level test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("paired samples must have equal non-zero length")
    if np.all(a == b):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.statistic), float(result.pvalue)


def direction_better(a: np.ndarray, b: np.ndarray, metric: str) -> tuple[int, int]:
    if metric == "hv_recomputed":
        return int(np.sum(a > b)), int(np.sum(a == b))
    return int(np.sum(a < b)), int(np.sum(a == b))


def per_instance_table(df: pd.DataFrame, metric: str, aggregator: str = "mean") -> pd.DataFrame:
    gb = df.groupby(["instance", "algo"])[metric]
    series = gb.mean() if aggregator == "mean" else gb.median()
    wide = series.unstack("algo")
    return wide.sort_index()


def contrast_rows(df: pd.DataFrame, dataset: str, comparisons, metric: str,
                  aggregator: str = "mean") -> list[dict]:
    sub = df[df["dataset"] == dataset] if "dataset" in df.columns else df
    wide = per_instance_table(sub, metric, aggregator)
    rows = []
    for a_name, b_name, label in comparisons:
        if a_name not in wide.columns or b_name not in wide.columns:
            raise ValueError(f"Missing algorithm in contrast: {a_name} vs {b_name}")
        matched = wide[[a_name, b_name]].dropna()
        a = matched[a_name].to_numpy(dtype=float)
        b = matched[b_name].to_numpy(dtype=float)
        wins, ties = direction_better(a, b, metric)
        w, p = paired_test(a, b)
        rows.append({
            "dataset": dataset,
            "unit": f"instance-level {aggregator} across repetitions",
            "comparison": label, "algorithm_A": a_name, "algorithm_B": b_name,
            "metric": metric, "n_units": len(matched), "A_better": wins, "ties": ties,
            "mean_A_across_units": float(a.mean()), "mean_B_across_units": float(b.mean()),
            "median_paired_difference_A_minus_B": float(np.median(a-b)),
            "wilcoxon_W": w, "p_raw": p,
        })
    return rows


def rank_table(wide: pd.DataFrame, algorithms: list[str], metric: str = "hv_recomputed") -> tuple[pd.DataFrame, dict]:
    missing = [a for a in algorithms if a not in wide.columns]
    if missing:
        raise ValueError(f"Missing algorithms for rank table: {missing}")
    data = wide[algorithms].dropna(axis=0, how="any")
    if data.shape[0] < 2:
        raise ValueError("At least two complete instances are required")
    # Rank 1 is always best; ties receive average ranks (unlike the old script).
    rows = []
    for inst, row in data.iterrows():
        vals = row.to_numpy(dtype=float)
        oriented = -vals if metric == "hv_recomputed" else vals
        ranks = rankdata(oriented, method="average")
        rows.append({"instance": inst, **dict(zip(algorithms, ranks))})
    rank_df = pd.DataFrame(rows).set_index("instance")
    mean_ranks = rank_df[algorithms].mean(axis=0).to_dict()
    return rank_df, mean_ranks


def run_friedman_nemenyi(syn: pd.DataFrame, out_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    wide = per_instance_table(syn, "hv_recomputed", "mean")
    complete = wide[ALGORITHMS].dropna(axis=0, how="any")
    stat, p = stats.friedmanchisquare(*[complete[a].to_numpy(dtype=float) for a in ALGORITHMS])
    rank_df, mean_ranks = rank_table(complete, ALGORITHMS, "hv_recomputed")
    k, n = len(ALGORITHMS), len(complete)
    q_crit = float(studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2.0))
    cd = float(q_crit * math.sqrt(k * (k+1) / (6.0*n)))
    rank_rows = [{"algorithm": a, "mean_rank": mean_ranks[a], "rank_1_is_best": True}
                 for a in ALGORITHMS]
    rank_df.to_csv(out_dir / "synthetic_friedman_ranks_by_instance.csv")
    ranks_summary = pd.DataFrame(rank_rows).sort_values("mean_rank")
    ranks_summary.to_csv(out_dir / "synthetic_friedman_mean_ranks.csv", index=False)

    pair_rows = []
    for a, b in itertools.combinations(ALGORITHMS, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        pair_rows.append({"algorithm_A": a, "algorithm_B": b,
                          "mean_rank_A": mean_ranks[a], "mean_rank_B": mean_ranks[b],
                          "absolute_rank_difference": diff, "nemenyi_critical_difference": cd,
                          "significant_at_0.05": bool(diff > cd)})
    nemenyi_df = pd.DataFrame(pair_rows).sort_values(
        ["significant_at_0.05", "absolute_rank_difference"], ascending=[False, False]
    )
    nemenyi_df.to_csv(out_dir / "synthetic_nemenyi_pairs.csv", index=False)

    # All 55 paired Wilcoxon tests on the nine instance-level mean HVs, Holm-corrected together.
    all_pairs = []
    for a, b in itertools.combinations(ALGORITHMS, 2):
        x = complete[a].to_numpy(dtype=float)
        y = complete[b].to_numpy(dtype=float)
        w, p_pair = paired_test(x, y)
        all_pairs.append({"algorithm_A": a, "algorithm_B": b,
                          "mean_HV_A": float(x.mean()), "mean_HV_B": float(y.mean()),
                          "mean_rank_A": mean_ranks[a], "mean_rank_B": mean_ranks[b],
                          "wilcoxon_W": w, "p_raw": p_pair})
    all_pairs_df = pd.DataFrame(all_pairs)
    all_pairs_df["p_holm_55"] = holm_adjust(all_pairs_df["p_raw"].to_numpy())
    all_pairs_df = all_pairs_df.sort_values(["p_holm_55", "p_raw"])
    all_pairs_df.to_csv(out_dir / "synthetic_allpairs_wilcoxon_holm.csv", index=False)

    summary = {
        "n_instances": int(n), "n_algorithms": int(k), "friedman_chi2": float(stat),
        "friedman_p": float(p), "nemenyi_q_critical": q_crit, "nemenyi_cd": cd,
        "mean_ranks": mean_ranks,
        "minimum_holm_p_55": float(all_pairs_df["p_holm_55"].min()),
        "nemenyi_significant_pairs": int(nemenyi_df["significant_at_0.05"].sum()),
    }
    pd.DataFrame([{
        "n_instances": n, "n_algorithms": k, "friedman_chi2": stat, "friedman_p": p,
        "nemenyi_q_critical": q_crit, "nemenyi_cd": cd,
        "minimum_holm_p_55": summary["minimum_holm_p_55"],
        "nemenyi_significant_pairs": summary["nemenyi_significant_pairs"],
    }]).to_csv(out_dir / "synthetic_friedman_summary.csv", index=False)
    return summary, ranks_summary, nemenyi_df


def build_descriptive_tables(syn: pd.DataFrame, lyon: pd.DataFrame, out_dir: Path) -> None:
    # Table 5: run-level mean +/- SD within each size category; ranks come from 9 instance means.
    group = syn.groupby(["category", "algo"])["hv_recomputed"].agg(["count", "mean", "std"]).reset_index()
    group.rename(columns={"count":"n_runs", "mean":"mean_hv", "std":"sd_hv", "algo":"algorithm", "category":"scale"}, inplace=True)
    group.to_csv(out_dir / "synthetic_all_algorithm_HV_by_scale.csv", index=False)

    six = syn[syn["algo"].isin(PRIMARY_ALGORITHMS)].copy()
    six_groups = six.groupby(["category", "algo"], sort=False).agg(
        n_runs=("hv_recomputed", "size"),
        mean_hv=("hv_recomputed", "mean"), sd_hv=("hv_recomputed", "std"),
        mean_igd=("igd_recomputed", "mean"), sd_igd=("igd_recomputed", "std"),
        mean_gd=("gd_recomputed", "mean"), sd_gd=("gd_recomputed", "std"),
        mean_n_evals=("n_evals", "mean"), mean_front_size=("front_size", "mean"),
        mean_unique_nondominated_points=("front_size_nondominated_unique", "mean"),
        mean_elapsed_seconds=("cpu_seconds", "mean"), max_budget=("budget", "max"),
    ).reset_index().rename(columns={"category":"scale", "algo":"algorithm"})
    six_groups.to_csv(out_dir / "synthetic_six_primary_conditions_by_scale.csv", index=False)

    ly = lyon.copy()
    ly["route_label"] = ly["instance"].map(ROUTE_LABELS).fillna(ly["instance"])
    route_summary = ly.groupby(["route_label", "algo"]).agg(
        n_runs=("hv_recomputed", "size"), mean_hv=("hv_recomputed", "mean"), sd_hv=("hv_recomputed", "std"),
        mean_igd=("igd_recomputed", "mean"), mean_gd=("gd_recomputed", "mean"),
        mean_n_evals=("n_evals", "mean"), mean_elapsed_seconds=("cpu_seconds", "mean"),
    ).reset_index().rename(columns={"algo":"algorithm"})
    route_summary.to_csv(out_dir / "lyon_all_algorithms_by_route.csv", index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=".", help="Project root; default=current directory")
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    step2 = root / "postprocessing_reports" / "step2_recomputed"
    out_dir = root / "postprocessing_reports" / "step3_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    syn = load_jsonl(step2 / "synthetic_indicators_recomputed.jsonl", "synthetic", 1650)
    lyon = load_jsonl(step2 / "lyon_indicators_recomputed.jsonl", "lyon", 195)

    # Primary endpoint: four predeclared HV comparisons, Holm correction over exactly four tests.
    primary_df = pd.DataFrame(contrast_rows(syn, "synthetic", PRIMARY_COMPARISONS, "hv_recomputed", "mean"))
    primary_df["p_holm_4_primary_HV"] = holm_adjust(primary_df["p_raw"].to_numpy())
    primary_df.to_csv(out_dir / "synthetic_primary_HV_instance_level.csv", index=False)

    median_df = pd.DataFrame(contrast_rows(syn, "synthetic", PRIMARY_COMPARISONS, "hv_recomputed", "median"))
    median_df["p_holm_4_primary_HV"] = holm_adjust(median_df["p_raw"].to_numpy())
    median_df.to_csv(out_dir / "synthetic_primary_HV_instance_median_sensitivity.csv", index=False)

    # Secondary endpoints: IGD and GD only => 4 contrasts x 2 metrics = 8 p-values.
    secondary_rows = []
    for metric in ("igd_recomputed", "gd_recomputed"):
        secondary_rows.extend(contrast_rows(syn, "synthetic", PRIMARY_COMPARISONS, metric, "mean"))
    secondary_df = pd.DataFrame(secondary_rows)
    secondary_df["p_holm_8_secondary_IGD_GD"] = holm_adjust(secondary_df["p_raw"].to_numpy())
    secondary_df.to_csv(out_dir / "synthetic_secondary_IGD_GD_instance_level.csv", index=False)

    # Pooled repetition-level matched tests are sensitivity-only (not independent benchmark units).
    pooled = []
    for a_name, b_name, label in PRIMARY_COMPARISONS:
        for metric in METRICS:
            a = syn[syn.algo == a_name][["instance", "rep", "seed", metric]].rename(columns={"seed":"seed_a", metric:"A"})
            b = syn[syn.algo == b_name][["instance", "rep", "seed", metric]].rename(columns={"seed":"seed_b", metric:"B"})
            m = a.merge(b, on=["instance", "rep"], validate="one_to_one")
            if m.empty or (m.seed_a != m.seed_b).any():
                raise ValueError(f"Unmatched repetition seeds: {a_name} vs {b_name}")
            w, pval = paired_test(m.A.to_numpy(), m.B.to_numpy())
            wins, ties = direction_better(m.A.to_numpy(), m.B.to_numpy(), metric)
            pooled.append({"comparison":label,"algorithm_A":a_name,"algorithm_B":b_name,"metric":metric,
                           "unit":"matched run (sensitivity only)","n_pairs":len(m),"A_better":wins,"ties":ties,
                           "mean_A":float(m.A.mean()),"mean_B":float(m.B.mean()),"wilcoxon_W":w,"p_raw":pval})
    pd.DataFrame(pooled).to_csv(out_dir / "synthetic_pooled_run_sensitivity.csv", index=False)

    # Lyon is descriptive: route means are only three units and testing has very low resolution.
    lyon_tests = pd.DataFrame(contrast_rows(lyon, "lyon", LYON_COMPARISONS, "hv_recomputed", "mean"))
    lyon_tests.to_csv(out_dir / "lyon_route_level_HV_comparisons.csv", index=False)

    omnibus, mean_ranks, nemenyi_df = run_friedman_nemenyi(syn, out_dir)
    build_descriptive_tables(syn, lyon, out_dir)

    # A compact human-readable report records values and decisions used in the manuscript.
    report = [
        "PAPER POST-PROCESSING REPRODUCTION REPORT",
        f"Project root: {root.name} (resolved at runtime)",
        "Optimization algorithms executed: NO.",
        "Input source: postprocessing_reports/step2_recomputed/*.jsonl; raw inputs unchanged.",
        "", "PRIMARY HV TESTS (9 INSTANCE-LEVEL MEANS; HOLM FAMILY SIZE = 4)",
    ]
    for _, r in primary_df.iterrows():
        report.append(f"{r.comparison}: wins={int(r.A_better)}/{int(r.n_units)}, mean_A={r.mean_A_across_units:.6f}, mean_B={r.mean_B_across_units:.6f}, p={r.p_raw:.8g}, Holm4={r.p_holm_4_primary_HV:.8g}")
    report += ["", "SECONDARY IGD/GD TESTS (HOLM FAMILY SIZE = 8)"]
    for _, r in secondary_df.iterrows():
        report.append(f"{r.comparison} [{r.metric}]: p={r.p_raw:.8g}, Holm8={r.p_holm_8_secondary_IGD_GD:.8g}")
    report += [
        "", "11-ALGORITHM OMNIBUS AND NEMENYI (9 INSTANCE-LEVEL MEANS)",
        f"Friedman chi2={omnibus['friedman_chi2']:.8f}; p={omnibus['friedman_p']:.8g}; k={omnibus['n_algorithms']}; N={omnibus['n_instances']}",
        f"Nemenyi q_critical={omnibus['nemenyi_q_critical']:.8f}; CD={omnibus['nemenyi_cd']:.8f}; significant pairs={omnibus['nemenyi_significant_pairs']}",
        f"Minimum pairwise Wilcoxon p after Holm correction over 55 pairs={omnibus['minimum_holm_p_55']:.8g}",
        "Mean ranks (rank 1 = best):",
    ]
    for _, r in mean_ranks.iterrows():
        report.append(f"  {r['algorithm']}: {r['mean_rank']:.6f}")
    report.append("Nemenyi-significant pairs (absolute mean-rank difference > CD):")
    for _, r in nemenyi_df[nemenyi_df["significant_at_0.05"]].iterrows():
        report.append(f"  {r['algorithm_A']} vs {r['algorithm_B']}: rank difference={r['absolute_rank_difference']:.6f}")
    report += ["", "LYON ROUTE-LEVEL HV COMPARISONS (DESCRIPTIVE; N=3 ROUTES)"]
    for _, r in lyon_tests.iterrows():
        report.append(f"{r.comparison}: wins={int(r.A_better)}/3, mean_A={r.mean_A_across_units:.6f}, mean_B={r.mean_B_across_units:.6f}, p={r.p_raw:.8g}")
    report += [
        "", "INTERPRETATION NOTES",
        "- Repetition-level pooled tests are sensitivity analyses only; they are not the main inference across instances.",
        "- The primary HV family uses Holm correction across four prespecified contrasts.",
        "- The secondary family uses Holm correction across the eight IGD/GD contrasts only (4 contrasts x 2 metrics).",
        "- The omnibus Friedman/Nemenyi and all-pairs Wilcoxon analyses are a separate exploratory family across 11 algorithms.",
        "- IGD/GD should not be ranked between algorithm groups with different reference_policy labels.",
        "- cpu_seconds is measured as elapsed wall-clock time in the raw-run code; it is not CPU time.",
    ]
    (out_dir / "paper_reproduction_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Paper analysis reproduction complete; no optimization runs executed.")
    print(f"Outputs: {out_dir}")
    print(f"Friedman: chi2={omnibus['friedman_chi2']:.6f}, p={omnibus['friedman_p']:.6g}; Nemenyi CD={omnibus['nemenyi_cd']:.4f}")
    print(f"Primary HV results: {out_dir / 'synthetic_primary_HV_instance_level.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
