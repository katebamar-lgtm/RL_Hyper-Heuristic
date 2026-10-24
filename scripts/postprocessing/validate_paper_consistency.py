#!/usr/bin/env python3
"""Regression check: ensure stored data and regenerated statistics match this paper version.

Run AFTER run_full_postprocessing.py:
    python scripts/postprocessing/validate_paper_consistency.py .

This validator fails intentionally if source data or reported values change;
update its expected values only after an explicit manuscript/results revision.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import pandas as pd

EXPECTED = {
    "primary_p_raw": [0.07421875, 0.65234375, 0.00390625, 0.00390625],
    "primary_p_holm": [0.1484375, 0.65234375, 0.015625, 0.015625],
    "friedman_chi2": 85.31313131313135,
    "friedman_p": 4.527763257332609e-14,
    "nemenyi_cd": 5.032274535076397,
    "min_p_holm_55": 0.21484375,
}


def require_close(name: str, observed: float, expected: float, tol: float = 1e-9):
    if not math.isclose(float(observed), float(expected), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name}: observed {observed:.12g}, expected {expected:.12g}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=".")
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    report_dir = root / "postprocessing_reports"
    step2 = report_dir / "step2_recomputed"
    step3 = report_dir / "step3_statistics"

    for generated_rel, stored_rel, dataset_label in [
        ("synthetic_indicators_recomputed.jsonl", "results/phase4_indicators.jsonl", "synthetic"),
        ("lyon_indicators_recomputed.jsonl", "data/lyon/case_study_indicators.jsonl", "lyon"),
    ]:
        generated = [json.loads(x) for x in (step2 / generated_rel).open(encoding="utf-8") if x.strip()]
        stored = [json.loads(x) for x in (root / stored_rel).open(encoding="utf-8") if x.strip()]
        if len(generated) != len(stored):
            raise AssertionError(f"{dataset_label}: generated {len(generated)} rows, stored {len(stored)} rows")
        key = lambda r: (r["instance"], r["algo"], r["rep"])
        gmap, smap = {key(r): r for r in generated}, {key(r): r for r in stored}
        if gmap.keys() != smap.keys():
            raise AssertionError(f"{dataset_label}: generated/stored run keys differ")
        for k in gmap:
            for metric in ("hv_recomputed", "igd_recomputed", "gd_recomputed"):
                require_close(f"{dataset_label} {k} {metric}", gmap[k][metric], smap[k][metric], tol=0.0)
            if gmap[k].get("reference_policy") != smap[k].get("reference_policy"):
                raise AssertionError(f"{dataset_label} {k}: reference policy differs")
        print(f"PASS: {dataset_label}: {len(generated)} recomputed indicator rows reproduce stored indicators exactly")

    primary = pd.read_csv(step3 / "synthetic_primary_HV_instance_level.csv")
    if len(primary) != 4:
        raise AssertionError(f"Primary contrast count is {len(primary)}, expected 4")
    for i, row in primary.reset_index(drop=True).iterrows():
        require_close(f"primary raw p[{i}]", row["p_raw"], EXPECTED["primary_p_raw"][i])
        require_close(f"primary Holm p[{i}]", row["p_holm_4_primary_HV"], EXPECTED["primary_p_holm"][i])
    print("PASS: four primary HV comparisons match the published raw/Holm-adjusted p-values")

    omnibus = pd.read_csv(step3 / "synthetic_friedman_summary.csv").iloc[0]
    require_close("Friedman chi-square", omnibus["friedman_chi2"], EXPECTED["friedman_chi2"])
    require_close("Friedman p-value", omnibus["friedman_p"], EXPECTED["friedman_p"], tol=1e-8)
    require_close("Nemenyi CD", omnibus["nemenyi_cd"], EXPECTED["nemenyi_cd"])
    require_close("minimum Holm p over 55 pairs", omnibus["minimum_holm_p_55"], EXPECTED["min_p_holm_55"])
    print("PASS: 11-algorithm Friedman, Nemenyi critical difference, and all-pairs Holm results match the paper")

    secondary = pd.read_csv(step3 / "synthetic_secondary_IGD_GD_instance_level.csv")
    if len(secondary) != 8:
        raise AssertionError(f"Secondary test count is {len(secondary)}, expected 8 (4 contrasts x IGD/GD)")
    omf_secondary = secondary[secondary["algorithm_A"] == "OMF-RL"]
    if len(omf_secondary) != 4:
        raise AssertionError(f"Expected four OMF-RL secondary tests, found {len(omf_secondary)}")
    for value in omf_secondary["p_holm_8_secondary_IGD_GD"]:
        require_close("OMF secondary Holm p", value, 0.03125)
    print("PASS: secondary IGD/GD family uses exactly 8 tests; OMF-RL adjusted p = 0.03125")
    print("ALL CHECKS PASSED. No optimization runs were required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
