#!/usr/bin/env python3
"""Run the paper's reproducible post-processing pipeline (no optimizations).

Usage from project root:
    python scripts/postprocessing/run_full_postprocessing.py .

Reads the included raw-run JSONL files and writes only under
postprocessing_reports/. It does not overwrite raw runs or the legacy
results/phase4_indicators.jsonl files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.postprocessing.recompute_indicators import main as recompute_main
from scripts.postprocessing.analyze_paper_results import main as analyze_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", nargs="?", default=".")
    args = parser.parse_args(argv)
    root = str(Path(args.project_root).expanduser().resolve())
    print("[1/2] Recomputing final indicators from stored fronts; no optimizers are run.")
    code = recompute_main([root])
    if code != 0:
        return code
    print("[2/2] Running the paper's statistical analyses and exporting tables.")
    return analyze_main([root])


if __name__ == "__main__":
    raise SystemExit(main())
