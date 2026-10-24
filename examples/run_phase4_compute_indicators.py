"""Backward-compatible entry point for the FINAL, paper-consistent indicators.

This script intentionally no longer writes the legacy fields `hv`, `igd`, and
`gd` into results/phase4_indicators.jsonl. It delegates to the shared final
post-processing pipeline, which writes recomputed indicators under
postprocessing_reports/step2_recomputed/ and preserves the source files.

From the project root:
    python examples/run_phase4_compute_indicators.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.postprocessing.recompute_indicators import main

if __name__ == "__main__":
    print("Compatibility entry point: computing paper-consistent indicators under postprocessing_reports/.")
    raise SystemExit(main([str(ROOT)]))
