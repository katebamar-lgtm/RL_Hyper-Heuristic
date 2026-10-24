"""Compatibility entry point for the final paper-specific analysis.

Older versions read fields named `hv`, `igd`, and `gd`, and contained a
Nemenyi critical-value table limited to ten algorithms. The final manuscript
uses `*_recomputed` indicators and eleven algorithm conditions. To avoid
silently producing empty tables or outdated ranks, this module now delegates
to the canonical post-processing analysis.

Run from the project root after recomputing indicators:
    python -m model.stats_analysis .

Full pipeline (including indicator recomputation, without running optimizers):
    python scripts/postprocessing/run_full_postprocessing.py .
"""
from __future__ import annotations
from pathlib import Path
import sys


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    from scripts.postprocessing.analyze_paper_results import main as analyse_main
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = [str(root)]
    return analyse_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
