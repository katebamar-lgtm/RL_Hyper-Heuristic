#!/usr/bin/env python3
"""Compatibility wrapper; runs all statistical analyses used in the paper.

Preferred command:
    python scripts/postprocessing/analyze_paper_results.py .
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.postprocessing.analyze_paper_results import main

if __name__ == "__main__":
    raise SystemExit(main())
