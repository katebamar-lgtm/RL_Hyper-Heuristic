"""Backward-compatible Lyon indicator entry point using the final protocol.

The paper's indicators require shared normalization and common Monte-Carlo
samples per instance. This wrapper therefore recomputes both synthetic and
Lyon indicators into postprocessing_reports/step2_recomputed/ rather than
writing an incompatible Lyon-only legacy file.

Usage:
    python scripts/compute_trajet_indicators.py data/lyon
"""
from pathlib import Path
import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default="data/lyon",
                        help="Lyon data directory; used to locate the project root")
    args = parser.parse_args(argv)
    given = Path(args.data_dir).expanduser().resolve()
    if given.name == "lyon" and given.parent.name == "data":
        root = given.parent.parent
    elif (given / "data" / "lyon").is_dir():
        root = given
    else:
        raise SystemExit(f"Cannot resolve project root from {given}. Expected <project>/data/lyon.")
    sys.path.insert(0, str(root))
    from scripts.postprocessing.recompute_indicators import main as recompute_main
    print("Compatibility entry point: recomputes synthetic and Lyon indicators under postprocessing_reports/.")
    return recompute_main([str(root)])


if __name__ == "__main__":
    raise SystemExit(main())
