from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import evaluate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth/area against ground truth")
    parser.add_argument("--results-csv", default="outputs/results.csv")
    parser.add_argument("--gt-csv", required=True, help="CSV with gt_depth_m and gt_area_m2")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_reports(args.results_csv, args.gt_csv, args.output_dir)
    print(f"Evaluation complete. Reports at: {args.output_dir}")


if __name__ == "__main__":
    main()
