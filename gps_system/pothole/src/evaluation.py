from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_error_percent(pred: float, gt: float) -> float:
    if gt <= 0:
        return 0.0
    return abs(pred - gt) / gt * 100.0


def evaluate_reports(results_csv: str, gt_csv: str, output_dir: str) -> None:
    pred_df = pd.read_csv(results_csv)
    gt_df = pd.read_csv(gt_csv)

    merged = pred_df.merge(gt_df, on=["frame_idx", "x1", "y1", "x2", "y2"], how="inner")
    if merged.empty:
        raise RuntimeError("No matched records between prediction and ground truth.")

    merged["depth_error_percent"] = merged.apply(
        lambda r: compute_error_percent(r["depth_m"], r["gt_depth_m"]), axis=1
    )
    merged["area_error_percent"] = merged.apply(
        lambda r: compute_error_percent(r["area_m2"], r["gt_area_m2"]), axis=1
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    depth_report = merged[["frame_idx", "depth_m", "gt_depth_m", "depth_error_percent"]]
    area_report = merged[["frame_idx", "area_m2", "gt_area_m2", "area_error_percent"]]

    depth_report.to_csv(out / "depth_error_report.txt", index=False)
    area_report.to_csv(out / "area_error_report.txt", index=False)

    summary = {
        "depth_error_mean_percent": float(depth_report["depth_error_percent"].mean()),
        "area_error_mean_percent": float(area_report["area_error_percent"].mean()),
        "depth_error_median_percent": float(depth_report["depth_error_percent"].median()),
        "area_error_median_percent": float(area_report["area_error_percent"].median()),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
