"""
Đánh giá sai số depth/area trên tập test ảnh so với ground-truth.csv.

Quy ước:
  - Ảnh `data/images/test/img-<n>.jpg` ↔ các dòng GT có `frame_idx == n`.
  - Bbox trong CSV được gán trên không gian ảnh tham chiếu (mặc định 576×970, suy từ max tọa độ
    trong toàn bộ CSV; có thể chỉnh bằng --gt-ref-w / --gt-ref-h).
  - Scale bbox GT về kích thước ảnh test hiện tại, chạy lại estimate_depth_m / estimate_area_m2
    trên cùng bbox (so với chân trị depth_m, area_m2 trong CSV). Không cần khớp YOLO detector.

Outputs (mặc định `test_Estimation/`):
  - per_detection_errors.csv
  - summary_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CameraConfig  # noqa: E402
from src.geometry import estimate_area_m2, estimate_depth_m  # noqa: E402
from src.models import DepthOnnxEstimator  # noqa: E402

_IMG_STEM = re.compile(r"^img-(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-set depth/area error vs ground-truth.csv (scaled GT boxes)")
    p.add_argument(
        "--gt-csv",
        default=str(PROJECT_ROOT / "output_dept_area" / "ground-truth.csv"),
    )
    p.add_argument(
        "--test-images",
        default=str(PROJECT_ROOT / "data" / "images" / "test"),
    )
    p.add_argument(
        "--gt-ref-w",
        type=int,
        default=None,
        help="Chiều rộng không gian bbox trong CSV (mặc định: max x2 trong CSV + 1)",
    )
    p.add_argument(
        "--gt-ref-h",
        type=int,
        default=None,
        help="Chiều cao không gian bbox trong CSV (mặc định: max y2 trong CSV + 1)",
    )
    p.add_argument("--depth-onnx", default=None)
    p.add_argument("--depth-input-size", type=int, default=None)
    p.add_argument("--camera-config", default=str(PROJECT_ROOT / "configs" / "camera_config.yaml"))
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "test_Estimation"),
    )
    return p.parse_args()


def frame_id_from_stem(stem: str) -> Optional[int]:
    m = _IMG_STEM.match(stem)
    return int(m.group(1)) if m else None


def err_pct(pred: float, gt: float) -> float:
    if gt == 0.0:
        return 0.0 if pred == 0.0 else float("inf")
    return abs(pred - gt) / abs(gt) * 100.0


def fallback_depth_from_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= 255.0
    return 1.0 - gray


def scale_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ref_w: float,
    ref_h: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    sx = img_w / ref_w
    sy = img_h / ref_h
    nx1 = int(round(x1 * sx))
    ny1 = int(round(y1 * sy))
    nx2 = int(round(x2 * sx))
    ny2 = int(round(y2 * sy))
    nx1 = max(0, min(nx1, img_w - 1))
    ny1 = max(0, min(ny1, img_h - 1))
    nx2 = max(0, min(nx2, img_w - 1))
    ny2 = max(0, min(ny2, img_h - 1))
    if nx2 <= nx1:
        nx2 = min(nx1 + 1, img_w)
    if ny2 <= ny1:
        ny2 = min(ny1 + 1, img_h)
    return nx1, ny1, nx2, ny2


def load_gt_rows(gt_path: Path) -> List[Dict[str, Any]]:
    with gt_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("GT CSV is empty or has no header")
        need = {"frame_idx", "x1", "y1", "x2", "y2", "depth_m", "area_m2"}
        miss = need - {h.strip() for h in reader.fieldnames}
        if miss:
            raise SystemExit(f"GT CSV missing columns: {miss}")
        rows: List[Dict[str, Any]] = []
        for raw in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            rows.append(
                {
                    "frame_idx": int(float(row["frame_idx"])),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "depth_m": float(row["depth_m"]),
                    "area_m2": float(row["area_m2"]),
                }
            )
        return rows


def gt_rows_for_frame(all_rows: List[Dict[str, Any]], fid: int) -> List[Dict[str, Any]]:
    return [r for r in all_rows if r["frame_idx"] == fid]


def median_np(a: np.ndarray) -> float:
    a = np.sort(a)
    n = len(a)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(a[mid])
    return float((a[mid - 1] + a[mid]) / 2.0)


def iter_test_images(folder: Path) -> List[Tuple[Path, int]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: List[Tuple[Path, int]] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        fid = frame_id_from_stem(p.stem)
        if fid is None:
            continue
        out.append((p, fid))
    return out


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_csv).resolve()
    test_dir = Path(args.test_images).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.is_file():
        raise SystemExit(f"Missing GT: {gt_path}")
    if not test_dir.is_dir():
        raise SystemExit(f"Missing test folder: {test_dir}")

    gt_rows = load_gt_rows(gt_path)
    if not gt_rows:
        raise SystemExit("GT CSV has no data rows")

    ref_w = float(args.gt_ref_w) if args.gt_ref_w else float(max(int(r["x2"]) for r in gt_rows) + 1)
    ref_h = float(args.gt_ref_h) if args.gt_ref_h else float(max(int(r["y2"]) for r in gt_rows) + 1)
    if ref_w <= 0 or ref_h <= 0:
        raise SystemExit("Invalid --gt-ref-w / --gt-ref-h")

    cam = CameraConfig.from_yaml(args.camera_config)

    depth_est: Optional[DepthOnnxEstimator] = None
    if args.depth_onnx and Path(args.depth_onnx).is_file():
        depth_est = DepthOnnxEstimator(args.depth_onnx, input_size=args.depth_input_size)

    rows_out: List[Dict[str, Any]] = []

    for img_path, fid in iter_test_images(test_dir):
        gtf = gt_rows_for_frame(gt_rows, fid)
        if not gtf:
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] cannot read {img_path}")
            continue

        ih, iw = bgr.shape[:2]
        if depth_est is not None:
            dmap = depth_est.estimate(bgr)
        else:
            dmap = fallback_depth_from_gray(bgr)

        for row in gtf:
            gx1, gy1, gx2, gy2 = scale_xyxy(
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"],
                ref_w,
                ref_h,
                iw,
                ih,
            )
            bbox = (gx1, gy1, gx2, gy2)
            pred_d = float(estimate_depth_m(dmap, bbox, cam))
            pred_a = float(estimate_area_m2(bgr, bbox, meters_per_pixel_bev=cam.meters_per_pixel_bev))
            gt_d = float(row["depth_m"])
            gt_a = float(row["area_m2"])
            dep_pct = err_pct(pred_d, gt_d)
            area_pct = err_pct(pred_a, gt_a)

            rows_out.append(
                {
                    "image": img_path.name,
                    "frame_idx": fid,
                    "gt_ref_w": int(ref_w),
                    "gt_ref_h": int(ref_h),
                    "img_w": iw,
                    "img_h": ih,
                    "x1_scaled": gx1,
                    "y1_scaled": gy1,
                    "x2_scaled": gx2,
                    "y2_scaled": gy2,
                    "pred_depth_m": pred_d,
                    "pred_area_m2": pred_a,
                    "gt_depth_m": gt_d,
                    "gt_area_m2": gt_a,
                    "depth_abs_err_m": abs(pred_d - gt_d),
                    "area_abs_err_m2": abs(pred_a - gt_a),
                    "depth_err_pct": dep_pct if np.isfinite(dep_pct) else np.nan,
                    "area_err_pct": area_pct if np.isfinite(area_pct) else np.nan,
                }
            )

    csv_out = out_dir / "per_detection_errors.csv"
    if rows_out:
        fieldnames = list(rows_out[0].keys())

        def _cell(v: Any) -> Any:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return ""
            return v

        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_out:
                w.writerow({k: _cell(v) for k, v in r.items()})

    summary: Dict[str, Any] = {
        "gt_csv": str(gt_path),
        "test_images_dir": str(test_dir),
        "gt_ref_size_wh": [int(ref_w), int(ref_h)],
        "depth_source": args.depth_onnx if depth_est else "gray_fallback",
        "rows_evaluated": int(len(rows_out)),
    }

    if rows_out:
        d_abs = np.array([r["depth_abs_err_m"] for r in rows_out], dtype=np.float64)
        a_abs = np.array([r["area_abs_err_m2"] for r in rows_out], dtype=np.float64)
        d_pct_vals = [
            r["depth_err_pct"] for r in rows_out if isinstance(r["depth_err_pct"], (int, float)) and np.isfinite(r["depth_err_pct"])
        ]
        a_pct_vals = [
            r["area_err_pct"] for r in rows_out if isinstance(r["area_err_pct"], (int, float)) and np.isfinite(r["area_err_pct"])
        ]
        d_pct_arr = np.array(d_pct_vals, dtype=np.float64) if d_pct_vals else np.array([])
        a_pct_arr = np.array(a_pct_vals, dtype=np.float64) if a_pct_vals else np.array([])

        summary["depth_mae_m"] = float(np.mean(d_abs))
        summary["area_mae_m2"] = float(np.mean(a_abs))
        summary["depth_error_mean_percent"] = float(np.mean(d_pct_arr)) if len(d_pct_arr) else None
        summary["depth_error_median_percent"] = float(median_np(d_pct_arr)) if len(d_pct_arr) else None
        summary["area_error_mean_percent"] = float(np.mean(a_pct_arr)) if len(a_pct_arr) else None
        summary["area_error_median_percent"] = float(median_np(a_pct_arr)) if len(a_pct_arr) else None

    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {csv_out}")


if __name__ == "__main__":
    main()
