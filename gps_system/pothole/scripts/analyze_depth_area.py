"""
Phân tích độ sâu (heuristic từ depth map) và diện tích BEV (IPM gần đúng) cho từng ổ được YOLO detect.

Dùng chung logic với `src/pipeline.PotholePipeline`: `estimate_depth_m`, `estimate_area_m2`, `classify_severity`.

Usage:
  cd pothole
  python scripts/analyze_depth_area.py ^
    --source data/images/test ^
    --yolo-onnx runs/detect/runs/pothole/train_runs/weights/best.onnx ^
    --depth-onnx path/to/depth_anything.onnx ^
    --camera-config configs/camera_config.yaml ^
    --severity-config configs/severity_rules.yaml ^
    --data data/dataset.yaml ^
    --output-dir outputs/depth_area_analysis

Không có --depth-onnx: depth tương đối từ cờ xám (fallback) giống pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from src.models import DepthOnnxEstimator, YoloOnnxDetector  # noqa: E402
from src.severity import SeverityThreshold, classify_severity  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze pothole depth_m and area_m2 per detection")
    p.add_argument("--source", required=True, help="Ảnh đơn, thư mục ảnh, hoặc a.jpg,b.jpg")
    p.add_argument(
        "--yolo-onnx",
        default=str(PROJECT_ROOT / "runs/detect/runs/pothole/train_runs/weights/best.onnx"),
    )
    p.add_argument("--depth-onnx", default=None, help="ONNX monocular depth (optional)")
    p.add_argument("--depth-input-size", type=int, default=None)
    p.add_argument("--camera-config", default=str(PROJECT_ROOT / "configs/camera_config.yaml"))
    p.add_argument("--severity-config", default=str(PROJECT_ROOT / "configs/severity_rules.yaml"))
    p.add_argument("--data", default=str(PROJECT_ROOT / "data/dataset.yaml"), help="Tên lớp (names)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/depth_area_analysis"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--viz", action="store_true", help="Lưu ảnh có box + depth/area/severity")
    return p.parse_args()


def load_class_names(data_yaml: Path) -> Dict[int, str]:
    if yaml is None or not data_yaml.is_file():
        return {}
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def iter_images(source: str) -> List[Path]:
    p = Path(source.strip())
    if p.is_file():
        return [p.resolve()]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(x for x in p.iterdir() if x.suffix.lower() in exts)
    parts = [Path(s.strip()) for s in source.split(",") if s.strip()]
    return [x.resolve() for x in parts if x.is_file()]


def fallback_depth_from_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= 255.0
    return 1.0 - gray


def try_matplotlib_save_scatter(areas: List[float], depths: List[float], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[warn] matplotlib not installed; skip scatter plot")
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(areas, depths, alpha=0.5, s=18)
    plt.xlabel("area_m2 (BEV heuristic)")
    plt.ylabel("depth_m (relative heuristic)")
    plt.title("Depth vs area (all detections)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def try_hist(vals: List[float], title: str, xlabel: str, out_png: Path) -> None:
    if not vals:
        return
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=min(40, max(8, len(vals) // 3)), color="#2563eb", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_cfg = CameraConfig.from_yaml(args.camera_config)
    sev_cfg = SeverityThreshold.from_yaml(args.severity_config)
    names = load_class_names(Path(args.data))

    detector = YoloOnnxDetector(args.yolo_onnx, conf_thres=args.conf, iou_thres=args.iou)
    depth_est: Optional[DepthOnnxEstimator] = (
        DepthOnnxEstimator(args.depth_onnx, input_size=args.depth_input_size)
        if args.depth_onnx and Path(args.depth_onnx).is_file()
        else None
    )

    if args.depth_onnx and depth_est is None:
        print(f"[warn] depth onnx not found: {args.depth_onnx}, using gray fallback")

    images = iter_images(args.source)
    if not images:
        raise SystemExit(f"No images under source: {args.source}")

    rows: List[Dict[str, Any]] = []
    all_depth: List[float] = []
    all_area: List[float] = []

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] cannot read {img_path}")
            continue

        if depth_est is not None:
            dmap = depth_est.estimate(bgr)
        else:
            dmap = fallback_depth_from_gray(bgr)

        dets = detector.infer(bgr)
        vis = bgr.copy() if args.viz else None

        for j, det in enumerate(dets):
            area_m2 = estimate_area_m2(
                bgr,
                det.bbox_xyxy,
                meters_per_pixel_bev=cam_cfg.meters_per_pixel_bev,
            )
            depth_m = estimate_depth_m(dmap, det.bbox_xyxy, cam_cfg)
            sev = classify_severity(depth_m, area_m2, sev_cfg)
            cls_name = names.get(det.cls_id, str(det.cls_id))
            x1, y1, x2, y2 = det.bbox_xyxy

            try:
                img_cell = str(img_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
            except ValueError:
                img_cell = str(img_path)

            rows.append(
                {
                    "image": img_cell,
                    "det_idx": j,
                    "cls_id": det.cls_id,
                    "cls_name": cls_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": round(det.conf, 6),
                    "depth_m": round(depth_m, 6),
                    "area_m2": round(area_m2, 6),
                    "severity": sev,
                }
            )
            all_depth.append(depth_m)
            all_area.append(area_m2)

            if vis is not None:
                color = {
                    "minor": (0, 255, 0),
                    "moderate": (0, 255, 255),
                    "severe": (0, 0, 255),
                }.get(sev, (255, 255, 255))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                lbl = f"{cls_name} {sev} d={depth_m:.3f}m a={area_m2:.3f}m2 c={det.conf:.2f}"
                cv2.putText(
                    vis,
                    lbl,
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if vis is not None:
            viz_dir = out_dir / "viz"
            viz_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_analysis.jpg"), vis)

    csv_path = out_dir / "depth_area_detections.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        csv_path.write_text("", encoding="utf-8")

    def _stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {}
        arr = np.array(xs, dtype=np.float64)
        return {
            "count": float(len(xs)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p50": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    summary: Dict[str, Any] = {
        "images_processed": len({r["image"] for r in rows}) if rows else 0,
        "num_detections": len(rows),
        "depth_source": args.depth_onnx if depth_est else "gray_fallback",
        "camera_config": args.camera_config,
        "severity_config": args.severity_config,
        "depth_m_stats": _stats(all_depth),
        "area_m2_stats": _stats(all_area),
        "severity_counts": {},
    }
    for r in rows:
        s = r["severity"]
        summary["severity_counts"][s] = summary["severity_counts"].get(s, 0) + 1

    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_plots and rows:
        try_hist(all_depth, "depth_m distribution", "depth_m (heuristic)", out_dir / "hist_depth_m.png")
        try_hist(all_area, "area_m2 distribution", "area_m2 (BEV heuristic)", out_dir / "hist_area_m2.png")
        try_matplotlib_save_scatter(all_area, all_depth, out_dir / "scatter_depth_vs_area.png")
        # scatter uses areas as x - fix order in function - I used areas x, depths y - good

    print(json.dumps(summary, indent=2))
    print(f"Wrote CSV: {csv_path.resolve()}")
    print(f"Wrote summary: {(out_dir / 'analysis_summary.json').resolve()}")


if __name__ == "__main__":
    main()
