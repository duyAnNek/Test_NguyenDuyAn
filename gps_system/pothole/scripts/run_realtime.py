from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CameraConfig
from src.pipeline import PotholePipeline
from src.severity import SeverityThreshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time pothole pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo-onnx", required=True)
    parser.add_argument("--depth-onnx", default=None)
    parser.add_argument("--camera-config", default="configs/camera_config.yaml")
    parser.add_argument("--severity-config", default="configs/severity_rules.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument(
        "--depth-interval",
        type=int,
        default=3,
        help="Estimate depth every N frames and reuse previous depth in-between",
    )
    parser.add_argument(
        "--depth-input-size",
        type=int,
        default=384,
        help="Depth model input size (smaller = faster, less precise)",
    )
    parser.add_argument(
        "--metrics-json",
        default="outputs/detection_metrics.json",
        help="Path to detection metrics JSON to display reference mAP@0.5 on video",
    )
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Lưu video đã overlay nhận diện. Không kèm đường dẫn: "
            "<output-dir>/<tên-video>_detected.mp4 ; có thể dùng --no-visualize chỉ để encode nhanh hơn"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.yolo_onnx).exists():
        raise FileNotFoundError(f"YOLO ONNX not found: {args.yolo_onnx}")

    depth_onnx_path = args.depth_onnx
    if depth_onnx_path and not Path(depth_onnx_path).exists():
        print(f"[WARN] Depth ONNX not found: {depth_onnx_path}. Fallback to relative depth.")
        depth_onnx_path = None

    camera_cfg = CameraConfig.from_yaml(args.camera_config)
    severity_cfg = SeverityThreshold.from_yaml(args.severity_config)
    map50_reference = None
    metrics_path = Path(args.metrics_json)
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            map50_reference = float(data.get("mAP@0.5"))
        except Exception:
            map50_reference = None

    pipeline = PotholePipeline(
        yolo_onnx_path=args.yolo_onnx,
        depth_onnx_path=depth_onnx_path,
        camera_cfg=camera_cfg,
        severity_cfg=severity_cfg,
        conf_thres=args.conf_thres,
        depth_interval=args.depth_interval,
        depth_input_size=args.depth_input_size if depth_onnx_path else None,
    )

    save_video_path: Optional[str] = None
    if args.save_video is not None:
        if args.save_video == "":
            stem = Path(args.video).stem
            save_video_path = str(Path(args.output_dir) / f"{stem}_detected.mp4")
        else:
            save_video_path = str(Path(args.save_video))

    pipeline.run_video(
        video_path=args.video,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        map50_reference=map50_reference,
        save_video_path=save_video_path,
    )
    print(f"Completed. Reports saved in: {args.output_dir}")
    if save_video_path is not None:
        print(f"Annotated video: {Path(save_video_path).as_posix()}")


if __name__ == "__main__":
    main()
