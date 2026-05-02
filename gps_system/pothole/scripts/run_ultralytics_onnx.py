from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO ONNX inference on an image.")
    parser.add_argument("--model", default="best.onnx", help="Path to ONNX model file.")
    parser.add_argument("--source", default="test.jpg", help="Path to input image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", default="cpu", help='Device, e.g. "cpu" or "0" (GPU).')
    parser.add_argument("--save", action="store_true", help="Save prediction image to runs/detect/predict*.")
    parser.add_argument("--show", action="store_true", help="Show prediction window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path.resolve()}")

    model = YOLO(str(model_path))
    results = model(
        str(source_path),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        verbose=False,
    )

    if args.show:
        for r in results:
            r.show()

    print(f"Done. Detections: {sum(len(r.boxes) for r in results)}")


if __name__ == "__main__":
    main()
