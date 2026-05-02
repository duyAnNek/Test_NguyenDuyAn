from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pt file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    print("ONNX export completed.")


if __name__ == "__main__":
    main()
