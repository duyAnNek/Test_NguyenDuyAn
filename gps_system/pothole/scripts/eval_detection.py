from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def split_images_dir(data_yaml: Path, split: str) -> Path:
    """Resolve folder of images for a split (matches Ultralytics dataset.yaml layout)."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = cfg.get("path")
    if not root:
        root = data_yaml.parent
    root = Path(root).expanduser().resolve()
    rel = cfg.get(split)
    if rel is None:
        raise ValueError(f"dataset.yaml has no '{split}' key")
    if isinstance(rel, list):
        rel = rel[0]
    return (root / str(rel)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pothole detection on test set")
    parser.add_argument(
        "--weights",
        default="runs/detect/runs/train/pothole_yolov8/weights/best.pt",
    )
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device for val(), e.g. "cpu", "0" (GPU)',
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable Ultralytics val curves and first-batch preview images.",
    )
    parser.add_argument(
        "--save-all-images",
        action="store_true",
        help="Export every image in the split with boxes (runs predict(save=True)); not limited to 3 val batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_path = Path(args.data).resolve()

    model = YOLO(args.weights)
    # Plots=True: metrics curves + val_batch*_labels/pred.jpg for the FIRST 3 batches only (Ultralytics default).
    # project/name puts those under output_dir so they are easy to find.
    metrics = model.val(
        data=str(data_yaml_path),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=0.6,
        verbose=False,
        plots=not args.no_plots,
        project=str(output_dir.resolve()),
        name="ultralytics_val",
        exist_ok=True,
    )

    if args.save_all_images:
        src = split_images_dir(data_yaml_path, args.split)
        if not src.exists():
            raise FileNotFoundError(f"Split image folder not found: {src}")
        model.predict(
            source=str(src),
            imgsz=args.imgsz,
            device=args.device,
            conf=0.001,
            iou=0.6,
            save=True,
            project=str(output_dir.resolve()),
            name=f"{args.split}_predictions",
            exist_ok=True,
            verbose=False,
        )

    results = {
        "weights": str(Path(args.weights).resolve()),
        "device": args.device,
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    (output_dir / "detection_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_dir / "map50.txt").write_text(
        f"mAP@0.5: {results['mAP@0.5']:.6f}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))

    val_art = output_dir / "ultralytics_val"
    if not args.no_plots:
        print(
            f"Val plots and first 3 batch previews (val_batch*_pred.jpg): {val_art.resolve()} "
            "(Ultralytics only renders the first 3 val batches by design.)"
        )
    if args.save_all_images:
        pred_dir = output_dir / f"{args.split}_predictions"
        print(f"All split images with boxes saved under: {pred_dir.resolve()}")


if __name__ == "__main__":
    main()
