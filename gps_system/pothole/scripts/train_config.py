from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

BACKBONE_COMPARISON = [
    {
        "model": "yolov8n.pt",
        "params_M": 3.2,
        "gflops": 8.7,
        "coco_map50": 52.6,
        "coco_map50_95": 37.3,
        "cpu_ms": 80,
        "note": "Nhanh nhat, phu hop CPU real-time",
    },
]

# NOTE: Only keys that Ultralytics `YOLO.train()` understands should live here.
TRAIN_CONFIG: Dict[str, Any] = {
    "model": "yolov8n.pt",
    "data": "data/dataset.yaml",
    "epochs": 100,
    "patience": 10,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "device": "cpu",
    # Windows + PyTorch multiprocessing: if dataloader errors, set workers=0.
    "workers": 4,
    # Avoid Ultralytics doubling path as runs/detect/runs/detect/... (task prefix + project).
    "project": "runs/pothole",
    "name": "train_runs",
    "exist_ok": True,
    "save": True,
    "plots": True,
}

AUGMENT_CONFIG: Dict[str, Any] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,
    "erasing": 0.0,
    "fraction": 1.0,
}


def print_backbone_comparison() -> None:
    header = (
        f"{'Model':<12s} {'Params':>8s} {'GFLOPs':>8s} "
        f"{'mAP@50':>8s} {'mAP@50-95':>10s} {'CPU(ms)':>8s}  Note"
    )
    sep = "-" * 100

    print("\n" + sep)
    print("  YOLOv8 Backbone Comparison (COCO val2017)")
    print(sep)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for m in BACKBONE_COMPARISON:
        row = (
            f"  {m['model']:<12s} {m['params_M']:>7.1f}M {m['gflops']:>7.1f} "
            f"{m['coco_map50']:>7.1f}% {m['coco_map50_95']:>9.1f}% "
            f"{m['cpu_ms']:>7d}   {m['note']}"
        )
        print(row)

    print(sep)


def print_train_config() -> None:
    print("\n" + "=" * 60)
    print("  Training Configuration")
    print("=" * 60)

    sections = {
        "Model & Data": ["model", "data"],
        "Schedule": ["epochs", "patience", "batch", "imgsz"],
        "Learning Rate": ["lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"],
        "Loss Weights": ["box", "cls", "dfl"],
        "Hardware": ["device", "workers"],
        "Output": ["project", "name", "exist_ok", "save", "plots"],
    }

    for section_name, keys in sections.items():
        print(f"\n  --- {section_name} ---")
        for k in keys:
            if k in TRAIN_CONFIG:
                print(f"    {k:<20s}: {TRAIN_CONFIG[k]}")

    print()


def print_augment_config() -> None:
    print("\n" + "=" * 60)
    print("  Augmentation Configuration")
    print("=" * 60)

    sections = {
        "Color / Light": ["hsv_h", "hsv_s", "hsv_v"],
        "Geometry": ["degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud"],
        "Advanced": ["mosaic", "mixup", "copy_paste", "erasing", "fraction"],
    }

    for section_name, keys in sections.items():
        print(f"\n  --- {section_name} ---")
        for k in keys:
            if k in AUGMENT_CONFIG:
                print(f"    {k:<20s}: {AUGMENT_CONFIG[k]}")

    print()


def get_full_config() -> Dict[str, Any]:
    """Display-only merged config."""
    config: Dict[str, Any] = {}
    config.update(TRAIN_CONFIG)
    config.update(AUGMENT_CONFIG)
    return config


def build_train_kwargs(
    data_yaml: str,
    device_override: str | None,
    model_override: str | None,
) -> Dict[str, Any]:
    train_cfg = dict(TRAIN_CONFIG)
    if model_override:
        train_cfg["model"] = model_override
    if data_yaml:
        train_cfg["data"] = data_yaml
    if device_override is not None:
        train_cfg["device"] = device_override

    # Pull model file out of train kwargs (not a train() arg)
    init_weights = train_cfg.pop("model")

    kwargs: Dict[str, Any] = dict(train_cfg)
    kwargs.update(AUGMENT_CONFIG)
    return {"init_weights": init_weights, "train": kwargs}


def run_training(
    device_override: str | None = None,
    model_override: str | None = None,
    data_yaml: str | None = None,
) -> object | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Can cai dat ultralytics: pip install ultralytics")
        return None

    bundle = build_train_kwargs(
        data_yaml=data_yaml or str(TRAIN_CONFIG["data"]),
        device_override=device_override,
        model_override=model_override,
    )
    init_weights = bundle["init_weights"]
    train_kwargs = bundle["train"]

    print(f"\n[Train] Loading model: {init_weights}")
    model = YOLO(init_weights)

    print(f"[Train] Starting training on device: {train_kwargs.get('device')}")
    print(f"[Train] Data: {train_kwargs.get('data')}")
    print(f"[Train] Epochs: {train_kwargs.get('epochs')} (patience={train_kwargs.get('patience')})")

    results = model.train(**train_kwargs)

    print("\n[Train] Exporting best model to ONNX...")
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        onnx_path = best_model.export(format="onnx", imgsz=int(train_kwargs.get("imgsz", 640)), dynamic=False)
        print(f"[Train] ONNX model saved: {onnx_path}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        dst = models_dir / "yolov8_pothole.onnx"
        shutil.copy2(onnx_path, dst)
        print(f"[Train] Copied to: {dst}")
    else:
        print(f"[Train] Warning: best.pt not found at {best_pt}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 Training Config for Pothole Detection (repo-aware)")
    parser.add_argument("--show", action="store_true", help="In toan bo cau hinh")
    parser.add_argument("--compare", action="store_true", help="In bang so sanh backbone")
    parser.add_argument("--train", action="store_true", help="Chay training")
    parser.add_argument("--device", type=str, default=None, help="Override device (0 = GPU, cpu = CPU)")
    parser.add_argument("--model", type=str, default=None, help="Override backbone (vd: yolov8s.pt)")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override dataset yaml (default: data/dataset.yaml)",
    )
    args = parser.parse_args()

    if args.compare or args.show:
        print_backbone_comparison()

    if args.show:
        print_train_config()
        print_augment_config()
        # Also show merged display config
        print("\n" + "=" * 60)
        print("  Merged display config (train + augment)")
        print("=" * 60)
        for k, v in get_full_config().items():
            print(f"    {k:<22s}: {v}")
        print()

    if args.train:
        print_backbone_comparison()
        print_train_config()
        print_augment_config()
        run_training(device_override=args.device, model_override=args.model, data_yaml=args.data)

    if not (args.show or args.compare or args.train):
        print("Usage:")
        print("  python scripts/train_config.py --show")
        print("  python scripts/train_config.py --compare")
        print("  python scripts/train_config.py --train")
        print("  python scripts/train_config.py --train --device cpu")
        print("  python scripts/train_config.py --train --device 0")
        print("  python scripts/train_config.py --train --model yolov8s.pt")
        print("  python scripts/train_config.py --train --data data/dataset.yaml")


if __name__ == "__main__":
    main()
