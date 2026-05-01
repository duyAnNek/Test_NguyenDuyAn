from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 checkpoint to ONNX")
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to cloned Depth Anything V2 repository",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Depth Anything V2 .pth checkpoint",
    )
    parser.add_argument(
        "--encoder",
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder size matching checkpoint",
    )
    parser.add_argument("--img-size", type=int, default=518, help="Square export input size")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", default="models/depth_anything_v2.onnx")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def encoder_config(encoder: str) -> Dict[str, object]:
    configs: Dict[str, Dict[str, object]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_path = Path(args.output).resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo dir not found: {repo_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as exc:
        raise ImportError(
            "Cannot import DepthAnythingV2. Ensure --repo-dir points to the official "
            "Depth-Anything-V2 repository root."
        ) from exc

    cfg = encoder_config(args.encoder)
    model = DepthAnythingV2(**cfg)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    model = model.to(device)

    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["depth"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "depth": {0: "batch", 1: "height", 2: "width"},
            },
        )

    print(f"Exported ONNX: {out_path}")
    print(
        "Next step: python scripts/run_realtime.py --video <video.mp4> "
        "--yolo-onnx <best.onnx> --depth-onnx "
        f"\"{out_path}\" --output-dir outputs"
    )


if __name__ == "__main__":
    main()
