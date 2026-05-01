"""
Bước đầu với ONNX Runtime (Python): in cấu hình tensor, chạy 1 inference (ảo hoặc từ ảnh).
Không thay Ultralytics decoder; chỉ để chứng minh ONNX + ORT hoạt động và xem shape output.

Usage:
  python scripts/smoke_onnx_runtime.py --model runs/detect/runs/pothole/train_runs/weights/best.onnx
  python scripts/smoke_onnx_runtime.py --model path/to/model.onnx --image data/images/test/img-01.jpg --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test ONNX Runtime inference")
    p.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "runs/detect/runs/pothole/train_runs/weights/best.onnx"),
        help="Đường dẫn file .onnx",
    )
    p.add_argument("--image", default=None, help="Ảnh RGB/BGR để preprocess (opencv). Mặc định chạy input toàn số 0.")
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="cuda cần cài onnxruntime-gpu và driver CUDA đúng phiên bản",
    )
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def fix_dynamic_shape(shape: list, imgsz: int) -> list[int]:
    fixed: list[int] = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            fixed.append(dim)
            continue
        if i == 0:
            fixed.append(1)
        elif i == 1:
            fixed.append(3)
        elif i == 2 or i == 3:
            fixed.append(imgsz)
        else:
            fixed.append(1)
    return fixed


def preprocess_bgr_resize_center(img_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize cứng về imgsz x imgsz, BGR->RGB, HWC -> NCHW, float32 [0,1]."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit("Install opencv-python: pip install opencv-python") from e

    resized = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    return batch


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.model).resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    if args.device == "cuda":
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit("Install onnxruntime-gpu for CUDA inference") from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        import onnxruntime as ort  # noqa: PLC0415

        providers = ["CPUExecutionProvider"]

    print(f"[1] ONNX file: {onnx_path}")
    print(f"[1] ONNX Runtime providers: {providers}")

    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    print("\n[2] Inputs:")
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        print(f"    name={inp.name!r}  type={inp.type}  shape={inp.shape}")
        shp = fix_dynamic_shape(list(inp.shape), args.imgsz)

        if args.image:
            p = Path(args.image)
            if not p.is_file():
                raise SystemExit(f"Image not found: {p}")
            import cv2  # noqa: PLC0415

            bgr = cv2.imread(str(p))
            if bgr is None:
                raise SystemExit(f"opencv could not read image: {p}")
            arr = preprocess_bgr_resize_center(bgr, args.imgsz)
            if tuple(arr.shape) != tuple(shp):
                raise SystemExit(f"preprocess shape {arr.shape} != model {shp}; fix --imgsz or model.")
            feeds[inp.name] = arr.astype(np.float32)
        else:
            feeds[inp.name] = np.zeros(shp, dtype=np.float32)
            print(f"    >> dummy zeros shape={tuple(shp)}")

    print("\n[3] Outputs:")
    for out in sess.get_outputs():
        print(f"    name={out.name!r}  type={out.type}  shape={out.shape}")

    print("\n[4] Inference...")
    outs = sess.run(None, feeds)
    for idx, tensor in enumerate(outs):
        t = np.asarray(tensor)
        print(f"    output[{idx}] shape={t.shape} dtype={t.dtype} min={np.nanmin(t):.4g} max={np.nanmax(t):.4g}")

    print("\n[DONE] Session ran successfully; ONNX Runtime is working.")
    print("For bbox like Ultralytics, add YOLOv8 decode + NMS (not in this script).")


if __name__ == "__main__":
    main()
