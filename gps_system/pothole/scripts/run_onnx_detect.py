"""
ONNX Runtime + YOLOv8 Ultralytics decode (LetterBox like predict) + NMS + optional save.

Reference pipeline for comparing with YOLO().predict(), and for porting logic to Android (ORT + Kotlin).

Examples:
  python scripts/run_onnx_detect.py ^
    --model runs/detect/runs/pothole/train_runs/weights/best.onnx ^
    --source data/images/test --data data/dataset.yaml --save-dir outputs/onnx_infer

  python scripts/run_onnx_detect.py --model best.onnx --source data/images/test/img-1.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils import nms  # noqa: E402
from ultralytics.utils.ops import scale_boxes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 ONNX inference with ONNX Runtime")
    p.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "runs/detect/runs/pothole/train_runs/weights/best.onnx"),
        help="Path to exported .onnx",
    )
    p.add_argument("--source", required=True, help="Image path, folder, or comma-separated image paths")
    p.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data/dataset.yaml"),
        help="dataset.yaml for class names",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ORT execution providers")
    p.add_argument("--save-dir", default=None, help="Write annotated JPGs here")
    p.add_argument("--show", action="store_true", help="cv2.imshow (needs GUI)")
    return p.parse_args()


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.is_file():
        return {}
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    nc = int(cfg.get("nc", 0))
    return {i: f"class{i}" for i in range(nc)}


def list_sources(source: str) -> list[Path]:
    raw = Path(source.strip())
    if raw.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(p for p in raw.iterdir() if p.suffix.lower() in exts)
    parts = [Path(s.strip()) for s in source.split(",") if s.strip()]
    if not parts:
        raise SystemExit("No paths in --source")
    missing = [p for p in parts if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing file(s): {missing}")
    return parts


def letterbox_blob(bgr_uint8_hwc: np.ndarray, imgsz: int) -> tuple[np.ndarray, tuple[int, int]]:
    """LetterBox + RGB + NCHW float [0,1], shape (1,3,h,w). Returns (blob, resized_hw_for_scale_boxes)."""
    lb_hwc = LetterBox((imgsz, imgsz), auto=False, stride=32)(image=bgr_uint8_hwc)
    rgb = lb_hwc[..., ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    blob = (chw[None] / 255.0).astype(np.float32)
    h, w = blob.shape[-2:]
    return blob, (int(h), int(w))


def color_for(cls_id: int) -> tuple[int, int, int]:
    r = ((cls_id * 71 + 137) % 200) + 55
    g = ((cls_id * 17 + 29) % 200) + 55
    b = ((cls_id * 113 + 19) % 200) + 55
    return (r, g, b)


def draw_detections(img_bgr: np.ndarray, dets_np: np.ndarray, names: dict[int, str]) -> None:
    for row in dets_np:
        x1, y1, x2, y2, conf, cl = row[:6].tolist()
        cid = int(cl)
        label = names.get(cid, str(cid))
        text = f"{label} {conf:.2f}"
        c = color_for(cid)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img_bgr, p1, p2, c, 2)
        cv2.putText(img_bgr, text, (p1[0], max(18, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.model).resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    yaml_path = Path(args.data).resolve()
    names = load_class_names(yaml_path)

    if args.device == "cuda":
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit("Install onnxruntime-gpu for --device cuda") from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        import onnxruntime as ort  # noqa: PLC0415

        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    in_meta = session.get_inputs()[0]
    in_name = in_meta.name
    out_metas = session.get_outputs()
    shp = list(in_meta.shape)
    if (
        len(shp) == 4
        and isinstance(shp[2], int)
        and shp[2] > 0
        and isinstance(shp[3], int)
        and shp[3] > 0
    ):
        if shp[2] != args.imgsz:
            print(f"[warn] ONNX input H,W=({shp[2]},{shp[3]}); aligning --imgsz to {shp[2]}")
            args.imgsz = int(shp[2])

    sources = list_sources(args.source)
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    t_infer = 0.0
    n_run = 0

    for src in sources:
        im0 = cv2.imread(str(src))
        if im0 is None:
            print(f"[skip] cannot read {src}")
            continue

        blob, proc_hw = letterbox_blob(im0, args.imgsz)
        if tuple(blob.shape) != (1, 3, args.imgsz, args.imgsz):
            raise SystemExit(f"blob shape {blob.shape} != (1,3,{args.imgsz},{args.imgsz})")

        t0 = time.perf_counter()
        raw = session.run([o.name for o in out_metas], {in_name: blob})
        t_infer += time.perf_counter() - t0
        n_run += 1

        pred = torch.from_numpy(np.asarray(raw[0], dtype=np.float32))
        if pred.ndim != 3:
            raise SystemExit(f"Unexpected output rank {pred.ndim}; expected (N, 4+nc, anchors)")

        # End-to-end NMS in graph: (B, N, 6)
        if pred.shape[-1] == 6:
            dets = nms.non_max_suppression(
                pred,
                args.conf,
                args.iou,
                max_det=args.max_det,
                end2end=True,
            )[0]
        else:
            nc = pred.shape[1] - 4
            if not names and nc > 0:
                names = {i: f"class{i}" for i in range(nc)}
            dets = nms.non_max_suppression(
                pred,
                args.conf,
                args.iou,
                max_det=args.max_det,
                nc=nc,
                end2end=False,
            )[0]

        if dets.shape[0]:
            dets = scale_boxes(proc_hw, dets, im0.shape[:2])

        dets_np = dets.cpu().numpy()
        print(f"{src.name}: {dets_np.shape[0]} detections")

        vis = im0.copy()
        if dets_np.shape[0]:
            draw_detections(vis, dets_np, names)

        if save_dir is not None:
            out_path = save_dir / f"{src.stem}_onnx.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  saved {out_path.as_posix()}")

        if args.show:
            cv2.imshow("onnx", vis)
            cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    if n_run:
        ms = 1000.0 * t_infer / n_run
        print(f"Avg inference (ORT only, {n_run} image(s)): {ms:.1f} ms")


if __name__ == "__main__":
    main()
