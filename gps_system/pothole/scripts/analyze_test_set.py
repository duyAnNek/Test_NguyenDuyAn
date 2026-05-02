from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


BBox = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze test set predictions: distributions (conf/area/etc.) and an approximate AP@IoU metric "
            "from YOLO labels (not inferred from JPG overlays)."
        )
    )
    p.add_argument("--weights", default="runs/detect/runs/train/pothole_yolov8/weights/best.pt")
    p.add_argument("--images", default="data/images/test")
    p.add_argument("--labels", default="data/labels/test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device for analysis run, e.g. cpu | 0",
    )
    p.add_argument(
        "--map-iou",
        type=float,
        default=0.5,
        help="IoU threshold for counting TP vs FP when approximating AP@IoU",
    )
    p.add_argument("--output-dir", default="outputs/test_analysis")
    return p.parse_args()


def bbox_iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def parse_yolo_gt_xyxy(lines: Iterable[str], w: int, h: int) -> Tuple[List[int], List[BBox]]:
    classes: List[int] = []
    boxes: List[BBox] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])

        cx = cx_n * w
        cy = cy_n * h
        bw = bw_n * w
        bh = bh_n * h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        gx1 = max(0.0, min(float(w), x1))
        gy1 = max(0.0, min(float(h), y1))
        gx2 = max(0.0, min(float(w), x2))
        gy2 = max(0.0, min(float(h), y2))
        if gx2 <= gx1 or gy2 <= gy1:
            continue
        classes.append(cls_id)
        boxes.append((gx1, gy1, gx2, gy2))
    return classes, boxes


def iter_images(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def count_gt_instances(labels_dir: Path, images_dir: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for lf in sorted(labels_dir.glob("*.txt")):
        stem = lf.stem
        img_match: Path | None = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img_match = cand
                break
        if img_match is None:
            continue
        img = cv2.imread(str(img_match))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts_cls, _ = parse_yolo_gt_xyxy(lf.read_text(encoding="utf-8", errors="ignore").splitlines(), w=w, h=h)
        for c in gts_cls:
            counts[c] = counts.get(c, 0) + 1
    return counts


def compute_ap_single_class(gt_pos: int, preds: pd.DataFrame) -> float:
    if gt_pos <= 0:
        return float("nan")
    if preds.empty:
        return 0.0

    preds = preds.copy()
    preds["tp_bool"] = preds["tp"].astype(bool)

    order = preds["conf"].to_numpy(dtype=np.float64).argsort()[::-1]
    tp = preds["tp_bool"].to_numpy(dtype=np.float64)[order]
    fp = (~preds["tp_bool"]).to_numpy(dtype=np.float64)[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / gt_pos
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # VOC-style interpolated precision envelope along recall
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return ap


def safe_seaborn_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.exists():
        raise FileNotFoundError(str(images_dir))
    if not labels_dir.exists():
        raise FileNotFoundError(str(labels_dir))

    model = YOLO(args.weights)

    det_rows: List[Dict[str, object]] = []
    map_rows: List[Dict[str, object]] = []
    counts_per_image: List[Dict[str, object]] = []

    img_paths = list(iter_images(images_dir))

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        gt_path = labels_dir / f"{img_path.stem}.txt"
        gt_lines = gt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if gt_path.exists() else []
        gts_cls, gts_xyxy = parse_yolo_gt_xyxy(gt_lines, w=w, h=h)

        res = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        preds_tmp: List[Tuple[int, BBox, float]] = []
        boxes = res.boxes
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for i_det in range(int(xyxy.shape[0])):
                x1, y1, x2, y2 = map(float, xyxy[i_det].tolist())
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_pred = int(classes[i_det])
                conf = float(confs[i_det])
                bw_det = max(1.0, x2 - x1)
                bh_det = max(1.0, y2 - y1)
                cx_det = (x1 + x2) / 2.0 / max(w, 1)
                cy_det = (y1 + y2) / 2.0 / max(h, 1)

                det_rows.append(
                    {
                        "image": img_path.name,
                        "width": w,
                        "height": h,
                        "cls_pred": cls_pred,
                        "conf": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "area_frac": float((bw_det * bh_det) / max(w * h, 1)),
                        "aspect": float(bw_det / bh_det),
                        "cx_frac": float(cx_det),
                        "cy_frac": float(cy_det),
                        "gt_count_img": len(gts_xyxy),
                    }
                )
                preds_tmp.append((cls_pred, (x1, y1, x2, y2), conf))

        # Greedy matching within image (confidence order) -> TP labels for approximation
        used_gt = np.zeros(len(gts_xyxy), dtype=bool)
        preds_tmp.sort(key=lambda t: (-t[2], t[0]))
        for cls_pred, pred_box, conf in preds_tmp:
            tp = False
            best_idx = -1
            best_iou = 0.0
            for gi, (cls_gt, gb) in enumerate(zip(gts_cls, gts_xyxy)):
                if used_gt[gi]:
                    continue
                if cls_gt != cls_pred:
                    continue
                iou = bbox_iou_xyxy(pred_box, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx >= 0 and best_iou >= args.map_iou:
                used_gt[best_idx] = True
                tp = True

            map_rows.append(
                {
                    "image": img_path.name,
                    "cls": cls_pred,
                    "conf": conf,
                    "tp": tp,
                    "iou": float(best_iou) if tp else float(best_iou),
                }
            )

        counts_per_image.append({"image": img_path.name, "width": w, "height": h, "n_det": len(preds_tmp)})

    df_det = pd.DataFrame(det_rows)
    df_img = pd.DataFrame(counts_per_image)

    df_det_path = out_dir / "detections_per_box.csv"
    merged_path = out_dir / "detections_per_image.csv"
    df_det.to_csv(df_det_path, index=False)
    df_img.to_csv(merged_path, index=False)

    safe_seaborn_style()

    if not df_det.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(df_det["conf"], bins=30, color="#2563eb", edgecolor="#1f2937", alpha=0.85)
        plt.title("Confidence distribution (test)")
        plt.xlabel("confidence")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "dist_conf.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist(df_det["area_frac"], bins=30, color="#059669", edgecolor="#1f2937", alpha=0.85)
        plt.title("BBox area fraction vs image area")
        plt.xlabel("(w_box * h_box) / (W*H)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "dist_area_frac.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.scatter(df_det["cx_frac"], df_det["cy_frac"], c=df_det["conf"], s=14, cmap="viridis", alpha=0.6)
        plt.colorbar(label="confidence")
        plt.title("BBox center distribution")
        plt.xlabel("cx / W")
        plt.ylabel("cy / H")
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_center_conf.png", dpi=160)
        plt.close()

    # Detections-per-image histogram and top-ish bar chart
    if not df_img.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(df_img["n_det"], bins=range(0, int(df_img["n_det"].max()) + 2), color="#f59e0b", alpha=0.9)
        plt.title("Distribution of detection counts per image")
        plt.xlabel("n_detections")
        plt.ylabel("images")
        plt.tight_layout()
        plt.savefig(out_dir / "dist_detections_per_image.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 4))
        top = df_img.sort_values(["n_det", "image"], ascending=[False, True]).head(25)
        plt.bar(top["image"], top["n_det"], color="#fb7185")
        plt.xticks(rotation=90)
        plt.title("Top images by detection count (top 25)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_top_counts.png", dpi=160)
        plt.close()

    gt_counts = count_gt_instances(labels_dir=labels_dir, images_dir=images_dir)

    df_map = pd.DataFrame(map_rows)
    ap_per_class: Dict[str, float] = {}
    for cls_id in sorted(gt_counts.keys()):
        subset = df_map[df_map["cls"] == cls_id]
        preds_df = subset[["conf", "tp"]].copy()

        gp = int(gt_counts.get(cls_id, 0))
        ap_val = compute_ap_single_class(gt_pos=gp, preds=preds_df)
        ap_per_class[f"class_{cls_id}"] = float(ap_val)

    ap_values = list(ap_per_class.values())
    mean_ap = float(np.nanmean(ap_values)) if ap_values else float("nan")

    summary = {
        "important_note": (
            "These charts are computed from RAW predictions over data/images/test, not decoded from JPG overlays "
            "in runs/detect/outputs/test_predictions.\n\n"
            f"Approx AP@IoU={args.map_iou} uses per-image greedy matching (class must match); it is NOT identical "
            "to Ultralytics mAP aggregation, which is multi-threshold PR over the whole dataset split."
        ),
        "weights": str(Path(args.weights).resolve()),
        "images": str(images_dir.resolve()),
        "labels": str(labels_dir.resolve()),
        "prediction_conf_thresh": args.conf,
        "prediction_iou_thresh": args.iou,
        "iou_for_tp_approx": args.map_iou,
        "mean_ap_approx": mean_ap,
        "per_class_ap_approx": ap_per_class,
        "per_class_gt_count": {f"class_{k}": v for k, v in sorted(gt_counts.items())},
        "ultralytics_reference_from_outputs_detection_metrics_json": None,
    }

    ref_path = Path("outputs/detection_metrics.json")
    if ref_path.exists():
        try:
            summary["ultralytics_reference_from_outputs_detection_metrics_json"] = json.loads(
                ref_path.read_text(encoding="utf-8")
            )
        except Exception:
            summary["ultralytics_reference_from_outputs_detection_metrics_json"] = "unreadable"

    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if ap_values:
        names = list(ap_per_class.keys())
        vals = [ap_per_class[k] for k in names]
        plt.figure(figsize=(8, 4))
        plt.bar(names, np.nan_to_num(vals, nan=0.0), color="#9333ea")
        plt.title(f"Approx AP@IoU={args.map_iou} per class")
        plt.ylabel("AP (approx)")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "bar_ap_per_class_approx.png", dpi=160)
        plt.close()

    print(f"Wrote: {df_det_path}")
    print(f"Wrote: {merged_path}")
    print(f"Wrote charts + summary into: {out_dir}")
    print(json.dumps({k: summary[k] for k in ["mean_ap_approx", "per_class_ap_approx"] if k in summary}, indent=2))


if __name__ == "__main__":
    main()
