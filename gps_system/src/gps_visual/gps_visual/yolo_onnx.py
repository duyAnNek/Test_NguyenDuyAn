"""YOLO-like ONNX detector (CPU) — optional for ghost vs detection semantic IoU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


@dataclass
class DetBox:
    xyxy: Tuple[int, int, int, int]
    conf: float
    cls_id: int


class YoloOnnxDetector:
    """Minimal Ultralytics-export style decode (single output tensor)."""

    def __init__(self, onnx_path: str, conf_thres: float = 0.35, iou_thres: float = 0.45) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime required")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.session.get_inputs()[0].name
        self.input_hw = (640, 640)
        shp = self.session.get_inputs()[0].shape
        if len(shp) == 4 and isinstance(shp[2], int):
            self.input_hw = (int(shp[2]), int(shp[3]) if isinstance(shp[3], int) else int(shp[2]))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def infer(self, frame_bgr: np.ndarray) -> List[DetBox]:
        h0, w0 = frame_bgr.shape[:2]
        ih, iw = self.input_hw
        img = cv2.resize(frame_bgr, (iw, ih))
        x = img[:, :, ::-1].astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = self.session.run(None, {self.in_name: x})[0]
        out = np.asarray(out)
        # expect (1, N, C) or (1, C, N)
        if out.ndim != 3:
            return []
        if out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))
        preds = out[0]
        boxes: List[DetBox] = []
        scale_x = w0 / iw
        scale_y = h0 / ih
        for row in preds:
            if row.shape[0] < 6:
                continue
            conf = float(row[4])
            if conf < self.conf_thres:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            cls_id = int(row[5]) if row.shape[0] > 5 else 0
            boxes.append(DetBox((x1, y1, x2, y2), conf, cls_id))
        return self._nms(boxes)

    def _nms(self, boxes: List[DetBox]) -> List[DetBox]:
        if not boxes:
            return []
        arr = np.array([[*b.xyxy, b.conf, b.cls_id] for b in boxes], dtype=np.float32)
        xyxy = arr[:, :4]
        scores = arr[:, 4]
        idx = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        if len(idx) == 0:
            return []
        idx = idx.flatten() if isinstance(idx, np.ndarray) else [i[0] for i in idx]
        return [boxes[int(i)] for i in idx]
