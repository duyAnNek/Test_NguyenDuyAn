from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


@dataclass
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    conf: float
    cls_id: int


class YoloOnnxDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45):
        # Let Ultralytics handle ONNX output decoding to avoid format mismatch issues.
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        detections: List[Detection] = []
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            imgsz=640,
        )
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),))

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            x1 = int(np.clip(x1, 0, w0 - 1))
            y1 = int(np.clip(y1, 0, h0 - 1))
            x2 = int(np.clip(x2, x1 + 1, w0))
            y2 = int(np.clip(y2, y1 + 1, h0))
            detections.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=float(confs[i]),
                    cls_id=int(clss[i]),
                )
            )
        return detections


class DepthOnnxEstimator:
    def __init__(self, model_path: str, input_size: int | None = None):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        h_dim = input_shape[2] if len(input_shape) > 2 else 518
        w_dim = input_shape[3] if len(input_shape) > 3 else 518
        # Dynamic ONNX exports may expose symbolic dims like "height"/"width".
        default_h = int(h_dim) if isinstance(h_dim, int) else 518
        default_w = int(w_dim) if isinstance(w_dim, int) else 518
        if input_size is not None:
            self.input_h = int(input_size)
            self.input_w = int(input_size)
        else:
            self.input_h = default_h
            self.input_w = default_w

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        h0, w0 = frame_bgr.shape[:2]
        x = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        try:
            depth = self.session.run(None, {self.input_name: x})[0]
        except Exception:
            # Some Depth Anything ONNX exports fail with custom dynamic sizes.
            # Fallback to the canonical 518x518 shape for stability.
            fallback_size = 518
            x = cv2.resize(frame_bgr, (fallback_size, fallback_size))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            depth = self.session.run(None, {self.input_name: x})[0]
        depth = np.squeeze(depth).astype(np.float32)
        depth = cv2.resize(depth, (w0, h0))
        depth -= depth.min()
        depth /= max(depth.max(), 1e-6)
        return depth
