from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import CameraConfig
from .geometry import estimate_area_m2, estimate_depth_m
from .models import DepthOnnxEstimator, YoloOnnxDetector
from .severity import SeverityThreshold, classify_severity


@dataclass
class DetectionRecord:
    frame_idx: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    depth_m: float
    area_m2: float
    severity: str


class PotholePipeline:
    def __init__(
        self,
        yolo_onnx_path: str,
        depth_onnx_path: Optional[str],
        camera_cfg: CameraConfig,
        severity_cfg: SeverityThreshold,
        conf_thres: float = 0.25,
        depth_interval: int = 1,
        depth_input_size: Optional[int] = None,
    ):
        self.detector = YoloOnnxDetector(yolo_onnx_path, conf_thres=conf_thres)
        self.depth = (
            DepthOnnxEstimator(depth_onnx_path, input_size=depth_input_size)
            if depth_onnx_path
            else None
        )
        self.camera_cfg = camera_cfg
        self.severity_cfg = severity_cfg
        self.depth_interval = max(1, int(depth_interval))

    def _estimate_depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.depth is not None:
            return self.depth.estimate(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray /= 255.0
        return 1.0 - gray

    def run_video(
        self,
        video_path: str,
        output_dir: str,
        visualize: bool = True,
        map50_reference: Optional[float] = None,
        save_video_path: Optional[str] = None,
    ) -> List[DetectionRecord]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_cap = float(cap.get(cv2.CAP_PROP_FPS))
        if fps_cap < 1.0:
            fps_cap = 30.0
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w_frame <= 0 or h_frame <= 0:
            raise RuntimeError(f"Invalid video size: {w_frame}x{h_frame}")

        writer: Optional[cv2.VideoWriter] = None
        if save_video_path is not None:
            Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video_path, fourcc, fps_cap, (w_frame, h_frame))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter for: {save_video_path}")

        annotate = visualize or (save_video_path is not None)

        fps_times: List[float] = []
        records: List[DetectionRecord] = []
        frame_idx = 0
        cached_depth_map: Optional[np.ndarray] = None
        last_frame_ts = time.perf_counter()
        fps_window = deque(maxlen=30)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            if cached_depth_map is None or frame_idx % self.depth_interval == 0:
                cached_depth_map = self._estimate_depth_map(frame)
            depth_map = cached_depth_map
            detections = self.detector.infer(frame)

            for det in detections:
                area_m2 = estimate_area_m2(
                    frame,
                    det.bbox_xyxy,
                    meters_per_pixel_bev=self.camera_cfg.meters_per_pixel_bev,
                )
                depth_m = estimate_depth_m(depth_map, det.bbox_xyxy, self.camera_cfg)
                severity = classify_severity(depth_m, area_m2, self.severity_cfg)
                x1, y1, x2, y2 = det.bbox_xyxy
                records.append(
                    DetectionRecord(
                        frame_idx=frame_idx,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        conf=det.conf,
                        depth_m=depth_m,
                        area_m2=area_m2,
                        severity=severity,
                    )
                )
                if annotate:
                    label = f"{severity} d={depth_m:.3f}m a={area_m2:.3f}m2"
                    color = {
                        "minor": (0, 255, 0),      # green
                        "moderate": (0, 255, 255), # yellow
                        "severe": (0, 0, 255),     # red
                    }.get(severity, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

            proc_dt = time.perf_counter() - t0
            now = time.perf_counter()
            e2e_dt = now - last_frame_ts
            last_frame_ts = now
            fps_times.append(e2e_dt)
            fps_window.append(1.0 / max(e2e_dt, 1e-6))
            if annotate:
                fps_inst = 1.0 / max(e2e_dt, 1e-6)
                fps_avg = float(np.mean(fps_window)) if fps_window else fps_inst
                cv2.putText(
                    frame,
                    f"FPS: {fps_inst:.2f} (avg30: {fps_avg:.2f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"PROC: {proc_dt * 1000.0:.1f} ms",
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 50, 50),
                    2,
                )
                if map50_reference is not None:
                    cv2.putText(
                        frame,
                        f"Ref mAP@0.5 (test set): {map50_reference:.3f}",
                        (20, 86),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
            if writer is not None:
                writer.write(frame)
            if visualize:
                cv2.imshow("Pothole Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()

        self._write_outputs(records, fps_times, output_dir)
        return records

    def _write_outputs(self, records: List[DetectionRecord], fps_times: List[float], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else [
                "frame_idx", "x1", "y1", "x2", "y2", "conf", "depth_m", "area_m2", "severity"
            ])
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))

        mean_dt = float(np.mean(fps_times)) if fps_times else 0.0
        metrics = {
            "frames_processed": int(len(fps_times)),
            "detections": int(len(records)),
            "avg_fps": float(1.0 / max(mean_dt, 1e-6)) if fps_times else 0.0,
            "avg_latency_ms": float(mean_dt * 1000.0),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out / "fps_log.txt").write_text(
            "\n".join(f"{(1.0 / max(t, 1e-6)):.3f}" for t in fps_times),
            encoding="utf-8",
        )
