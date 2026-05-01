from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import CameraConfig


def estimate_depth_m(
    depth_map_relative: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    cfg: CameraConfig,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = depth_map_relative.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))

    roi = depth_map_relative[y1:y2, x1:x2]
    road_context = depth_map_relative[max(0, y1 - 20) : y1 + 5, x1:x2]
    if roi.size == 0:
        return 0.0

    roi_depth = float(np.median(roi))
    road_depth = float(np.median(road_context)) if road_context.size else roi_depth
    relative_drop = max(0.0, roi_depth - road_depth)
    scale = cfg.camera_height_m / 2.0
    return float(relative_drop * scale)


def approximate_ground_homography(width: int, height: int) -> np.ndarray:
    src = np.float32(
        [
            [width * 0.30, height * 0.58],
            [width * 0.70, height * 0.58],
            [width * 0.95, height * 0.98],
            [width * 0.05, height * 0.98],
        ]
    )
    dst = np.float32(
        [
            [width * 0.20, 0],
            [width * 0.80, 0],
            [width * 0.80, height],
            [width * 0.20, height],
        ]
    )
    return cv2.getPerspectiveTransform(src, dst)


def estimate_area_m2(
    frame_bgr: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    meters_per_pixel_bev: float,
) -> float:
    h, w = frame_bgr.shape[:2]
    h_mat = approximate_ground_homography(w, h)
    bev = cv2.warpPerspective(frame_bgr, h_mat, (w, h))

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    roi = bev[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixel_count = int(np.count_nonzero(mask))
    return float(pixel_count * (meters_per_pixel_bev**2))
