"""
Module 5B — Lane position from IPM/BEV heuristics (CPU).

CLRKDNet-style learned detector is not bundled; this module uses homography + lane mask
centroid split for LEFT/RIGHT/UNKNOWN enum values.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple

import cv2
import numpy as np


class LanePosition(IntEnum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2


def default_ipm_matrix(image_wh: Tuple[int, int], bev_wh: Tuple[int, int]) -> np.ndarray:
    """Weak default homography (replace with calibrated IPM in deployment)."""
    w, h = image_wh
    bw, bh = bev_wh
    src = np.float32([[w * 0.45, h * 0.55], [w * 0.55, h * 0.55], [w * 0.95, h], [w * 0.05, h]])
    dst = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    return cv2.getPerspectiveTransform(src, dst)


class LaneBEVDetector:
    def __init__(
        self,
        bev_size: Tuple[int, int] = (200, 200),
        H_ipm: Optional[np.ndarray] = None,
    ) -> None:
        self.bev_w, self.bev_h = bev_size
        self.H = H_ipm

    def set_homography(self, H: np.ndarray) -> None:
        self.H = H.astype(np.float64)

    def infer_lane_position(self, frame_bgr: np.ndarray) -> LanePosition:
        h, w = frame_bgr.shape[:2]
        H = self.H
        if H is None:
            H = default_ipm_matrix((w, h), (self.bev_w, self.bev_h))
        bev = cv2.warpPerspective(frame_bgr, H, (self.bev_w, self.bev_h))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # assume bright lanes
        if float(mask.mean()) > 127:
            mask = 255 - mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.where(mask > 200)
        if len(xs) < 80:
            return LanePosition.UNKNOWN
        cx = float(np.median(xs))
        mid = self.bev_w * 0.5
        if cx < mid - 15:
            return LanePosition.LEFT
        if cx > mid + 15:
            return LanePosition.RIGHT
        return LanePosition.UNKNOWN
