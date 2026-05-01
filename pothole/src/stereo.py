from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class StereoCalibration:
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray
    image_size: Tuple[int, int]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StereoCalibration":
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(f"Cannot open stereo calibration file: {path}")
        try:
            k1 = fs.getNode("K1").mat()
            d1 = fs.getNode("D1").mat()
            k2 = fs.getNode("K2").mat()
            d2 = fs.getNode("D2").mat()
            r = fs.getNode("R").mat()
            t = fs.getNode("T").mat()
            width = int(fs.getNode("image_width").real())
            height = int(fs.getNode("image_height").real())
        finally:
            fs.release()
        if any(m is None for m in [k1, d1, k2, d2, r, t]) or width <= 0 or height <= 0:
            raise ValueError("Invalid stereo calibration file content.")
        return cls(k1=k1, d1=d1, k2=k2, d2=d2, r=r, t=t, image_size=(width, height))


class StereoDepthEstimator:
    def __init__(self, calib: StereoCalibration):
        self.calib = calib
        size = calib.image_size
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            calib.k1, calib.d1, calib.k2, calib.d2, size, calib.r, calib.t, flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.q = q
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            calib.k1, calib.d1, r1, p1, size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            calib.k2, calib.d2, r2, p2, size, cv2.CV_32FC1
        )
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 8,
            blockSize=7,
            P1=8 * 3 * 7 * 7,
            P2=32 * 3 * 7 * 7,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def estimate_depth_map(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        w, h = self.calib.image_size
        left = cv2.resize(left_bgr, (w, h))
        right = cv2.resize(right_bgr, (w, h))
        left_rect = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp[disp <= 0.5] = np.nan

        points_3d = cv2.reprojectImageTo3D(disp, self.q)
        depth = points_3d[:, :, 2]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        return depth
