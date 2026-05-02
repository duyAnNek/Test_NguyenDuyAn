"""
Module 5A — Monocular VO (OpenCV) with optional wheel/CAN scale.

ORB + two-view geometry + recoverPose; scale from longitudinal wheel speed when available.
ORB-SLAM3 is not bundled; this module is a lightweight CPU fallback compatible with ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VOPose2D:
    x: float
    y: float
    theta: float  # rad, vehicle heading in local frame


class MonocularWheelScaledVO:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        max_features: int = 1500,
        vo_scale_fallback_m_per_unit: float = 1.0,
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.dist = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float64)
        self._orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_desc = None
        self.pose = VOPose2D(0.0, 0.0, 0.0)
        self.vo_scale_fallback = float(vo_scale_fallback_m_per_unit)
        self._last_wheel_m_s: float = 0.0

    def set_wheel_speed(self, v_longitudinal_m_s: float) -> None:
        self._last_wheel_m_s = float(v_longitudinal_m_s)

    def reset(self) -> None:
        self._prev_gray = None
        self.pose = VOPose2D(0.0, 0.0, 0.0)

    def _estimate_scale(self, t_vec: np.ndarray, dt: float) -> float:
        """t_vec unit-norm direction from recoverPose; scale using wheel odometry."""
        if dt <= 1e-6:
            return self.vo_scale_fallback
        speed = abs(self._last_wheel_m_s)
        if speed < 0.05:
            return self.vo_scale_fallback
        tn = np.linalg.norm(t_vec.reshape(3))
        if tn < 1e-6:
            return self.vo_scale_fallback
        # approximate: distance traveled / translation magnitude in camera units
        dist = speed * dt
        return float(dist / (tn + 1e-9))

    def step(self, frame_bgr: np.ndarray, dt: float) -> VOPose2D:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if self._prev_gray is None or desc is None or self._prev_desc is None:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return self.pose

        matches = self._bf.match(self._prev_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return self.pose

        pts0 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts1 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(pts0, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self._prev_gray = gray
            self._prev_kp = kp
            self._prev_desc = desc
            return self.pose

        _, R, t, _ = cv2.recoverPose(E, pts0, pts1, self.K, mask=mask)
        scale = self._estimate_scale(t, dt)
        ts = (scale * t.reshape(3)).astype(np.float64)
        # OpenCV camera: X right, Y down, Z forward — map to vehicle planar delta (x forward, y left).
        dx = float(ts[2])
        dy = float(-ts[0])
        yaw = float(np.arctan2(R[0, 2], R[2, 2]))
        self.pose = VOPose2D(
            self.pose.x + dx,
            self.pose.y + dy,
            self._wrap_pi(self.pose.theta + yaw),
        )

        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_desc = desc
        return self.pose

    def apply_gps_relock_correction(self, dx: float, dy: float, dtheta: float) -> None:
        """Apply correction in local VO frame (Module 5D / fusion handover)."""
        self.pose.x += dx
        self.pose.y += dy
        self.pose.theta = self._wrap_pi(self.pose.theta + dtheta)

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return float(np.arctan2(np.sin(a), np.cos(a)))
