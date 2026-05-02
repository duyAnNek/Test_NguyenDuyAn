"""
Module 7 — lightweight EKF on SE(2) slice: state [x, y, theta].

Predict: VO odometry. Update: GPS (full pose xy + optional theta from course),
landmark position measurements, soft lane lateral constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EKFState:
    x: float
    y: float
    theta: float
    P: np.ndarray  # 3x3 covariance


class PoseEKF:
    def __init__(
        self,
        sigma_vo_xy: float = 0.15,
        sigma_vo_theta: float = 0.02,
        sigma_gps_xy: float = 3.0,
        sigma_gps_theta: float = 0.3,
        sigma_landmark_xy: float = 2.0,
    ) -> None:
        self.Q = np.diag([sigma_vo_xy**2, sigma_vo_xy**2, sigma_vo_theta**2]).astype(np.float64)
        self.R_gps = np.diag([sigma_gps_xy**2, sigma_gps_xy**2, sigma_gps_theta**2]).astype(np.float64)
        self.R_lm = np.diag([sigma_landmark_xy**2, sigma_landmark_xy**2]).astype(np.float64)
        self.s = EKFState(0.0, 0.0, 0.0, np.eye(3, dtype=np.float64) * 25.0)

    def predict_map_delta(self, dx: float, dy: float, dtheta: float) -> None:
        """VO cumulative delta already expressed in the same local map as the EKF."""
        self.s.x += float(dx)
        self.s.y += float(dy)
        self.s.theta = self._wrap(self.s.theta + float(dtheta))
        F = np.eye(3, dtype=np.float64)
        self.s.P = F @ self.s.P @ F.T + self.Q

    def predict_odom(self, dx: float, dy: float, dtheta: float) -> None:
        """Body-frame incremental odometry (x forward, y left) rotated by current heading."""
        c, s = math.cos(self.s.theta), math.sin(self.s.theta)
        mx = c * dx - s * dy
        my = s * dx + c * dy
        self.s.x += mx
        self.s.y += my
        self.s.theta = self._wrap(self.s.theta + dtheta)
        F = np.eye(3, dtype=np.float64)
        self.s.P = F @ self.s.P @ F.T + self.Q

    def update_gps(self, x: float, y: float, theta: Optional[float] = None) -> None:
        z = np.array([x, y, self.s.theta if theta is None else float(theta)], dtype=np.float64)
        xh = np.array([self.s.x, self.s.y, self.s.theta], dtype=np.float64)
        innov = z - xh
        innov[2] = self._wrap(float(innov[2]))
        H = np.eye(3, dtype=np.float64)
        S = H @ self.s.P @ H.T + self.R_gps
        K = self.s.P @ H.T @ np.linalg.inv(S)
        xh_new = xh + K @ innov
        self.s.x = float(xh_new[0])
        self.s.y = float(xh_new[1])
        self.s.theta = self._wrap(float(xh_new[2]))
        I = np.eye(3, dtype=np.float64)
        self.s.P = (I - K @ H) @ self.s.P

    def update_landmark_xy(self, lx: float, ly: float) -> None:
        """Direct position measurement (e.g. from ghost+detection association)."""
        z = np.array([lx, ly], dtype=np.float64)
        h = np.array([self.s.x, self.s.y], dtype=np.float64)
        innov = z - h
        H = np.zeros((2, 3), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        S = H @ self.s.P @ H.T + self.R_lm
        K = self.s.P @ H.T @ np.linalg.inv(S)
        delta = K @ innov
        self.s.x += float(delta[0])
        self.s.y += float(delta[1])
        self.s.theta = self._wrap(self.s.theta + float(delta[2]))
        I = np.eye(3, dtype=np.float64)
        self.s.P = (I - K @ H) @ self.s.P

    def soft_lane_update(self, lane: int, strength: float = 0.08) -> None:
        """
        lane: 1=LEFT pull vehicle lateral +y in body left direction, 2=RIGHT -y.
        Very soft nudge — avoids hard jumps.
        """
        if lane == 0:
            return
        c, s = math.cos(self.s.theta), math.sin(self.s.theta)
        # lateral axis in map: (-sin, cos) for body +y left
        nx, ny = -s, c
        if lane == 1:
            self.s.x += strength * nx
            self.s.y += strength * ny
        elif lane == 2:
            self.s.x -= strength * nx
            self.s.y -= strength * ny

    @staticmethod
    def _wrap(a: float) -> float:
        return float(np.arctan2(math.sin(a), math.cos(a)))
