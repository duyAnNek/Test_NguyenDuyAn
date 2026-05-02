"""
Ghost projection + semantic bbox association (Module 7).

Projects landmark map position into image via weak pinhole model; compares to YOLO boxes
using class consistency + IoU. Reprojection error proxy drives EKF landmark update.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .yolo_onnx import DetBox


def project_xy_to_uv(
    mx: float,
    my: float,
    vehicle_xytheta: Tuple[float, float, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    landmark_z_m: float = 0.0,
) -> Tuple[float, float]:
    """Map frame landmark (mx,my) to image uv in vehicle camera approx (forward Z)."""
    vx, vy, th = vehicle_xytheta
    c, s = np.cos(-th), np.sin(-th)
    rx = c * (mx - vx) - s * (my - vy)
    ry = s * (mx - vx) + c * (my - vy)
    # camera: X right = ry (lateral), Z forward = rx
    X, Z = float(ry), float(rx) + 1e-3
    Y = float(landmark_z_m)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter + 1e-6
    return inter / union


def match_ghosts_to_detections(
    ghost_boxes_uv: List[Tuple[float, float, float, float]],
    dets: List[DetBox],
    min_iou: float = 0.15,
) -> List[Tuple[int, int, float]]:
    """Returns list of (ghost_idx, det_idx, iou)."""
    matches: List[Tuple[int, int, float]] = []
    for gi, g in enumerate(ghost_boxes_uv):
        gx1, gy1, gx2, gy2 = g
        gxy = (int(gx1), int(gy1), int(gx2), int(gy2))
        best = (-1, 0.0)
        for dj, d in enumerate(dets):
            iou = bbox_iou(gxy, d.xyxy)
            if iou > best[1]:
                best = (dj, iou)
        if best[0] >= 0 and best[1] >= min_iou:
            matches.append((gi, int(best[0]), float(best[1])))
    return matches
