"""Module 5C — U-turn from heading change within time window."""

from __future__ import annotations

import collections
import math
import time
from typing import Deque, Tuple


class UTurnDetector:
    def __init__(self, angle_deg: float = 150.0, window_sec: float = 2.0) -> None:
        self.angle_rad = math.radians(angle_deg)
        self.window_sec = float(window_sec)
        self._buf: Deque[Tuple[float, float]] = collections.deque()
        self._latched_event = False

    def update(self, heading_rad: float) -> bool:
        now = time.monotonic()
        self._buf.append((now, heading_rad))
        while self._buf and now - self._buf[0][0] > self.window_sec:
            self._buf.popleft()
        if len(self._buf) < 2:
            return False
        heads = [h for _, h in self._buf]
        d = max(self._angle_diff(heads[i], heads[j]) for i in range(len(heads)) for j in range(i + 1, len(heads)))
        active = d >= self.angle_rad
        if active and not self._latched_event:
            self._latched_event = True
            return True
        if not active:
            self._latched_event = False
        return False

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)
