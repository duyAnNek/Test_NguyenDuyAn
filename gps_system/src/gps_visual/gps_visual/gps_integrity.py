"""
Module 6 — GPS integrity state machine + latched high-confidence WGS84.

States: GPS_GOOD → GPS_DEGRADED → GPS_LOST → GPS_GOOD

HDOP / satellite count / SNR may arrive on auxiliary topics; NavSatFix alone is used
for STATUS_NO_FIX and covariance heuristics when extras are absent.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


class GpsIntegrityState(enum.IntEnum):
    GPS_GOOD = 0
    GPS_DEGRADED = 1
    GPS_LOST = 2


@dataclass
class LatchedFix:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    stamp_sec: float = 0.0
    valid: bool = False


@dataclass
class GpsQualitySample:
    hdop: float = 1.0
    n_satellites: int = 12
    mean_snr_dbhz: float = 40.0
    fix_ok: bool = True
    stamp_sec: float = field(default_factory=lambda: time.time())


class GPSIntegrityMonitor:
    def __init__(
        self,
        hdop_degraded: float = 5.0,
        min_satellites: int = 4,
    ) -> None:
        self.hdop_degraded = float(hdop_degraded)
        self.min_satellites = int(min_satellites)
        self.state = GpsIntegrityState.GPS_GOOD
        self.latched = LatchedFix()
        self._last_good = LatchedFix()
        self._prev_state = GpsIntegrityState.GPS_GOOD
        self._lost_since: Optional[float] = None
        self._relock_since: Optional[float] = None

    def update(self, q: GpsQualitySample, lat: float, lon: float, alt: float = 0.0) -> GpsIntegrityState:
        self._prev_state = self.state
        good_signal = q.fix_ok and q.hdop <= self.hdop_degraded and q.n_satellites >= self.min_satellites

        if not q.fix_ok or q.n_satellites < 2:
            self.state = GpsIntegrityState.GPS_LOST
        elif not good_signal:
            self.state = GpsIntegrityState.GPS_DEGRADED
        else:
            self.state = GpsIntegrityState.GPS_GOOD

        if self.state == GpsIntegrityState.GPS_GOOD:
            self._last_good = LatchedFix(lat=lat, lon=lon, alt=alt, stamp_sec=q.stamp_sec, valid=True)
            self.latched = self._last_good

        if self._prev_state != GpsIntegrityState.GPS_LOST and self.state == GpsIntegrityState.GPS_LOST:
            self._lost_since = q.stamp_sec
        if self._prev_state == GpsIntegrityState.GPS_LOST and self.state != GpsIntegrityState.GPS_LOST:
            self._relock_since = q.stamp_sec

        return self.state

    def handover_latency_sec(self, now_sec: float) -> Optional[float]:
        """Elapsed time since edge into LOST or since re-lock (for external logging only)."""
        if self._relock_since is None:
            return None
        return max(0.0, now_sec - self._relock_since)

    @staticmethod
    def enu_offset_m(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
        """Flat-earth approximate ENU meters from (lat0,lon0) to (lat1,lon1)."""
        R = 6378137.0
        dlat = math.radians(lat1 - lat0)
        dlon = math.radians(lon1 - lon0)
        north = dlat * R
        east = dlon * R * math.cos(math.radians(lat0))
        return east, north
