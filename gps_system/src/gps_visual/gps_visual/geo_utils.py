"""Local tangent plane ENU meters from a reference lat/lon."""

from __future__ import annotations

import math
from typing import Tuple


def enu_from_latlon(lat0: float, lon0: float, lat: float, lon: float) -> Tuple[float, float]:
    R = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = dlat * R
    east = dlon * R * math.cos(math.radians(lat0))
    return east, north
