from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SeverityThreshold:
    minor_depth_m: float = 0.02
    minor_area_m2: float = 0.10
    moderate_depth_m: float = 0.05
    moderate_area_m2: float = 0.30

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "SeverityThreshold":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(**data)


def classify_severity(depth_m: float, area_m2: float, rule: SeverityThreshold) -> str:
    if depth_m < rule.minor_depth_m and area_m2 < rule.minor_area_m2:
        return "minor"
    if depth_m < rule.moderate_depth_m and area_m2 < rule.moderate_area_m2:
        return "moderate"
    return "severe"
