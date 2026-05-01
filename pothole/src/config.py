from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class CameraConfig:
    camera_height_m: float = 1.2
    camera_pitch_deg: float = 12.0
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 640.0
    cy: float = 360.0
    meters_per_pixel_bev: float = 0.02

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "CameraConfig":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(**data)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
