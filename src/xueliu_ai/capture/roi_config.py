from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xueliu_ai.config import load_yaml, save_yaml
from xueliu_ai.paths import resolve_path


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def crop(self, image: np.ndarray) -> np.ndarray:
        if self.is_empty:
            raise ValueError("ROI is empty. Run roi-calibrate or edit configs/screen_profile.yaml.")
        return image[self.y : self.y + self.height, self.x : self.x + self.width].copy()

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class ScreenProfile:
    monitor: int
    rois: dict[str, Roi]
    width: int | None = None
    height: int | None = None

    def get_roi(self, name: str) -> Roi:
        if name not in self.rois:
            raise KeyError(f"ROI {name!r} is not configured.")
        return self.rois[name]


def load_screen_profile(path: str | Path = "configs/screen_profile.yaml") -> ScreenProfile:
    raw = load_yaml(path)
    screen = raw.get("screen", {})
    rois = {
        name: Roi(
            x=int(value.get("x", 0) or 0),
            y=int(value.get("y", 0) or 0),
            width=int(value.get("width", 0) or 0),
            height=int(value.get("height", 0) or 0),
        )
        for name, value in raw.get("rois", {}).items()
    }
    return ScreenProfile(
        monitor=int(screen.get("monitor", 1) or 1),
        width=screen.get("width"),
        height=screen.get("height"),
        rois=rois,
    )


def save_screen_profile(
    profile: ScreenProfile, path: str | Path = "configs/screen_profile.yaml"
) -> None:
    save_yaml(
        path,
        {
            "screen": {
                "monitor": profile.monitor,
                "width": profile.width,
                "height": profile.height,
            },
            "rois": {name: roi.to_dict() for name, roi in profile.rois.items()},
        },
    )


def update_roi(
    name: str,
    roi: Roi,
    path: str | Path = "configs/screen_profile.yaml",
) -> None:
    profile_path = resolve_path(path)
    profile = load_screen_profile(profile_path)
    profile.rois[name] = roi
    save_screen_profile(profile, profile_path)


def clear_rois(
    names: list[str] | tuple[str, ...],
    path: str | Path = "configs/screen_profile.yaml",
) -> None:
    """Clear selected ROI definitions while preserving the screen profile."""
    profile_path = resolve_path(path)
    profile = load_screen_profile(profile_path)
    for name in names:
        profile.rois[name] = Roi(0, 0, 0, 0)
    save_screen_profile(profile, profile_path)
