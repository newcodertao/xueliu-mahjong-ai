from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

from xueliu_ai.capture.roi_config import load_screen_profile
from xueliu_ai.capture.screen_capture import ScreenCapture
from xueliu_ai.paths import resolve_path


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def collect_frames(
    interval_seconds: float = 0.5,
    limit: int | None = None,
    profile_path: str | Path = "configs/screen_profile.yaml",
    fullscreen_dir: str | Path = "data/raw/fullscreen",
    my_hand_dir: str | Path = "data/raw/my_hand",
) -> int:
    profile = load_screen_profile(profile_path)
    capture = ScreenCapture(profile.monitor)
    fullscreen_path = resolve_path(fullscreen_dir)
    my_hand_path = resolve_path(my_hand_dir)
    fullscreen_path.mkdir(parents=True, exist_ok=True)
    my_hand_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    while limit is None or saved < limit:
        frame = capture.grab().image_bgr
        name = f"{_stamp()}.png"
        cv2.imwrite(str(fullscreen_path / name), frame)

        roi = profile.get_roi("my_hand")
        if not roi.is_empty:
            cv2.imwrite(str(my_hand_path / name), roi.crop(frame))

        saved += 1
        if limit is None or saved < limit:
            time.sleep(interval_seconds)
    return saved
