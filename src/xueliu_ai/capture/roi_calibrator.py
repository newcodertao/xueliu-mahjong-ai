from __future__ import annotations

from pathlib import Path

import cv2

from xueliu_ai.capture.roi_config import Roi, update_roi
from xueliu_ai.capture.screen_capture import ScreenCapture
from xueliu_ai.paths import resolve_path


def calibrate_roi(
    name: str = "my_hand",
    profile_path: str | Path = "configs/screen_profile.yaml",
    monitor: int = 1,
) -> Roi:
    frame = ScreenCapture(monitor).grab().image_bgr
    rect = cv2.selectROI(f"Select ROI: {name}", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, width, height = [int(v) for v in rect]
    roi = Roi(x=x, y=y, width=width, height=height)
    update_roi(name, roi, resolve_path(profile_path))
    return roi
