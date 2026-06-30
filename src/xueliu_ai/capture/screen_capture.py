from __future__ import annotations

from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    image_bgr: np.ndarray
    monitor_index: int


class ScreenCapture:
    def __init__(self, monitor_index: int = 1) -> None:
        self.monitor_index = monitor_index

    def grab(self) -> CapturedFrame:
        with mss.mss() as sct:
            monitors = sct.monitors
            if self.monitor_index >= len(monitors):
                raise ValueError(
                    f"Monitor {self.monitor_index} is unavailable. Found {len(monitors) - 1} monitors."
                )
            raw = np.array(sct.grab(monitors[self.monitor_index]))
        image_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return CapturedFrame(image_bgr=image_bgr, monitor_index=self.monitor_index)
