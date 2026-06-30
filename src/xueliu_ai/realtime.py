from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from xueliu_ai.capture.roi_config import load_screen_profile
from xueliu_ai.capture.screen_capture import ScreenCapture
from xueliu_ai.game_logging.game_logger import GameLogger
from xueliu_ai.strategy.discard_advisor import advise_discard
from xueliu_ai.vision.detection_validator import StableHandTracker, validate_hand_detections
from xueliu_ai.vision.yolo_detector import YoloDetector


@dataclass(frozen=True)
class RealtimeTick:
    valid: bool
    tiles: list[str]
    recommendation: str | None
    message: str


def run_realtime_loop(
    model_path: str | Path,
    missing_suit: str | None = None,
    interval_seconds: float = 0.5,
    limit: int | None = None,
    profile_path: str | Path = "configs/screen_profile.yaml",
    log_path: str | Path = "data/games/realtime.jsonl",
) -> list[RealtimeTick]:
    profile = load_screen_profile(profile_path)
    roi = profile.get_roi("my_hand")
    capture = ScreenCapture(profile.monitor)
    detector = YoloDetector(model_path)
    tracker = StableHandTracker(stable_frames=2)
    logger = GameLogger(log_path)
    ticks: list[RealtimeTick] = []

    index = 0
    while limit is None or index < limit:
        frame = capture.grab().image_bgr
        crop = roi.crop(frame)
        detections = detector.detect_image(crop)
        result = validate_hand_detections(detections)
        stable = tracker.update(result)
        recommendation = None
        message = stable.reason or "stable"
        if stable.valid and len(stable.tiles) == 14:
            advice = advise_discard(stable.tiles, missing_suit)
            recommendation = advice.recommended
            message = advice.explanation

        tick = RealtimeTick(stable.valid, stable.tiles, recommendation, message)
        logger.log("realtime_tick", tick.__dict__)
        print(json.dumps(tick.__dict__, ensure_ascii=False))
        ticks.append(tick)
        index += 1
        if limit is None or index < limit:
            time.sleep(interval_seconds)
    return ticks
