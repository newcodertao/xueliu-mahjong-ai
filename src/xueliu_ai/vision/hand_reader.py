from __future__ import annotations

from pathlib import Path

import cv2

from xueliu_ai.capture.roi_config import load_screen_profile
from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.detection_validator import StableHandTracker, validate_hand_detections
from xueliu_ai.vision.yolo_detector import YoloDetector


class HandReader:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.5,
        stable_frames: int = 2,
    ) -> None:
        self.detector = YoloDetector(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracker = StableHandTracker(stable_frames=stable_frames)

    def read_image(self, image_path: str | Path) -> list[str]:
        detections = self.detector.detect_image(resolve_path(image_path), self.confidence_threshold, self.iou_threshold)
        result = validate_hand_detections(detections, self.confidence_threshold, self.iou_threshold)
        stable = self.tracker.update(result)
        if not stable.valid:
            raise ValueError(stable.reason)
        return stable.tiles

    def read_fullscreen(self, image_path: str | Path, profile_path: str | Path = "configs/screen_profile.yaml") -> list[str]:
        image = cv2.imread(str(resolve_path(image_path)))
        if image is None:
            raise FileNotFoundError(resolve_path(image_path))
        roi = load_screen_profile(profile_path).get_roi("my_hand")
        crop = roi.crop(image)
        detections = self.detector.detect_image(crop, self.confidence_threshold, self.iou_threshold)
        result = validate_hand_detections(detections, self.confidence_threshold, self.iou_threshold)
        stable = self.tracker.update(result)
        if not stable.valid:
            raise ValueError(stable.reason)
        return stable.tiles
