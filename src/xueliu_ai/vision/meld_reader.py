from __future__ import annotations

from xueliu_ai.vision.detection_types import Detection
from xueliu_ai.vision.detection_validator import non_max_suppression


def read_meld_tiles(detections: list[Detection], confidence_threshold: float = 0.6) -> list[str]:
    filtered = [det for det in detections if det.confidence >= confidence_threshold]
    return [det.label for det in sorted(non_max_suppression(filtered), key=lambda item: item.center_x)]
