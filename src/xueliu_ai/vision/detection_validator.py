from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.vision.detection_types import Detection


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    tiles: list[str]
    reason: str = ""


def iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return 0.0 if union <= 0 else intersection / union


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(iou(det, existing) < iou_threshold for existing in kept):
            kept.append(det)
    return sorted(kept, key=lambda item: item.center_x)


def validate_hand_detections(
    detections: list[Detection],
    confidence_threshold: float = 0.6,
    iou_threshold: float = 0.5,
) -> ValidationResult:
    filtered = [d for d in detections if d.confidence >= confidence_threshold and d.label in TILE_SET]
    filtered = non_max_suppression(filtered, iou_threshold)
    tiles = [d.label for d in sorted(filtered, key=lambda item: item.center_x)]

    if len(tiles) not in (13, 14):
        return ValidationResult(False, tiles, f"hand tile count must be 13 or 14, got {len(tiles)}")
    counts = Counter(tiles)
    over = [tile for tile, count in counts.items() if count > 4]
    if over:
        return ValidationResult(False, tiles, f"tile count exceeds four: {', '.join(over)}")
    return ValidationResult(True, tiles)


@dataclass
class StableHandTracker:
    stable_frames: int = 2
    history: deque[tuple[str, ...]] = field(default_factory=deque)

    def update(self, result: ValidationResult) -> ValidationResult:
        if not result.valid:
            self.history.clear()
            return result
        current = tuple(result.tiles)
        self.history.append(current)
        while len(self.history) > self.stable_frames:
            self.history.popleft()
        if len(self.history) < self.stable_frames:
            return ValidationResult(False, result.tiles, "waiting for stable frames")
        if len(set(self.history)) == 1:
            return result
        return ValidationResult(False, result.tiles, "frames are not stable")
