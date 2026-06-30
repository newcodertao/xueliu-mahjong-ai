from xueliu_ai.vision.detection_types import Detection
from xueliu_ai.vision.detection_validator import StableHandTracker, validate_hand_detections


def _det(label: str, x: int) -> Detection:
    return Detection(label=label, confidence=0.9, x1=x, y1=0, x2=x + 10, y2=20)


def test_validate_valid_hand() -> None:
    labels = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B"]
    result = validate_hand_detections([_det(label, i * 12) for i, label in enumerate(labels)])
    assert result.valid
    assert result.tiles == labels


def test_stable_tracker_waits_for_two_frames() -> None:
    labels = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B"]
    result = validate_hand_detections([_det(label, i * 12) for i, label in enumerate(labels)])
    tracker = StableHandTracker(stable_frames=2)
    assert not tracker.update(result).valid
    assert tracker.update(result).valid
