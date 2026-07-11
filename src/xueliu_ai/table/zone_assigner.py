from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.capture.roi_config import Roi
from xueliu_ai.vision.detection_types import Detection


@dataclass(frozen=True)
class ZoneAssignment:
    detection: Detection
    zone: str
    reason: str


ROI_PRIORITY = (
    ("my_hand", "hand"),
    ("my_melds", "bottom_melds"),
    ("discards", "center_discards"),
    ("left_melds", "left_melds"),
    ("top_melds", "top_melds"),
    ("right_melds", "right_melds"),
)


def assign_by_roi_priority(
    detection: Detection,
    table_roi: Roi,
    rois: dict[str, Roi],
) -> ZoneAssignment | None:
    """Resolve overlapping ROIs with hand > meld > discard priority."""
    screen_x = table_roi.x + detection.center_x
    screen_y = table_roi.y + (detection.y1 + detection.y2) / 2
    for roi_name, zone in ROI_PRIORITY:
        roi = rois.get(roi_name)
        if roi and not roi.is_empty and _inside(screen_x, screen_y, roi):
            return ZoneAssignment(detection, zone, f"roi_priority:{roi_name}")
    return None


def _inside(x: float, y: float, roi: Roi) -> bool:
    return roi.x <= x <= roi.x + roi.width and roi.y <= y <= roi.y + roi.height
