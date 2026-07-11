from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.table.structured_types import ZoneTile


@dataclass(frozen=True)
class EventClassification:
    zone: str
    reason: str


def classify_isolated_tile(
    tile: ZoneTile,
    anchor: tuple[float, float, float, float] | None = None,
    stable_frames: int = 0,
    movement: float = 0.0,
) -> EventClassification:
    if movement > max(tile.width, tile.height) * 0.8:
        return EventClassification("event_tiles", "moving_animation_tile")
    if anchor and _inside(tile, anchor) and stable_frames >= 3:
        return EventClassification("hu_display_tiles", "stable_hu_anchor")
    return EventClassification("unknown_tiles", "isolated_without_anchor_evidence")


def _inside(tile: ZoneTile, anchor: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = anchor
    return x1 <= tile.center_x <= x2 and y1 <= tile.center_y <= y2
