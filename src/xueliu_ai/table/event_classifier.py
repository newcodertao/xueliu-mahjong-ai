from __future__ import annotations

from dataclasses import dataclass, replace

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
    oversized: bool = False,
) -> EventClassification:
    if oversized:
        return EventClassification("event_tiles", "oversized_animation_tile")
    if movement > max(tile.width, tile.height) * 0.8:
        return EventClassification("event_tiles", "moving_animation_tile")
    if anchor and _inside(tile, anchor) and stable_frames >= 3:
        return EventClassification("hu_display_tiles", "stable_hu_anchor")
    return EventClassification("unknown_tiles", "isolated_without_anchor_evidence")


@dataclass
class _Observation:
    tile: ZoneTile
    stable_frames: int = 1
    missed_frames: int = 0


class EventTileClassifier:
    """Separate stable HU displays from short-lived moving animation tiles."""

    def __init__(self, hu_stable_frames: int = 3, max_missed: int = 2) -> None:
        self.hu_stable_frames = hu_stable_frames
        self.max_missed = max_missed
        self._observations: dict[int, _Observation] = {}
        self._next_id = 1

    def update(self, zones, width: int, height: int):
        candidates = [
            tile
            for tile in zones.zone_tiles
            if tile.zone in {"unknown_tiles", "candidate_meld_tiles"}
        ]
        unmatched = set(self._observations)
        classified: list[ZoneTile] = []
        for tile in candidates:
            track_id = self._match(tile, unmatched)
            movement = 0.0
            stable_frames = 1
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            else:
                unmatched.remove(track_id)
                previous = self._observations[track_id]
                movement = _distance(previous.tile, tile)
                stable_frames = previous.stable_frames + 1 if movement <= max(tile.width, tile.height) * 0.25 else 1

            tracked = replace(tile, track_id=track_id)
            anchor = _hu_anchor(tracked, width, height)
            oversized = tracked.width > width * 0.10 or tracked.height > height * 0.11
            result = classify_isolated_tile(
                tracked,
                anchor,
                stable_frames,
                movement,
                oversized,
            )
            if result.zone == "unknown_tiles" and stable_frames < self.hu_stable_frames:
                result = EventClassification("event_tiles", "transient_unsettled_tile")
            assigned = replace(tracked, zone=result.zone, reason=result.reason)
            self._observations[track_id] = _Observation(assigned, stable_frames)
            classified.append(assigned)

        for track_id in unmatched:
            observation = self._observations[track_id]
            observation.missed_frames += 1
            if observation.missed_frames > self.max_missed:
                del self._observations[track_id]

        untouched = [
            tile
            for tile in zones.zone_tiles
            if tile.zone not in {"unknown_tiles", "candidate_meld_tiles"}
        ]
        unknown = [tile for tile in classified if tile.zone == "unknown_tiles"]
        hu = [tile for tile in classified if tile.zone == "hu_display_tiles"]
        events = [tile for tile in classified if tile.zone == "event_tiles"]
        return replace(
            zones,
            unknown_tiles=[tile.label for tile in unknown],
            candidate_meld_tiles=[],
            hu_display_tiles=[tile.label for tile in hu],
            event_tiles=[tile.label for tile in events],
            zone_tiles=[*untouched, *classified],
        )

    def _match(self, tile: ZoneTile, candidates: set[int]) -> int | None:
        matches = []
        for track_id in candidates:
            previous = self._observations[track_id].tile
            if previous.label != tile.label:
                continue
            distance = _distance(previous, tile)
            if distance <= max(tile.width, tile.height) * 1.25:
                matches.append((distance, track_id))
        return min(matches)[1] if matches else None


def _inside(tile: ZoneTile, anchor: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = anchor
    return x1 <= tile.center_x <= x2 and y1 <= tile.center_y <= y2


def _distance(left: ZoneTile, right: ZoneTile) -> float:
    return ((left.center_x - right.center_x) ** 2 + (left.center_y - right.center_y) ** 2) ** 0.5


def _hu_anchor(tile: ZoneTile, width: int, height: int) -> tuple[float, float, float, float] | None:
    nx = tile.center_x / max(1, width)
    ny = tile.center_y / max(1, height)
    if nx <= 0.22:
        return (0, height * 0.18, width * 0.22, height * 0.82)
    if nx >= 0.78:
        return (width * 0.78, height * 0.18, width, height * 0.82)
    if ny <= 0.24:
        return (width * 0.22, 0, width * 0.78, height * 0.24)
    if ny >= 0.72:
        return (width * 0.18, height * 0.72, width * 0.82, height)
    return None
