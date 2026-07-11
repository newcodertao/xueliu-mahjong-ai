from __future__ import annotations

from statistics import median

from xueliu_ai.table.structured_types import ZoneTile


class HandSlotTracker:
    def __init__(self, max_missed: int = 2) -> None:
        self.max_missed = max_missed
        self._previous: list[ZoneTile] = []
        self._misses: dict[int, int] = {}

    def update(
        self,
        hand_tiles: list[ZoneTile],
        low_confidence_candidates: list[ZoneTile] | None = None,
    ) -> list[ZoneTile]:
        current = sorted(hand_tiles, key=lambda tile: tile.center_x)
        candidates = low_confidence_candidates or []
        if len(self._previous) == len(current) + 1 and len(self._previous) >= 3:
            missing_index = self._find_missing_slot(current)
            if missing_index is not None:
                recovered = self._candidate_for_slot(self._previous[missing_index], candidates)
                current.insert(missing_index, recovered)
        self._previous = current
        return current

    def _find_missing_slot(self, current: list[ZoneTile]) -> int | None:
        previous = self._previous
        best: tuple[float, int] | None = None
        for missing in range(len(previous)):
            remaining = previous[:missing] + previous[missing + 1 :]
            error = sum(abs(old.center_x - new.center_x) for old, new in zip(remaining, current))
            if best is None or error < best[0]:
                best = (error, missing)
        if best is None:
            return None
        tile_width = median(tile.width for tile in previous)
        return best[1] if best[0] / max(1, len(current)) <= tile_width * 0.75 else None

    def _candidate_for_slot(self, previous: ZoneTile, candidates: list[ZoneTile]) -> ZoneTile:
        close = [
            tile for tile in candidates
            if abs(tile.center_x - previous.center_x) <= previous.width * 0.7
            and abs(tile.center_y - previous.center_y) <= previous.height * 0.7
        ]
        if close:
            tile = max(close, key=lambda item: item.confidence)
            return _copy_inferred(tile, "low_confidence_slot_candidate", tile.confidence)
        return _copy_inferred(previous, "hand_slot_history", previous.confidence * 0.65)


def _copy_inferred(tile: ZoneTile, reason: str, confidence: float) -> ZoneTile:
    values = tile.to_dict()
    values.update(inferred=True, source="inferred", reason=reason, confidence=confidence)
    return ZoneTile(**values)
