from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.table.structured_types import ZoneTile


@dataclass
class _Track:
    tile: ZoneTile
    missed: int = 0


class TileTracker:
    def __init__(self, max_missed: int = 2, max_center_distance: float = 1.6) -> None:
        self.max_missed = max_missed
        self.max_center_distance = max_center_distance
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, tiles: list[ZoneTile]) -> list[ZoneTile]:
        unmatched = set(self._tracks)
        output: list[ZoneTile] = []
        for tile in sorted(tiles, key=lambda item: (item.center_x, item.center_y)):
            track_id = self._best_match(tile, unmatched)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            else:
                unmatched.remove(track_id)
            tracked = _copy_tile(tile, track_id=track_id)
            self._tracks[track_id] = _Track(tracked)
            output.append(tracked)

        for track_id in list(unmatched):
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self._tracks[track_id]
                continue
            retained = _copy_tile(
                track.tile,
                confidence=track.tile.confidence * (0.72 ** track.missed),
                inferred=True,
                source="track_history",
                reason="short_detection_gap",
            )
            track.tile = retained
            output.append(retained)
        return output

    def _best_match(self, tile: ZoneTile, candidates: set[int]) -> int | None:
        best: tuple[float, int] | None = None
        scale = max(tile.width, tile.height)
        for track_id in candidates:
            previous = self._tracks[track_id].tile
            distance = ((previous.center_x - tile.center_x) ** 2 + (previous.center_y - tile.center_y) ** 2) ** 0.5
            if distance > scale * self.max_center_distance:
                continue
            size_ratio = max(previous.width / tile.width, tile.width / previous.width, previous.height / tile.height, tile.height / previous.height)
            if size_ratio > 1.8:
                continue
            label_penalty = 0.35 if previous.label != tile.label else 0.0
            zone_penalty = 0.2 if previous.zone != tile.zone else 0.0
            score = distance / scale + (1.0 - _iou(previous, tile)) + label_penalty + zone_penalty
            if best is None or score < best[0]:
                best = (score, track_id)
        return best[1] if best else None


def _copy_tile(tile: ZoneTile, **changes) -> ZoneTile:
    values = tile.to_dict()
    values.update(changes)
    return ZoneTile(**values)


def _iou(left: ZoneTile, right: ZoneTile) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0
