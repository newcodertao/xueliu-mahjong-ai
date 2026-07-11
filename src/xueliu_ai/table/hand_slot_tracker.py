from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

from xueliu_ai.table.structured_types import ZoneTile


@dataclass
class HandSlotState:
    tile: ZoneTile | None
    missed_frames: int
    last_observed_frame: int
    inferred: bool


class HandSlotTracker:
    def __init__(self, max_missed: int = 2) -> None:
        self.max_missed = max_missed
        self._slots: list[HandSlotState] = []
        self._frame = 0

    def update(
        self,
        hand_tiles: list[ZoneTile],
        low_confidence_candidates: list[ZoneTile] | None = None,
    ) -> list[ZoneTile]:
        self._frame += 1
        observed = sorted((tile for tile in hand_tiles if not tile.inferred), key=lambda tile: tile.center_x)
        candidates = low_confidence_candidates or []
        if not self._slots:
            self._slots = [self._observed_slot(tile) for tile in observed]
            return observed

        if len(observed) == len(self._slots):
            self._slots = [self._observed_slot(tile) for tile in observed]
            return observed

        if len(observed) == len(self._slots) - 1 and self._slots:
            missing_index = self._find_missing_slot(observed)
            if missing_index is not None:
                self._update_matched_slots(observed, missing_index)
                missing = self._slots[missing_index]
                missing.missed_frames += 1
                if missing.missed_frames <= self.max_missed and missing.tile is not None:
                    candidate = self._candidate_for_slot(missing.tile, candidates)
                    missing.tile = candidate or missing.tile
                    missing.inferred = True
                    return self._output_tiles()

                del self._slots[missing_index]
                self._slots = [self._observed_slot(tile) for tile in observed]
                return observed

        # A large structural change is not safe to infer. Accept observations as a new baseline.
        self._slots = [self._observed_slot(tile) for tile in observed]
        return observed

    def _observed_slot(self, tile: ZoneTile) -> HandSlotState:
        return HandSlotState(replace(tile, inferred=False), 0, self._frame, False)

    def _find_missing_slot(self, observed: list[ZoneTile]) -> int | None:
        previous = [slot.tile for slot in self._slots if slot.tile is not None]
        if len(previous) != len(self._slots):
            return None
        best: tuple[float, int] | None = None
        for missing in range(len(previous)):
            remaining = previous[:missing] + previous[missing + 1 :]
            error = sum(abs(old.center_x - new.center_x) for old, new in zip(remaining, observed))
            if best is None or error < best[0]:
                best = (error, missing)
        if best is None:
            return None
        tile_width = median(tile.width for tile in previous)
        return best[1] if best[0] / max(1, len(observed)) <= tile_width * 1.1 else None

    def _update_matched_slots(self, observed: list[ZoneTile], missing_index: int) -> None:
        observed_index = 0
        for index, slot in enumerate(self._slots):
            if index == missing_index:
                continue
            tile = observed[observed_index]
            observed_index += 1
            slot.tile = replace(tile, inferred=False)
            slot.missed_frames = 0
            slot.last_observed_frame = self._frame
            slot.inferred = False

    def _candidate_for_slot(self, previous: ZoneTile, candidates: list[ZoneTile]) -> ZoneTile | None:
        close = [
            tile
            for tile in candidates
            if abs(tile.center_x - previous.center_x) <= previous.width * 0.8
            and abs(tile.center_y - previous.center_y) <= previous.height * 0.8
        ]
        if not close:
            return None
        tile = max(close, key=lambda item: item.confidence)
        return replace(
            tile,
            inferred=True,
            source="inferred",
            reason="low_confidence_slot_candidate",
        )

    def _output_tiles(self) -> list[ZoneTile]:
        output: list[ZoneTile] = []
        for slot in self._slots:
            if slot.tile is None:
                continue
            if slot.inferred:
                output.append(
                    replace(
                        slot.tile,
                        inferred=True,
                        confidence=slot.tile.confidence * 0.65,
                        source="inferred",
                        reason=slot.tile.reason or "hand_slot_history",
                    )
                )
            else:
                output.append(slot.tile)
        return sorted(output, key=lambda tile: tile.center_x)
