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
        self.last_recovery: dict[str, object] = {}

    def update(
        self,
        hand_tiles: list[ZoneTile],
        low_confidence_candidates: list[ZoneTile] | None = None,
    ) -> list[ZoneTile]:
        self._frame += 1
        observed = sorted((tile for tile in hand_tiles if not tile.inferred), key=lambda tile: tile.center_x)
        candidates = low_confidence_candidates or []
        seeded_candidates = self._gap_candidates(observed, candidates)
        if seeded_candidates:
            observed = sorted([*observed, *seeded_candidates], key=lambda tile: tile.center_x)
        self.last_recovery = {
            "observed_count": len(hand_tiles),
            "low_confidence_candidate_count": len(candidates),
            "recovered_from_candidate": len(seeded_candidates),
            "recovered_from_history": 0,
        }
        if not self._slots:
            self._slots = [self._observed_slot(tile) for tile in observed]
            return observed

        matched_slots, matched_observed = self._match_by_position(observed)
        for slot_index, observed_index in matched_slots.items():
            tile = observed[observed_index]
            slot = self._slots[slot_index]
            slot.tile = replace(tile, inferred=False)
            slot.missed_frames = 0
            slot.last_observed_frame = self._frame
            slot.inferred = False

        retained: list[HandSlotState] = []
        for index, slot in enumerate(self._slots):
            if index in matched_slots:
                retained.append(slot)
                continue
            slot.missed_frames += 1
            if slot.tile is None or slot.missed_frames > self.max_missed:
                continue
            candidate = self._candidate_for_slot(slot.tile, candidates)
            if candidate is not None:
                slot.tile = candidate
                slot.missed_frames = 0
                slot.last_observed_frame = self._frame
                self.last_recovery["recovered_from_candidate"] = int(
                    self.last_recovery["recovered_from_candidate"]
                ) + 1
            else:
                self.last_recovery["recovered_from_history"] = int(
                    self.last_recovery["recovered_from_history"]
                ) + 1
            slot.inferred = True
            retained.append(slot)

        for observed_index, tile in enumerate(observed):
            if observed_index not in matched_observed:
                retained.append(self._observed_slot(tile))

        self._slots = sorted(
            retained,
            key=lambda slot: slot.tile.center_x if slot.tile is not None else float("inf"),
        )
        self.last_recovery["output_count"] = len(self._slots)
        return self._output_tiles()

    def _match_by_position(self, observed: list[ZoneTile]) -> tuple[dict[int, int], set[int]]:
        pairs: list[tuple[float, int, int]] = []
        for slot_index, slot in enumerate(self._slots):
            if slot.tile is None:
                continue
            for observed_index, tile in enumerate(observed):
                width = max(slot.tile.width, tile.width)
                height = max(slot.tile.height, tile.height)
                dx = abs(slot.tile.center_x - tile.center_x)
                dy = abs(slot.tile.center_y - tile.center_y)
                if dx <= width * 0.65 and dy <= height * 0.65:
                    pairs.append((dx / width + dy / height, slot_index, observed_index))
        matched_slots: dict[int, int] = {}
        matched_observed: set[int] = set()
        for _, slot_index, observed_index in sorted(pairs):
            if slot_index in matched_slots or observed_index in matched_observed:
                continue
            matched_slots[slot_index] = observed_index
            matched_observed.add(observed_index)
        return matched_slots, matched_observed

    def _gap_candidates(
        self,
        observed: list[ZoneTile],
        candidates: list[ZoneTile],
    ) -> list[ZoneTile]:
        if len(observed) < 2 or len(observed) >= 14 or not candidates:
            return []
        tile_width = median(tile.width for tile in observed)
        tile_height = median(tile.height for tile in observed)
        row_y = median(tile.center_y for tile in observed)
        accepted: list[ZoneTile] = []
        occupied = list(observed)
        for candidate in sorted(candidates, key=lambda tile: tile.confidence, reverse=True):
            if abs(candidate.center_y - row_y) > tile_height * 0.45:
                continue
            if not (tile_width * 0.65 <= candidate.width <= tile_width * 1.35):
                continue
            if any(
                abs(candidate.center_x - tile.center_x) <= tile_width * 0.55
                and abs(candidate.center_y - tile.center_y) <= tile_height * 0.55
                for tile in occupied
            ):
                continue
            if any(
                slot.tile is not None
                and abs(candidate.center_x - slot.tile.center_x) <= tile_width * 0.8
                and abs(candidate.center_y - slot.tile.center_y) <= tile_height * 0.8
                for slot in self._slots
            ):
                continue
            ordered = sorted(occupied, key=lambda tile: tile.center_x)
            left = max(
                (tile for tile in ordered if tile.center_x < candidate.center_x),
                key=lambda tile: tile.center_x,
                default=None,
            )
            right = min(
                (tile for tile in ordered if tile.center_x > candidate.center_x),
                key=lambda tile: tile.center_x,
                default=None,
            )
            if left is None or right is None:
                continue
            left_gap = candidate.center_x - left.center_x
            right_gap = right.center_x - candidate.center_x
            if not (
                tile_width * 0.70 <= left_gap <= tile_width * 1.45
                and tile_width * 0.70 <= right_gap <= tile_width * 1.45
            ):
                continue
            recovered = replace(
                candidate,
                inferred=True,
                source="inferred",
                reason="low_confidence_geometric_gap",
            )
            accepted.append(recovered)
            occupied.append(recovered)
            if len(occupied) >= 14:
                break
        return accepted

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
