from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from xueliu_ai.table.hand_slot_tracker import HandSlotTracker
from xueliu_ai.table.meld_grouper import MeldGroupingResult, group_melds
from xueliu_ai.table.structured_types import MeldGroup, MeldKind, StructuredTableState, ZoneTile
from xueliu_ai.table.tile_tracker import TileTracker
from xueliu_ai.vision.detection_types import Detection


@dataclass
class MeldHistoryState:
    group_id: str
    stable_frames: int
    missed_frames: int
    last_center: tuple[float, float]
    tile_size: float
    zone: str
    label: str
    kind: MeldKind | None


class TableStateFusion:
    """Build one coherent table state from tracked detections and final zones."""

    tracked_zones = (
        "hand",
        "bottom_melds",
        "left_melds",
        "top_melds",
        "right_melds",
        "center_discards",
    )
    meld_axes = {
        "bottom_melds": "horizontal",
        "left_melds": "vertical",
        "top_melds": "horizontal",
        "right_melds": "vertical",
    }

    def __init__(self, max_missed: int = 2, meld_confirmation_frames: int = 3) -> None:
        self.tracker = TileTracker(max_missed=max_missed)
        self.hand_slots = HandSlotTracker(max_missed=max_missed)
        self.meld_confirmation_frames = meld_confirmation_frames
        self._meld_history: dict[str, MeldHistoryState] = {}
        self._next_meld_id = 1
        self._meld_history_gap = False
        self.last_state: StructuredTableState | None = None

    @property
    def hand_recovery_info(self) -> dict[str, object]:
        return dict(self.hand_slots.last_recovery)

    def update(
        self,
        zones,
        low_confidence_hand_tiles=None,
        low_confidence_meld_tiles=None,
        *,
        finalize: bool = True,
    ):
        """Fuse tile tracks; optionally finalize this frame and advance meld history."""
        tracked_input = [
            tile for tile in zones.zone_tiles if tile.zone in self.tracked_zones and not tile.inferred
        ]
        for tile in zones.zone_tiles:
            origin_zone = _candidate_meld_origin(tile)
            if origin_zone is None:
                continue
            restored_candidate = replace(
                tile,
                zone=origin_zone,
                reason="candidate_meld_retry",
            )
            if not _overlaps_existing(restored_candidate, tracked_input):
                tracked_input.append(restored_candidate)
        for tile in low_confidence_meld_tiles or []:
            if tile.zone not in self.meld_axes:
                continue
            if not _overlaps_existing(tile, tracked_input):
                tracked_input.append(tile)
        tracked_tiles = self.tracker.update(tracked_input)
        by_zone = {
            zone: [tile for tile in tracked_tiles if tile.zone == zone]
            for zone in self.tracked_zones
        }
        observed_hand = [tile for tile in by_zone["hand"] if not tile.inferred]
        by_zone["hand"] = self.hand_slots.update(observed_hand, low_confidence_hand_tiles)

        preliminary_groups, isolated_tiles = self._group_zone_tiles(by_zone, source="fusion")
        logical_meld_tiles = [tile for group in preliminary_groups for tile in group.logical_tiles]
        untouched = [
            tile
            for tile in zones.zone_tiles
            if tile.zone not in self.tracked_zones and _candidate_meld_origin(tile) is None
        ]
        untouched_unknown = [tile for tile in untouched if tile.zone == "unknown_tiles"]
        other_untouched = [tile for tile in untouched if tile.zone != "unknown_tiles"]
        zone_tiles = [
            *other_untouched,
            *untouched_unknown,
            *isolated_tiles,
            *by_zone["hand"],
            *logical_meld_tiles,
            *by_zone["center_discards"],
        ]
        fused_zones = replace(
            zones,
            hand=_labels(by_zone["hand"], horizontal=True),
            bottom_melds=_group_labels(preliminary_groups, "bottom_melds"),
            left_melds=_group_labels(preliminary_groups, "left_melds"),
            top_melds=_group_labels(preliminary_groups, "top_melds"),
            right_melds=_group_labels(preliminary_groups, "right_melds"),
            center_discards=_labels(by_zone["center_discards"], horizontal=False),
            unknown_tiles=[tile.label for tile in zone_tiles if tile.zone == "unknown_tiles"],
            candidate_meld_tiles=[
                tile.label for tile in zone_tiles if tile.zone == "candidate_meld_tiles"
            ],
            zone_tiles=zone_tiles,
            meld_groups=preliminary_groups,
        )
        if not finalize:
            return fused_zones
        return self.build_structured_state(fused_zones).zones

    def build_structured_state(self, final_zones) -> StructuredTableState:
        """Rebuild all derived fields and advance meld history once for a final frame."""
        raw_groups, isolated_tiles = self._regroup_final_zones(final_zones)
        meld_groups = self._update_meld_history(raw_groups)
        meld_zone_names = set(self.meld_axes)
        non_meld_tiles = [
            tile for tile in final_zones.zone_tiles if tile.zone not in meld_zone_names
        ]
        logical_meld_tiles = [tile for group in meld_groups for tile in group.logical_tiles]
        existing_unknown = [tile for tile in non_meld_tiles if tile.zone == "unknown_tiles"]
        existing_candidates = [
            tile for tile in non_meld_tiles if tile.zone == "candidate_meld_tiles"
        ]
        unknown_labels = [tile.label for tile in existing_unknown]
        represented_unknown = Counter(unknown_labels)
        for label in final_zones.unknown_tiles:
            if represented_unknown[label] > 0:
                represented_unknown[label] -= 1
            else:
                unknown_labels.append(label)
        rebuilt_zone_tiles = [*non_meld_tiles, *isolated_tiles, *logical_meld_tiles]
        rebuilt_zones = replace(
            final_zones,
            bottom_melds=_group_labels(meld_groups, "bottom_melds"),
            left_melds=_group_labels(meld_groups, "left_melds"),
            top_melds=_group_labels(meld_groups, "top_melds"),
            right_melds=_group_labels(meld_groups, "right_melds"),
            unknown_tiles=unknown_labels,
            candidate_meld_tiles=[
                tile.label for tile in [*existing_candidates, *isolated_tiles]
            ],
            meld_groups=meld_groups,
            zone_tiles=rebuilt_zone_tiles,
        )
        state = StructuredTableState(
            zones=rebuilt_zones,
            meld_groups=meld_groups,
            confirmed_open_melds=sum(
                1 for group in meld_groups if group.zone == "bottom_melds" and group.is_confirmed
            ),
            suspected_open_melds=sum(
                1 for group in meld_groups if group.zone == "bottom_melds" and group.is_suspected
            ),
            observed_visible_counts=_visible_counts(rebuilt_zones, logical=False),
            logical_visible_counts=_visible_counts(rebuilt_zones, logical=True),
            meld_history_transient=self._meld_history_gap,
        )
        self.last_state = state
        return state

    def _group_zone_tiles(
        self,
        by_zone: dict[str, list[ZoneTile]],
        *,
        source: str,
    ) -> tuple[list[MeldGroup], list[ZoneTile]]:
        groups: list[MeldGroup] = []
        isolated: list[ZoneTile] = []
        for zone, axis in self.meld_axes.items():
            observed = [tile for tile in by_zone[zone] if not tile.inferred]
            grouped = group_melds([_to_detection(tile) for tile in observed], zone, axis, source)
            restored_groups, restored_isolated = _restore_grouping(grouped, observed)
            groups.extend(restored_groups)
            isolated.extend(restored_isolated)
        return groups, isolated

    def _regroup_final_zones(self, zones) -> tuple[list[MeldGroup], list[ZoneTile]]:
        by_zone = {
            zone: [
                tile
                for tile in zones.zone_tiles
                if tile.zone == zone and not tile.inferred
            ]
            for zone in self.meld_axes
        }
        return self._group_zone_tiles(by_zone, source="final")

    def _update_meld_history(self, groups: list[MeldGroup]) -> list[MeldGroup]:
        unmatched_history = set(self._meld_history)
        active_history: set[str] = set()
        result: list[MeldGroup] = []
        for group in sorted(groups, key=_group_sort_key):
            history_id = self._match_history(group, unmatched_history)
            if history_id is None:
                history_id = f"meld_{self._next_meld_id}"
                self._next_meld_id += 1
                previous = None
            else:
                unmatched_history.remove(history_id)
                previous = self._meld_history[history_id]

            frames = (previous.stable_frames if previous else 0) + 1
            previous_kind = previous.kind if previous else None
            kind = group.kind
            if group.is_suspected:
                if group.conflicting_tiles:
                    kind = group.kind
                elif previous_kind in {MeldKind.PONG, MeldKind.KONG}:
                    kind = previous_kind
                elif frames >= self.meld_confirmation_frames:
                    kind = (
                        MeldKind.PONG
                        if group.kind == MeldKind.SUSPECTED_PONG
                        else MeldKind.KONG
                    )
            confirmed_kind = kind if kind in {MeldKind.PONG, MeldKind.KONG} else previous_kind
            assigned = _assign_group_id(replace(group, kind=kind), history_id)
            center = _meld_center(assigned)
            self._meld_history[history_id] = MeldHistoryState(
                group_id=history_id,
                stable_frames=frames,
                missed_frames=0,
                last_center=center,
                tile_size=_meld_size(assigned),
                zone=assigned.zone,
                label=assigned.label,
                kind=confirmed_kind,
            )
            active_history.add(history_id)
            result.append(assigned)

        self._meld_history_gap = False
        for history_id in list(self._meld_history):
            if history_id in active_history:
                continue
            history = self._meld_history[history_id]
            history.missed_frames += 1
            if history.missed_frames > 2:
                del self._meld_history[history_id]
            else:
                self._meld_history_gap = True
        return result

    def _match_history(self, group: MeldGroup, candidates: set[str]) -> str | None:
        center = _meld_center(group)
        size = _meld_size(group)
        best: tuple[float, str] | None = None
        for history_id in candidates:
            history = self._meld_history[history_id]
            if history.zone != group.zone:
                continue
            distance = (
                (center[0] - history.last_center[0]) ** 2
                + (center[1] - history.last_center[1]) ** 2
            ) ** 0.5
            scale = max(20.0, size, history.tile_size)
            if distance > scale * 1.75:
                continue
            label_penalty = 0.75 if history.label != group.label else 0.0
            score = distance / scale + label_penalty
            if best is None or score < best[0]:
                best = (score, history_id)
        return best[1] if best else None


def _to_detection(tile: ZoneTile) -> Detection:
    return Detection(tile.label, tile.confidence, tile.x1, tile.y1, tile.x2, tile.y2)


def _restore_grouping(
    grouped: MeldGroupingResult,
    source_tiles: list[ZoneTile],
) -> tuple[list[MeldGroup], list[ZoneTile]]:
    remaining = list(source_tiles)
    restored_groups: list[MeldGroup] = []
    for group in grouped.groups:
        observed: list[ZoneTile] = []
        for grouped_tile in group.observed_tiles:
            source = _take_nearest(remaining, grouped_tile)
            if source is None:
                observed.append(grouped_tile)
                continue
            observed.append(
                replace(
                    source,
                    zone=group.zone,
                    group_id=group.group_id,
                    reason="fused_meld_group",
                )
            )
        conflict_labels = Counter(tile.label for tile in group.conflicting_tiles)
        conflicts: list[ZoneTile] = []
        for tile in observed:
            if conflict_labels[tile.label] > 0:
                conflicts.append(tile)
                conflict_labels[tile.label] -= 1
        restored_groups.append(
            replace(group, observed_tiles=observed, conflicting_tiles=conflicts)
        )

    restored_isolated: list[ZoneTile] = []
    for isolated in grouped.isolated_tiles:
        source = _take_nearest(remaining, isolated)
        if source is None:
            restored_isolated.append(isolated)
            continue
        restored_isolated.append(
            replace(
                source,
                zone="candidate_meld_tiles",
                group_id=None,
                reason=isolated.reason,
            )
        )
    return restored_groups, restored_isolated


def _take_nearest(remaining: list[ZoneTile], target: ZoneTile) -> ZoneTile | None:
    if not remaining:
        return None
    source = min(
        remaining,
        key=lambda tile: (
            abs(tile.center_x - target.center_x)
            + abs(tile.center_y - target.center_y)
            + (1000 if tile.label != target.label else 0)
        ),
    )
    remaining.remove(source)
    return source


def _assign_group_id(group: MeldGroup, group_id: str) -> MeldGroup:
    observed = [replace(tile, group_id=group_id) for tile in group.observed_tiles]
    inferred = [replace(tile, group_id=group_id) for tile in group.inferred_tiles]
    conflict_labels = Counter(tile.label for tile in group.conflicting_tiles)
    conflicts: list[ZoneTile] = []
    for tile in observed:
        if conflict_labels[tile.label] > 0:
            conflicts.append(tile)
            conflict_labels[tile.label] -= 1
    return replace(
        group,
        group_id=group_id,
        observed_tiles=observed,
        inferred_tiles=inferred,
        conflicting_tiles=conflicts,
    )


def _meld_center(group: MeldGroup) -> tuple[float, float]:
    tiles = group.observed_tiles or group.logical_tiles
    return (
        sum(tile.center_x for tile in tiles) / max(1, len(tiles)),
        sum(tile.center_y for tile in tiles) / max(1, len(tiles)),
    )


def _meld_size(group: MeldGroup) -> float:
    tiles = group.observed_tiles or group.logical_tiles
    return sum(max(tile.width, tile.height) for tile in tiles) / max(1, len(tiles))


def _group_sort_key(group: MeldGroup) -> tuple:
    center = _meld_center(group)
    return group.zone, center[1], center[0], group.label


def _group_labels(groups: list[MeldGroup], zone: str) -> list[str]:
    return [
        group.label
        for group in groups
        if group.zone == zone
        for _ in range(group.logical_count)
    ]


def _labels(tiles, horizontal: bool) -> list[str]:
    key = (
        (lambda tile: (tile.center_x, tile.center_y))
        if horizontal
        else (lambda tile: (tile.center_y, tile.center_x))
    )
    return [tile.label for tile in sorted(tiles, key=key)]


def _visible_counts(zones, logical: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tile in zones.zone_tiles:
        if tile.zone == "center_discards" and (logical or not tile.inferred):
            counts[tile.label] += 1
    for group in zones.meld_groups:
        tiles = group.logical_tiles if logical else group.observed_only_tiles
        counts.update(tile.label for tile in tiles)
    return dict(counts)


def _overlaps_existing(candidate: ZoneTile, existing: list[ZoneTile]) -> bool:
    for tile in existing:
        intersection_width = max(0.0, min(candidate.x2, tile.x2) - max(candidate.x1, tile.x1))
        intersection_height = max(0.0, min(candidate.y2, tile.y2) - max(candidate.y1, tile.y1))
        intersection = intersection_width * intersection_height
        if intersection <= 0:
            continue
        union = candidate.width * candidate.height + tile.width * tile.height - intersection
        if union > 0 and intersection / union >= 0.45:
            return True
    return False


def _candidate_meld_origin(tile: ZoneTile) -> str | None:
    if tile.zone != "candidate_meld_tiles" or not tile.group_id:
        return None
    return next(
        (zone for zone in TableStateFusion.meld_axes if tile.group_id.startswith(zone)),
        None,
    )
