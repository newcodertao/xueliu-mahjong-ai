from __future__ import annotations

from collections import Counter
from dataclasses import replace

from xueliu_ai.table.hand_slot_tracker import HandSlotTracker
from xueliu_ai.table.meld_grouper import group_melds
from xueliu_ai.table.structured_types import MeldGroup, MeldKind, StructuredTableState, ZoneTile
from xueliu_ai.table.tile_tracker import TileTracker
from xueliu_ai.vision.detection_types import Detection


class TableStateFusion:
    """Build one coherent table state from tracked detections and regrouped melds."""

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
        self._meld_history: dict[tuple, tuple[int, MeldKind | None]] = {}
        self.last_state: StructuredTableState | None = None

    def update(self, zones, low_confidence_hand_tiles=None):
        tracked_input = [
            tile for tile in zones.zone_tiles if tile.zone in self.tracked_zones and not tile.inferred
        ]
        tracked_tiles = self.tracker.update(tracked_input)
        by_zone = {
            zone: [tile for tile in tracked_tiles if tile.zone == zone]
            for zone in self.tracked_zones
        }
        observed_hand = [tile for tile in by_zone["hand"] if not tile.inferred]
        by_zone["hand"] = self.hand_slots.update(observed_hand, low_confidence_hand_tiles)

        meld_groups: list[MeldGroup] = []
        self._active_meld_keys: set[tuple] = set()
        for zone, axis in self.meld_axes.items():
            observed = [tile for tile in by_zone[zone] if not tile.inferred]
            grouped = group_melds([_to_detection(tile) for tile in observed], zone, axis, source="fusion")
            meld_groups.extend(self._restore_and_confirm_groups(grouped.groups, observed))
        self._meld_history = {
            key: value for key, value in self._meld_history.items() if key in self._active_meld_keys
        }

        logical_meld_tiles = [tile for group in meld_groups for tile in group.logical_tiles]
        untouched = [tile for tile in zones.zone_tiles if tile.zone not in self.tracked_zones]
        zone_tiles = [
            *untouched,
            *by_zone["hand"],
            *logical_meld_tiles,
            *by_zone["center_discards"],
        ]
        fused_zones = replace(
            zones,
            hand=_labels(by_zone["hand"], horizontal=True),
            bottom_melds=_group_labels(meld_groups, "bottom_melds"),
            left_melds=_group_labels(meld_groups, "left_melds"),
            top_melds=_group_labels(meld_groups, "top_melds"),
            right_melds=_group_labels(meld_groups, "right_melds"),
            center_discards=_labels(by_zone["center_discards"], horizontal=False),
            zone_tiles=zone_tiles,
            meld_groups=meld_groups,
        )
        confirmed = sum(1 for group in meld_groups if group.zone == "bottom_melds" and group.is_confirmed)
        suspected = sum(1 for group in meld_groups if group.zone == "bottom_melds" and group.is_suspected)
        state = StructuredTableState(
            zones=fused_zones,
            meld_groups=meld_groups,
            confirmed_open_melds=confirmed,
            suspected_open_melds=suspected,
            observed_visible_counts=_visible_counts(fused_zones, logical=False),
            logical_visible_counts=_visible_counts(fused_zones, logical=True),
        )
        self.last_state = state
        return fused_zones

    def _restore_and_confirm_groups(
        self,
        groups: list[MeldGroup],
        source_tiles: list[ZoneTile],
    ) -> list[MeldGroup]:
        result: list[MeldGroup] = []
        for group in groups:
            observed = _restore_observed_tiles(group, source_tiles)
            group = replace(group, observed_tiles=observed)
            key = _meld_key(group)
            self._active_meld_keys.add(key)
            previous_frames, previous_kind = self._meld_history.get(key, (0, None))
            frames = previous_frames + 1
            kind = group.kind
            if group.is_suspected:
                if previous_kind in {MeldKind.PONG, MeldKind.KONG}:
                    kind = previous_kind
                elif frames >= self.meld_confirmation_frames:
                    kind = MeldKind.PONG if group.kind == MeldKind.SUSPECTED_PONG else MeldKind.KONG
            confirmed_kind = kind if kind in {MeldKind.PONG, MeldKind.KONG} else previous_kind
            self._meld_history[key] = (frames, confirmed_kind)
            result.append(replace(group, kind=kind))
        return result


def _to_detection(tile: ZoneTile) -> Detection:
    return Detection(tile.label, tile.confidence, tile.x1, tile.y1, tile.x2, tile.y2)


def _restore_observed_tiles(group: MeldGroup, source_tiles: list[ZoneTile]) -> list[ZoneTile]:
    remaining = list(source_tiles)
    restored: list[ZoneTile] = []
    for grouped_tile in group.observed_tiles:
        matches = [
            tile for tile in remaining
            if tile.label == grouped_tile.label
        ]
        if not matches:
            restored.append(grouped_tile)
            continue
        source = min(
            matches,
            key=lambda tile: abs(tile.center_x - grouped_tile.center_x) + abs(tile.center_y - grouped_tile.center_y),
        )
        remaining.remove(source)
        restored.append(replace(source, zone=group.zone, group_id=group.group_id, reason="fused_meld_group"))
    return restored


def _meld_key(group: MeldGroup) -> tuple:
    tiles = group.observed_tiles or group.logical_tiles
    center_x = sum(tile.center_x for tile in tiles) / max(1, len(tiles))
    center_y = sum(tile.center_y for tile in tiles) / max(1, len(tiles))
    size = max(20.0, sum(max(tile.width, tile.height) for tile in tiles) / max(1, len(tiles)))
    return group.zone, group.label, round(center_x / size), round(center_y / size)


def _group_labels(groups: list[MeldGroup], zone: str) -> list[str]:
    return [tile.label for group in groups if group.zone == zone for tile in group.logical_tiles]


def _labels(tiles, horizontal: bool) -> list[str]:
    key = (lambda tile: (tile.center_x, tile.center_y)) if horizontal else (lambda tile: (tile.center_y, tile.center_x))
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
