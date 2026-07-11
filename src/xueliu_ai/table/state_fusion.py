from __future__ import annotations

from dataclasses import replace

from xueliu_ai.table.hand_slot_tracker import HandSlotTracker
from xueliu_ai.table.tile_tracker import TileTracker


class TableStateFusion:
    """Fuse per-frame assignments into short-lived, position-stable table state."""

    tracked_zones = (
        "hand",
        "bottom_melds",
        "left_melds",
        "top_melds",
        "right_melds",
        "center_discards",
    )

    def __init__(self, max_missed: int = 2) -> None:
        self.trackers = {zone: TileTracker(max_missed=max_missed) for zone in self.tracked_zones}
        self.hand_slots = HandSlotTracker(max_missed=max_missed)

    def update(self, zones):
        by_zone = {
            zone: [tile for tile in zones.zone_tiles if tile.zone == zone and not tile.inferred]
            for zone in self.tracked_zones
        }
        by_zone["hand"] = self.hand_slots.update(by_zone["hand"])
        tracked = {zone: self.trackers[zone].update(tiles) for zone, tiles in by_zone.items()}

        tracked_names = set(self.tracked_zones)
        untouched = [tile for tile in zones.zone_tiles if tile.zone not in tracked_names]
        zone_tiles = [*untouched, *(tile for zone in self.tracked_zones for tile in tracked[zone])]
        return replace(
            zones,
            hand=_labels(tracked["hand"], horizontal=True),
            bottom_melds=_labels(tracked["bottom_melds"], horizontal=True),
            left_melds=_labels(tracked["left_melds"], horizontal=False),
            top_melds=_labels(tracked["top_melds"], horizontal=True),
            right_melds=_labels(tracked["right_melds"], horizontal=False),
            center_discards=_labels(tracked["center_discards"], horizontal=False),
            zone_tiles=zone_tiles,
        )


def _labels(tiles, horizontal: bool) -> list[str]:
    key = (lambda tile: (tile.center_x, tile.center_y)) if horizontal else (lambda tile: (tile.center_y, tile.center_x))
    return [tile.label for tile in sorted(tiles, key=key)]
