from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from xueliu_ai.realtime_table import TableZones
from xueliu_ai.table.game_phase import expected_hand_counts


@dataclass(frozen=True)
class MyAreaAnalysis:
    concealed_hand: list[str]
    drawn_tile: str | None
    meld_tiles: list[str]
    meld_group_count: int
    expected_counts: tuple[int, int]
    legal_count: bool
    uncertain_tiles: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "concealed_hand": self.concealed_hand,
            "drawn_tile": self.drawn_tile,
            "meld_tiles": self.meld_tiles,
            "meld_group_count": self.meld_group_count,
            "expected_counts": list(self.expected_counts),
            "legal_count": self.legal_count,
            "uncertain_tiles": self.uncertain_tiles,
        }


def analyze_my_area(zones: TableZones) -> MyAreaAnalysis:
    meld_group_count = sum(group.open_meld_count for group in zones.meld_groups if group.zone == "bottom_melds")
    if not zones.meld_groups and zones.bottom_melds:
        # Compatibility for older serialized/test states that predate MeldGroup.
        counts = Counter(zones.bottom_melds)
        meld_group_count = sum(1 for count in counts.values() if count in (3, 4))
    meld_group_count = min(4, meld_group_count)
    concealed_count, drawn_count = expected_hand_counts(meld_group_count)
    drawn_tile = zones.hand[-1] if len(zones.hand) == drawn_count and zones.hand else None
    legal_count = len(zones.hand) in (concealed_count, drawn_count)
    uncertain_tiles: list[str] = []
    if not legal_count:
        uncertain_tiles = list(zones.hand)

    return MyAreaAnalysis(
        concealed_hand=zones.hand,
        drawn_tile=drawn_tile,
        meld_tiles=zones.bottom_melds,
        meld_group_count=meld_group_count,
        expected_counts=(concealed_count, drawn_count),
        legal_count=legal_count,
        uncertain_tiles=uncertain_tiles,
    )
