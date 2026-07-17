from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from xueliu_ai.mahjong.tiles import TILE_NAMES


PLAYERS = ("self", "left", "opposite", "right")


@dataclass(frozen=True)
class StrategyContext:
    hand: tuple[str, ...]
    missing_suit: str | None = None
    melds: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    discards: dict[str, tuple[str, ...]] = field(default_factory=dict)
    inferred_missing_suits: dict[str, str | None] = field(default_factory=dict)
    wall_remaining: int | None = None
    turn_index: int = 0
    phase: str = "unknown"
    wins_by_player: dict[str, int] = field(default_factory=dict)
    recognition_quality: float = 1.0
    inferred_tile_count: int = 0
    unknown_tile_count: int = 0
    rules_version: str = "xueliu-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "missing_suit", self.missing_suit.upper() if self.missing_suit else None)

    @property
    def own_open_melds(self) -> int:
        return len(self.melds.get("self", ()))

    def visible_counts(self, *, include_hand: bool = False) -> dict[str, int]:
        counts: Counter[str] = Counter(self.hand if include_hand else ())
        for piles in self.discards.values():
            counts.update(piles)
        for groups in self.melds.values():
            for group in groups:
                counts.update(group)
        return dict(counts)

    def remaining_counts(self) -> dict[str, int]:
        visible = self.visible_counts(include_hand=True)
        return {tile: max(0, 4 - visible.get(tile, 0)) for tile in TILE_NAMES}

    @classmethod
    def from_structured_state(
        cls,
        state: Any,
        *,
        missing_suit: str | None = None,
        phase: str = "unknown",
        recognition_quality: float = 1.0,
    ) -> "StrategyContext":
        zones = state.zones
        melds: dict[str, list[tuple[str, ...]]] = {player: [] for player in PLAYERS}
        zone_players = {
            "bottom_melds": "self",
            "left_melds": "left",
            "top_melds": "opposite",
            "right_melds": "right",
        }
        for group in state.meld_groups:
            player = zone_players.get(group.zone)
            if player:
                melds[player].append(tuple(tile.label for tile in group.logical_tiles))
        discards = {
            "self": tuple(zones.my_discards),
            "left": tuple(zones.left_discards),
            "opposite": tuple(zones.top_discards),
            "right": tuple(zones.right_discards),
        }
        return cls(
            hand=tuple(zones.hand),
            missing_suit=missing_suit,
            melds={player: tuple(groups) for player, groups in melds.items()},
            discards=discards,
            phase=phase,
            recognition_quality=recognition_quality,
            inferred_tile_count=state.inferred_tile_count,
            unknown_tile_count=len(zones.unknown_tiles) + len(zones.candidate_meld_tiles),
        )
