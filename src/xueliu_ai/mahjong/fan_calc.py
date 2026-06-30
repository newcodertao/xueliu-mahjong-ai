from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import is_complete_hand
from xueliu_ai.mahjong.tiles import tile_suit


@dataclass(frozen=True)
class FanResult:
    patterns: list[str]
    fan: int


def calculate_basic_fan(tiles: list[str]) -> FanResult:
    patterns: list[str] = []
    if is_complete_hand(tiles):
        patterns.append("胡牌")
    if len({tile_suit(tile) for tile in tiles}) == 1:
        patterns.append("清一色")
    return FanResult(patterns=patterns, fan=max(1, len(patterns)))
