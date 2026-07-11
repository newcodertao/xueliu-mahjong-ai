from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import is_complete_hand
from xueliu_ai.mahjong.tiles import tile_suit, validate_tiles


@dataclass(frozen=True)
class FanResult:
    patterns: list[str]
    fan: int


def calculate_basic_fan(tiles: list[str], open_melds: int = 0) -> FanResult:
    validate_tiles(tiles)
    patterns: list[str] = []
    if open_melds == 0 and is_complete_hand(tiles):
        patterns.append("胡牌")
    if len({tile_suit(tile) for tile in tiles}) == 1:
        patterns.append("清一色")
    if _is_seven_pairs(tiles) and open_melds == 0:
        patterns.append("七对")
    if _has_quad_like_count(tiles):
        patterns.append("带根")
    return FanResult(patterns=patterns or ["普通"], fan=max(1, len(patterns)))


def _is_seven_pairs(tiles: list[str]) -> bool:
    counts = Counter(tiles)
    return len(tiles) == 14 and len(counts) == 7 and all(count == 2 for count in counts.values())


def _has_quad_like_count(tiles: list[str]) -> bool:
    return any(count == 4 for count in Counter(tiles).values())
