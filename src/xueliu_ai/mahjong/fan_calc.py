from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import best_shanten, is_complete_hand
from xueliu_ai.mahjong.tiles import tile_suit, validate_tiles


@dataclass(frozen=True)
class FanResult:
    patterns: list[str]
    fan: int


def calculate_basic_fan(
    tiles: list[str],
    open_melds: int = 0,
    pattern_fans: dict[str, int] | None = None,
) -> FanResult:
    validate_tiles(tiles)
    patterns: list[str] = []
    complete = is_complete_hand(tiles) if open_melds == 0 else best_shanten(tiles, open_melds) < 0
    if complete:
        patterns.append("normal")
    if len({tile_suit(tile) for tile in tiles}) == 1:
        patterns.append("flush")
    if _is_seven_pairs(tiles) and open_melds == 0:
        patterns.append("seven_pairs")
    if _is_all_triplets(tiles, open_melds):
        patterns.append("all_triplets")
    if _has_quad_like_count(tiles):
        patterns.append("root")
    weights = pattern_fans or {
        "normal": 1,
        "flush": 2,
        "seven_pairs": 2,
        "all_triplets": 2,
        "root": 1,
    }
    unique = list(dict.fromkeys(patterns or ["normal"]))
    return FanResult(patterns=unique, fan=max(1, sum(weights.get(pattern, 0) for pattern in unique)))


def _is_seven_pairs(tiles: list[str]) -> bool:
    counts = Counter(tiles)
    return len(tiles) == 14 and len(counts) == 7 and all(count == 2 for count in counts.values())


def _has_quad_like_count(tiles: list[str]) -> bool:
    return any(count == 4 for count in Counter(tiles).values())


def _is_all_triplets(tiles: list[str], open_melds: int) -> bool:
    counts = Counter(tiles)
    pairs = sum(1 for count in counts.values() if count == 2)
    triplets = sum(count // 3 for count in counts.values())
    return pairs == 1 and triplets + open_melds == 4
