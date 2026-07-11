from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from xueliu_ai.mahjong.tiles import SUITS, tile_rank, tile_suit, validate_tiles


SUIT_TEXT = {"W": "万", "T": "筒", "B": "条"}


@dataclass(frozen=True)
class MissingSuitAdvice:
    suit: str
    score_by_suit: dict[str, float]
    reason: str


def advise_missing_suit(tiles: list[str]) -> MissingSuitAdvice:
    validate_tiles(tiles)
    scores = {suit: _suit_keep_value([tile for tile in tiles if tile_suit(tile) == suit]) for suit in SUITS}
    suit = min(scores, key=lambda item: (scores[item], _suit_count(tiles, item)))
    reason = f"建议定缺{SUIT_TEXT[suit]}：该花色张数和连搭价值最低"
    return MissingSuitAdvice(suit=suit, score_by_suit=scores, reason=reason)


def _suit_count(tiles: list[str], suit: str) -> int:
    return sum(1 for tile in tiles if tile_suit(tile) == suit)


def _suit_keep_value(tiles: list[str]) -> float:
    if not tiles:
        return 0.0
    ranks = [tile_rank(tile) for tile in tiles]
    counts = Counter(ranks)
    value = len(tiles) * 10.0
    value += sum(8.0 for count in counts.values() if count >= 2)
    for rank in range(1, 8):
        if rank in counts and rank + 1 in counts and rank + 2 in counts:
            value += 12.0
    for rank in range(1, 9):
        if rank in counts and rank + 1 in counts:
            value += 4.0
    return value

