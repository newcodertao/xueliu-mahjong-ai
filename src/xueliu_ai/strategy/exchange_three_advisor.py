from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.tiles import SUITS, tile_rank, tile_suit, validate_tiles
from xueliu_ai.strategy.missing_suit_advisor import SUIT_TEXT


@dataclass(frozen=True)
class ExchangeThreeAdvice:
    tiles: list[str]
    source_suit: str | None
    reason: str


def advise_exchange_three(tiles: list[str]) -> ExchangeThreeAdvice:
    validate_tiles(tiles)
    by_suit = {suit: [tile for tile in tiles if tile_suit(tile) == suit] for suit in SUITS}
    candidate_suits = [suit for suit, suit_tiles in by_suit.items() if len(suit_tiles) >= 3]
    if not candidate_suits:
        return ExchangeThreeAdvice([], None, "没有同花色三张，无法给出标准换三张建议")

    suit = min(candidate_suits, key=lambda item: (_suit_strength(by_suit[item]), len(by_suit[item])))
    selected = sorted(by_suit[suit], key=_exchange_priority, reverse=True)[:3]
    selected = sorted(selected, key=lambda tile: (tile_suit(tile), tile_rank(tile)))
    return ExchangeThreeAdvice(selected, suit, f"优先换出{SUIT_TEXT[suit]}里连接价值最低的三张")


def _suit_strength(tiles: list[str]) -> float:
    ranks = [tile_rank(tile) for tile in tiles]
    value = len(tiles) * 10.0
    for rank in ranks:
        if rank in (1, 9):
            value -= 1.5
    for rank in set(ranks):
        if rank + 1 in ranks:
            value += 4.0
        if rank + 2 in ranks:
            value += 2.0
    return value


def _exchange_priority(tile: str) -> float:
    rank = tile_rank(tile)
    terminal_bonus = 3.0 if rank in (1, 9) else 0.0
    edge_bonus = 1.0 if rank in (2, 8) else 0.0
    return terminal_bonus + edge_bonus

