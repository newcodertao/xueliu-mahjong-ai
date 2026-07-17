from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from xueliu_ai.mahjong.tiles import SUITS, tile_rank, tile_suit, validate_tiles
from xueliu_ai.strategy.missing_suit_advisor import SUIT_TEXT


@dataclass(frozen=True)
class ExchangeThreeAdvice:
    tiles: list[str]
    source_suit: str | None
    reason: str
    suggested_missing_suit: str | None = None
    candidate_scores: dict[tuple[str, ...], float] | None = None


def advise_exchange_three(tiles: list[str]) -> ExchangeThreeAdvice:
    validate_tiles(tiles)
    by_suit = {suit: [tile for tile in tiles if tile_suit(tile) == suit] for suit in SUITS}
    candidate_suits = [suit for suit, suit_tiles in by_suit.items() if len(suit_tiles) >= 3]
    if not candidate_suits:
        return ExchangeThreeAdvice([], None, "没有同花色三张，无法给出标准换三张建议")

    scores: dict[tuple[str, ...], float] = {}
    for suit in candidate_suits:
        indexed = list(enumerate(by_suit[suit]))
        seen: set[tuple[str, ...]] = set()
        for selection in combinations(indexed, 3):
            chosen = tuple(sorted((tile for _, tile in selection), key=tile_rank))
            if chosen in seen:
                continue
            seen.add(chosen)
            remaining = list(tiles)
            for tile in chosen:
                remaining.remove(tile)
            structural_loss = _suit_strength(by_suit[suit]) - _suit_strength(
                [tile for tile in remaining if tile_suit(tile) == suit]
            )
            isolation_bonus = sum(_exchange_priority(tile) for tile in chosen)
            duplicate_penalty = sum(max(0, chosen.count(tile) - 1) * 3 for tile in set(chosen))
            scores[chosen] = isolation_bonus * 5.0 - structural_loss - duplicate_penalty
    selected_tuple = max(scores, key=scores.get)
    selected = list(selected_tuple)
    suit = tile_suit(selected[0])
    remaining = list(tiles)
    for tile in selected:
        remaining.remove(tile)
    suggested_missing = min(
        SUITS,
        key=lambda item: (_suit_strength([tile for tile in remaining if tile_suit(tile) == item]), len([tile for tile in remaining if tile_suit(tile) == item])),
    )
    return ExchangeThreeAdvice(
        selected,
        suit,
        f"换出后保留的连接和对子价值最高，并倾向定缺{SUIT_TEXT[suggested_missing]}",
        suggested_missing_suit=suggested_missing,
        candidate_scores=scores,
    )


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
