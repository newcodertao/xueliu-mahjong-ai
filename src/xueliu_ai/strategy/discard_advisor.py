from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.rules_xueliu import legal_discards
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import tile_rank, tile_suit, validate_tiles
from xueliu_ai.mahjong.ukeire import effective_draw_count


@dataclass(frozen=True)
class DiscardCandidate:
    tile: str
    score: float
    shanten: int
    ukeire: int
    reason: str


@dataclass(frozen=True)
class DiscardAdvice:
    recommended: str
    candidates: list[DiscardCandidate]
    explanation: str


def advise_discard(
    tiles: list[str],
    missing_suit: str | None = None,
    visible_counts: dict[str, int] | None = None,
) -> DiscardAdvice:
    validate_tiles(tiles, allow_13_or_14=True)
    if len(tiles) != 14:
        raise ValueError(f"Discard advice needs 14 tiles, got {len(tiles)}")

    candidates: list[DiscardCandidate] = []
    restricted = legal_discards(tiles, missing_suit)
    for tile in restricted:
        after = tiles.copy()
        after.remove(tile)
        shanten = best_shanten(after)
        ukeire = effective_draw_count(after, visible_counts)
        score = -shanten * 100 + ukeire * 2 + _shape_score(after, tile, missing_suit)
        reason = f"打出后向听 {shanten}，有效进张 {ukeire} 枚"
        if missing_suit and tile_suit(tile) == missing_suit.upper():
            score += 1000
            reason = f"优先处理定缺门；{reason}"
        candidates.append(DiscardCandidate(tile, score, shanten, ukeire, reason))

    candidates.sort(key=lambda item: (item.score, item.ukeire, -_terminal_penalty(item.tile)), reverse=True)
    best = candidates[0]
    return DiscardAdvice(
        recommended=best.tile,
        candidates=candidates,
        explanation=f"推荐打 {best.tile}：{best.reason}。",
    )


def _shape_score(after: list[str], discarded: str, missing_suit: str | None) -> float:
    rank = tile_rank(discarded)
    score = 0.0
    if rank in (1, 9):
        score += 3
    if missing_suit and tile_suit(discarded) == missing_suit.upper():
        score += 5
    same_suit_ranks = [tile_rank(tile) for tile in after if tile_suit(tile) == tile_suit(discarded)]
    if rank - 1 in same_suit_ranks and rank + 1 in same_suit_ranks:
        score -= 8
    if rank - 1 in same_suit_ranks or rank + 1 in same_suit_ranks:
        score -= 3
    return score


def _terminal_penalty(tile: str) -> int:
    return 1 if tile_rank(tile) in (1, 9) else 0
