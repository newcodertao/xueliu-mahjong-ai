from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.tiles import validate_tiles
from xueliu_ai.mahjong.shanten import best_shanten


@dataclass(frozen=True)
class KongAdvice:
    should_kong: bool
    reason: str
    kong_type: str | None = None
    before_shanten: int | None = None
    after_shanten: int | None = None


def advise_kong(
    hand: list[str],
    tile: str | None = None,
    missing_suit: str | None = None,
    open_melds: int = 0,
) -> KongAdvice:
    validate_tiles(hand)
    target = (tile or "").upper() if tile else None
    candidates = [target] if target else sorted(set(hand))
    for candidate in candidates:
        if not candidate:
            continue
        if missing_suit and candidate.endswith(missing_suit.upper()):
            continue
        count = hand.count(candidate)
        if count >= 4:
            after_hand = list(hand)
            for _ in range(4):
                after_hand.remove(candidate)
            before = best_shanten(hand, open_melds=open_melds)
            after = best_shanten(after_hand, open_melds=open_melds + 1)
            should = after <= before
            reason = (
                f"{candidate} 暗杠后向听不变差，并获得杠分和补牌机会"
                if should
                else f"{candidate} 暗杠会使向听从 {before} 变为 {after}，建议保留"
            )
            return KongAdvice(should, reason, "an_kong", before, after)
        if target and count >= 3:
            after_hand = list(hand)
            for _ in range(3):
                after_hand.remove(candidate)
            before = best_shanten(hand, open_melds=open_melds)
            after = best_shanten(after_hand, open_melds=open_melds + 1)
            should = after <= before
            return KongAdvice(
                should,
                f"明杠后向听 {after}，杠前 {before}；{'收益可接受' if should else '牌效下降'}",
                "ming_kong",
                before,
                after,
            )
    return KongAdvice(False, "没有合适的杠牌机会")
