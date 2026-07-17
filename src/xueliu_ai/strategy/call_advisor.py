from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import validate_tiles
from xueliu_ai.strategy.discard_advisor import advise_discard


@dataclass(frozen=True)
class CallAdvice:
    action: str
    should_call: bool
    reason: str
    before_shanten: int | None = None
    after_shanten: int | None = None


def advise_peng(hand: list[str], tile: str, missing_suit: str | None = None, open_melds: int = 0) -> CallAdvice:
    validate_tiles(hand)
    count = hand.count(tile)
    if count < 2:
        return CallAdvice("peng", False, "手中不足两张，不建议碰")
    if missing_suit and tile.endswith(missing_suit.upper()):
        return CallAdvice("peng", False, "这是定缺花色，优先打出，不建议碰")

    before = best_shanten(hand, open_melds=open_melds)
    after_hand = hand.copy()
    after_hand.remove(tile)
    after_hand.remove(tile)
    after = best_shanten(after_hand, open_melds=open_melds + 1)
    best_discard_score = None
    if len(after_hand) == 14 - (open_melds + 1) * 3:
        followup = advise_discard(after_hand, missing_suit, open_melds=open_melds + 1)
        best_discard_score = followup.candidates[0].score
    should = after < before or (after == before and best_discard_score is not None and best_discard_score > -50)
    reason = (
        f"碰后不增加向听，最佳后续弃牌评分 {best_discard_score:.1f}"
        if should and best_discard_score is not None
        else "碰后能降低向听并固定一组面子"
        if should
        else "碰后向听、进张或手牌灵活性收益不足"
    )
    return CallAdvice("peng", should, reason, before_shanten=before, after_shanten=after)
