from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import validate_tiles


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
    should = after <= before
    reason = "碰后不增加向听，且能固定一组面子" if should else "碰后向听变差或收益不明显"
    return CallAdvice("peng", should, reason, before_shanten=before, after_shanten=after)

