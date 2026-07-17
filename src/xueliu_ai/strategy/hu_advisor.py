from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.fan_calc import calculate_basic_fan
from xueliu_ai.mahjong.tiles import tile_suit


@dataclass(frozen=True)
class HuAdvice:
    can_hu: bool
    reason: str
    fan: int = 0
    patterns: tuple[str, ...] = ()


def advise_hu(hand: list[str], missing_suit: str | None = None, open_melds: int = 0) -> HuAdvice:
    if missing_suit and any(tile_suit(tile) == missing_suit.upper() for tile in hand):
        return HuAdvice(False, "手牌仍包含定缺花色，不能胡")
    can_hu = best_shanten(hand, open_melds=open_melds) < 0
    if not can_hu:
        return HuAdvice(False, "牌型未完成，不能胡")
    fan = calculate_basic_fan(hand, open_melds=open_melds)
    return HuAdvice(True, f"可以胡：{'、'.join(fan.patterns)}，预计 {fan.fan} 番", fan.fan, tuple(fan.patterns))
