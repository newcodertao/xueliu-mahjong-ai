from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.shanten import is_complete_hand


@dataclass(frozen=True)
class HuAdvice:
    can_hu: bool
    reason: str


def advise_hu(hand: list[str]) -> HuAdvice:
    can_hu = is_complete_hand(hand)
    return HuAdvice(can_hu, "牌型已完成，可以胡" if can_hu else "牌型未完成，不能胡")
