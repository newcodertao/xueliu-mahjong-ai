from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KongAdvice:
    should_kong: bool
    reason: str


def advise_kong(hand: list[str], tile: str) -> KongAdvice:
    count = hand.count(tile)
    return KongAdvice(count >= 4, "四张齐，可考虑杠" if count >= 4 else "未形成四张，不杠")
