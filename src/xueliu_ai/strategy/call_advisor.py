from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallAdvice:
    action: str
    should_call: bool
    reason: str


def advise_peng(hand: list[str], tile: str) -> CallAdvice:
    count = hand.count(tile)
    return CallAdvice("peng", count >= 2, "手中已有两张可碰" if count >= 2 else "数量不足，不建议碰")
