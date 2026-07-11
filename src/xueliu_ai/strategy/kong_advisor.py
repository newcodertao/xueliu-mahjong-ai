from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.tiles import validate_tiles


@dataclass(frozen=True)
class KongAdvice:
    should_kong: bool
    reason: str
    kong_type: str | None = None


def advise_kong(hand: list[str], tile: str | None = None, missing_suit: str | None = None) -> KongAdvice:
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
            return KongAdvice(True, f"{candidate} 四张齐，可以考虑暗杠", "an_kong")
        if target and count >= 3:
            return KongAdvice(True, f"手中已有三张 {candidate}，可考虑明杠", "ming_kong")
    return KongAdvice(False, "没有合适的杠牌机会")

