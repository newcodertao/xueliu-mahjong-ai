from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.fan_calc import FanResult, calculate_basic_fan
from xueliu_ai.mahjong.rules_xueliu import XueliuRuleProfile, load_rule_profile, missing_suit_tiles
from xueliu_ai.mahjong.scoring import fan_to_base_score


@dataclass(frozen=True)
class SettlementEstimate:
    legal: bool
    fan_result: FanResult
    base_score: int
    total_score: int
    reasons: tuple[str, ...] = ()


def estimate_win_settlement(
    tiles: list[str],
    *,
    missing_suit: str | None = None,
    open_melds: int = 0,
    self_draw: bool = False,
    rules: XueliuRuleProfile | None = None,
) -> SettlementEstimate:
    rules = rules or load_rule_profile()
    if rules.check_forbidden_suit and missing_suit_tiles(tiles, missing_suit):
        return SettlementEstimate(False, FanResult([], 0), 0, 0, ("forbidden_suit_remaining",))
    result = calculate_basic_fan(tiles, open_melds, rules.pattern_fans)
    capped = min(result.fan, rules.fan_cap)
    base = fan_to_base_score(capped)
    multiplier = 3 if self_draw else 1
    return SettlementEstimate(True, result, base, base * multiplier)


def estimate_kong_score(kong_type: str, rules: XueliuRuleProfile | None = None) -> int:
    rules = rules or load_rule_profile()
    return int((rules.kong_scores or {}).get(kong_type, 0))
