from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.config import load_yaml

from xueliu_ai.mahjong.tiles import tile_suit, validate_tiles


def missing_suit_tiles(tiles: list[str], missing_suit: str | None) -> list[str]:
    if not missing_suit:
        return []
    suit = missing_suit.upper()
    return [tile for tile in tiles if tile_suit(tile) == suit]


def legal_discards(tiles: list[str], missing_suit: str | None = None) -> list[str]:
    validate_tiles(tiles)
    missing = missing_suit_tiles(tiles, missing_suit)
    candidates = missing if missing else tiles
    return sorted(set(candidates), key=lambda tile: (tile_suit(tile), int(tile[0])))


@dataclass(frozen=True)
class XueliuRuleProfile:
    require_missing_suit: bool = True
    discard_missing_suit_first: bool = True
    allow_chi: bool = False
    continue_after_hu: bool = True
    fan_cap: int = 5
    check_forbidden_suit: bool = True
    check_ready_hand: bool = True
    pattern_fans: dict[str, int] | None = None
    kong_scores: dict[str, int] | None = None


def load_rule_profile(path: str = "configs/rule_xueliu.yaml") -> XueliuRuleProfile:
    values = load_yaml(path).get("rules", {})
    settlement = values.get("settlement", {})
    return XueliuRuleProfile(
        require_missing_suit=bool(values.get("require_missing_suit", True)),
        discard_missing_suit_first=bool(values.get("discard_missing_suit_first", True)),
        allow_chi=bool(values.get("allow_chi", False)),
        continue_after_hu=bool(values.get("continue_after_hu", True)),
        fan_cap=int(settlement.get("fan_cap", 5)),
        check_forbidden_suit=bool(settlement.get("check_forbidden_suit", True)),
        check_ready_hand=bool(settlement.get("check_ready_hand", True)),
        pattern_fans={str(key): int(value) for key, value in values.get("patterns", {}).items()},
        kong_scores={str(key): int(value) for key, value in settlement.get("kong_scores", {}).items()},
    )
