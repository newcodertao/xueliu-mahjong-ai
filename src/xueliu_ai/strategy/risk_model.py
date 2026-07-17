from __future__ import annotations

from xueliu_ai.mahjong.tiles import tile_suit
from xueliu_ai.strategy.context import StrategyContext


def simple_discard_risk(tile: str, visible_counts: dict[str, int] | None = None) -> float:
    visible_counts = visible_counts or {}
    seen = visible_counts.get(tile, 0)
    return max(0.0, 1.0 - seen * 0.2)


def discard_risk_by_player(tile: str, context: StrategyContext) -> dict[str, float]:
    visible = context.visible_counts(include_hand=True)
    remaining_ratio = max(0.0, 4 - visible.get(tile, 0)) / 4.0
    result: dict[str, float] = {}
    for player in ("left", "opposite", "right"):
        missing = context.inferred_missing_suits.get(player)
        if missing and tile_suit(tile) == missing:
            result[player] = 0.02
            continue
        discards = context.discards.get(player, ())
        melds = context.melds.get(player, ())
        same_suit_discards = sum(1 for item in discards if tile_suit(item) == tile_suit(tile))
        same_suit_melds = sum(
            1 for group in melds for item in group if tile_suit(item) == tile_suit(tile)
        )
        turn_pressure = min(0.25, context.turn_index * 0.012)
        flush_pressure = min(0.25, same_suit_melds * 0.04)
        discard_safety = min(0.35, same_suit_discards * 0.05)
        result[player] = max(
            0.0,
            min(1.0, 0.12 + remaining_ratio * 0.25 + turn_pressure + flush_pressure - discard_safety),
        )
    return result
