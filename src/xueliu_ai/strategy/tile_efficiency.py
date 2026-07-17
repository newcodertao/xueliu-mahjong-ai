from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.rules_xueliu import legal_discards
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import TILE_NAMES
from xueliu_ai.mahjong.ukeire import effective_draws


@dataclass(frozen=True)
class TileEfficiency:
    shanten: int
    effective_draws: dict[str, int]
    ukeire: int
    wait_type: str | None
    two_step_value: float
    two_step_improvements: dict[str, float]


def evaluate_tile_efficiency(
    tiles: list[str],
    visible_counts: dict[str, int] | None = None,
    *,
    open_melds: int = 0,
    missing_suit: str | None = None,
    include_two_step: bool = True,
) -> TileEfficiency:
    draws = effective_draws(tiles, visible_counts, open_melds=open_melds)
    shanten = best_shanten(tiles, open_melds=open_melds)
    two_step = (
        two_step_improvement_value(
            tiles,
            visible_counts,
            open_melds=open_melds,
            missing_suit=missing_suit,
        )
        if include_two_step and shanten > 0
        else {}
    )
    return TileEfficiency(
        shanten=shanten,
        effective_draws=draws,
        ukeire=sum(draws.values()),
        wait_type=classify_wait_quality(draws) if shanten == 0 else None,
        two_step_value=sum(two_step.values()),
        two_step_improvements=two_step,
    )


def classify_wait_quality(draws: dict[str, int]) -> str:
    live_types = sum(1 for remaining in draws.values() if remaining > 0)
    total = sum(draws.values())
    if live_types >= 3 or total >= 8:
        return "multi_wait"
    if live_types == 2 or total >= 5:
        return "dual_wait"
    if live_types == 1 and total >= 3:
        return "single_live_wait"
    if total > 0:
        return "thin_wait"
    return "dead_wait"


def wait_quality_score(wait_type: str | None) -> float:
    return {
        "multi_wait": 1.0,
        "dual_wait": 0.7,
        "single_live_wait": 0.4,
        "thin_wait": 0.15,
        "dead_wait": 0.0,
        None: 0.0,
    }[wait_type]


def two_step_improvement_value(
    tiles: list[str],
    visible_counts: dict[str, int] | None = None,
    *,
    open_melds: int = 0,
    missing_suit: str | None = None,
) -> dict[str, float]:
    """Estimate next-draw flexibility using the best legal discard after each draw."""
    visible = visible_counts or {}
    baseline = best_shanten(tiles, open_melds=open_melds)
    result: dict[str, float] = {}
    for draw in TILE_NAMES:
        remaining = max(0, 4 - tiles.count(draw) - int(visible.get(draw, 0)))
        if remaining == 0:
            continue
        drawn = [*tiles, draw]
        best_followup = 0.0
        for discard in set(legal_discards(drawn, missing_suit)):
            after = drawn.copy()
            after.remove(discard)
            shanten = best_shanten(after, open_melds=open_melds)
            draws = effective_draws(after, visible, open_melds=open_melds)
            improvement = max(0, baseline - shanten) * 20.0 + sum(draws.values())
            best_followup = max(best_followup, improvement)
        if best_followup > 0:
            result[draw] = best_followup * remaining
    return result
