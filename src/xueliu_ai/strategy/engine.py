from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.discard_advisor import DiscardAdvice, advise_discard
from xueliu_ai.strategy.monte_carlo import SimulationResult, simulate_discards


@dataclass(frozen=True)
class StrategyDecision:
    advice: DiscardAdvice
    simulations: tuple[SimulationResult, ...] = ()
    mode: str = "deterministic"


class StrategyEngine:
    """Cached strategy facade shared by realtime UI and offline replay."""

    def __init__(self, cache_size: int = 256) -> None:
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[tuple[object, ...], StrategyDecision] = OrderedDict()

    def evaluate_discard(
        self,
        context: StrategyContext,
        *,
        monte_carlo_simulations: int = 0,
        monte_carlo_seed: int = 20260717,
        time_budget_ms: int | None = None,
    ) -> StrategyDecision:
        key = self._key(
            context,
            monte_carlo_simulations,
            monte_carlo_seed,
            time_budget_ms,
        )
        if key in self._cache:
            decision = self._cache.pop(key)
            self._cache[key] = decision
            return decision
        visible = context.visible_counts()
        advice = advise_discard(
            list(context.hand),
            context.missing_suit,
            visible_counts=visible,
            open_melds=context.own_open_melds,
            context=context,
        )
        simulations: tuple[SimulationResult, ...] = ()
        mode = "deterministic"
        if monte_carlo_simulations > 0:
            simulations = tuple(
                simulate_discards(
                    [candidate.tile for candidate in advice.candidates[:5]],
                    monte_carlo_simulations,
                    context=context,
                    seed=monte_carlo_seed,
                    time_budget_ms=time_budget_ms,
                )
            )
            mode = "monte_carlo"
        decision = StrategyDecision(advice, simulations, mode)
        self._cache[key] = decision
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return decision

    @staticmethod
    def _key(
        context: StrategyContext,
        simulations: int,
        seed: int,
        time_budget_ms: int | None,
    ) -> tuple[object, ...]:
        melds = tuple(
            (player, tuple(groups)) for player, groups in sorted(context.melds.items())
        )
        discards = tuple(
            (player, tuple(tiles)) for player, tiles in sorted(context.discards.items())
        )
        return (
            tuple(sorted(context.hand)),
            context.missing_suit,
            melds,
            discards,
            context.phase,
            round(context.recognition_quality, 2),
            context.inferred_tile_count,
            simulations,
            seed,
            time_budget_ms,
        )
