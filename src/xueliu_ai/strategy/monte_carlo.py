from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass

from xueliu_ai.mahjong.rules_xueliu import legal_discards
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import TILE_NAMES
from xueliu_ai.mahjong.tiles import tile_rank, tile_suit
from xueliu_ai.mahjong.ukeire import effective_draw_count
from xueliu_ai.strategy.context import StrategyContext


@dataclass(frozen=True)
class SimulationResult:
    discard: str
    expected_value: float
    simulations: int
    win_probability: float = 0.0
    ready_probability: float = 0.0
    average_turns: float | None = None
    standard_error: float = 0.0


def simulate_discards(
    candidates: list[str],
    simulations: int = 1000,
    *,
    context: StrategyContext | None = None,
    seed: int = 20260717,
    max_turns: int = 18,
    time_budget_ms: int | None = None,
) -> list[SimulationResult]:
    if context is None:
        return [SimulationResult(discard=tile, expected_value=0.0, simulations=0) for tile in candidates]
    unique = sorted(set(candidates))
    if not unique or simulations <= 0:
        return []
    deadline = time.perf_counter() + time_budget_ms / 1000 if time_budget_ms else None
    results = [
        _simulate_candidate(
            discard,
            context,
            simulations,
            random.Random(seed + index * 1009),
            max_turns,
            deadline,
        )
        for index, discard in enumerate(unique)
    ]
    return sorted(results, key=lambda item: item.expected_value, reverse=True)


def sample_opponent_hands(
    context: StrategyContext,
    rng: random.Random,
    hand_size: int = 13,
) -> dict[str, tuple[str, ...]]:
    pool = _remaining_pool(context)
    result: dict[str, tuple[str, ...]] = {}
    for player in ("left", "opposite", "right"):
        size = max(0, hand_size - len(context.melds.get(player, ())) * 3)
        missing = context.inferred_missing_suits.get(player)
        weighted = [tile for tile in pool if not missing or not tile.endswith(missing)]
        source = weighted if len(weighted) >= size else pool
        selected: list[str] = []
        for _ in range(min(size, len(source))):
            tile = rng.choice(source)
            source.remove(tile)
            pool.remove(tile)
            selected.append(tile)
        result[player] = tuple(sorted(selected))
    return result


def _simulate_candidate(
    discard: str,
    context: StrategyContext,
    simulations: int,
    rng: random.Random,
    max_turns: int,
    deadline: float | None,
) -> SimulationResult:
    outcomes: list[float] = []
    wins = 0
    ready = 0
    winning_turns: list[int] = []
    for _ in range(simulations):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        hand = list(context.hand)
        if discard not in hand:
            continue
        hand.remove(discard)
        pool = _remaining_pool(context)
        rng.shuffle(pool)
        outcome = 0.0
        for turn in range(1, min(max_turns, len(pool)) + 1):
            hand.append(pool.pop())
            if best_shanten(hand, open_melds=context.own_open_melds) < 0:
                wins += 1
                winning_turns.append(turn)
                outcome = 100.0 + (max_turns - turn) * 2.0
                break
            selected = _fast_best_discard(hand, context)
            hand.remove(selected)
        final_shanten = best_shanten(hand, open_melds=context.own_open_melds)
        if final_shanten <= 0:
            ready += 1
        if outcome == 0.0:
            outcome = -final_shanten * 12.0 + effective_draw_count(
                hand,
                context.visible_counts(),
                open_melds=context.own_open_melds,
            )
        outcomes.append(outcome)
    count = len(outcomes)
    if count == 0:
        return SimulationResult(discard, 0.0, 0)
    mean = sum(outcomes) / count
    variance = sum((value - mean) ** 2 for value in outcomes) / count
    return SimulationResult(
        discard=discard,
        expected_value=mean,
        simulations=count,
        win_probability=wins / count,
        ready_probability=ready / count,
        average_turns=sum(winning_turns) / len(winning_turns) if winning_turns else None,
        standard_error=(variance / count) ** 0.5,
    )


def _fast_best_discard(hand: list[str], context: StrategyContext) -> str:
    best: tuple[float, str] | None = None
    for tile in set(legal_discards(hand, context.missing_suit)):
        rank = tile_rank(tile)
        suit_ranks = [tile_rank(item) for item in hand if tile_suit(item) == tile_suit(tile)]
        connections = sum(1 for delta in (-2, -1, 1, 2) if rank + delta in suit_ranks)
        duplicates = hand.count(tile) - 1
        terminal = 1.5 if rank in (1, 9) else 0.0
        score = terminal - connections * 2.0 - duplicates * 3.0
        if best is None or score > best[0]:
            best = (score, tile)
    return best[1] if best else hand[-1]


def _remaining_pool(
    context: StrategyContext,
) -> list[str]:
    used = Counter(context.visible_counts(include_hand=True))
    return [tile for tile in TILE_NAMES for _ in range(max(0, 4 - used[tile]))]
