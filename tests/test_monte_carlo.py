import random

from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.monte_carlo import sample_opponent_hands, simulate_discards


def _context() -> StrategyContext:
    return StrategyContext(
        hand=("1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"),
        missing_suit="W",
        discards={"self": ("8B",), "left": (), "opposite": (), "right": ()},
        melds={"self": (), "left": (), "opposite": (), "right": ()},
    )


def test_monte_carlo_is_repeatable_for_fixed_seed() -> None:
    first = simulate_discards(["1W", "2W"], 20, context=_context(), seed=42, max_turns=4)
    second = simulate_discards(["1W", "2W"], 20, context=_context(), seed=42, max_turns=4)
    assert first == second
    assert all(result.simulations == 20 for result in first)


def test_sampled_opponent_hands_respect_four_copy_limit() -> None:
    context = _context()
    hands = sample_opponent_hands(context, random.Random(7))
    all_tiles = [*context.hand, *context.discards["self"]]
    for hand in hands.values():
        all_tiles.extend(hand)
    assert all(all_tiles.count(tile) <= 4 for tile in set(all_tiles))
