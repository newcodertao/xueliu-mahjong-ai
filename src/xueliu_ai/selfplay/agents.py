from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from xueliu_ai.mahjong.rules_xueliu import legal_discards
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import tile_rank, tile_suit
from xueliu_ai.mahjong.ukeire import effective_draw_count
from xueliu_ai.strategy.exchange_three_advisor import advise_exchange_three
from xueliu_ai.strategy.missing_suit_advisor import advise_missing_suit


@dataclass(frozen=True)
class AgentObservation:
    player: int
    hand: tuple[str, ...]
    missing_suit: str | None
    open_melds: int
    visible_counts: dict[str, int]
    discards: tuple[tuple[str, ...], ...]
    melds: tuple[tuple[tuple[str, ...], ...], ...]
    wall_remaining: int
    scores: tuple[int, ...]


class SelfPlayAgent(Protocol):
    name: str

    def exchange_three(self, hand: list[str], rng: random.Random) -> list[str]: ...

    def choose_missing_suit(self, hand: list[str]) -> str: ...

    def choose_discard(self, observation: AgentObservation) -> str: ...

    def should_peng(self, observation: AgentObservation, tile: str) -> bool: ...

    def should_kong(self, observation: AgentObservation, tile: str) -> bool: ...


@dataclass
class FastRuleAgent:
    name: str = "fast-rule"
    connection_weight: float = 2.0
    duplicate_weight: float = 3.0
    terminal_weight: float = 1.5
    peng_threshold: float = 0.0
    kong_enabled: bool = True

    def exchange_three(self, hand: list[str], rng: random.Random) -> list[str]:
        advice = advise_exchange_three(hand)
        return list(advice.tiles)

    def choose_missing_suit(self, hand: list[str]) -> str:
        return advise_missing_suit(hand).suit

    def choose_discard(self, observation: AgentObservation) -> str:
        candidates = legal_discards(list(observation.hand), observation.missing_suit)
        return max(candidates, key=lambda tile: (self._discard_score(tile, observation.hand), tile))

    def should_peng(self, observation: AgentObservation, tile: str) -> bool:
        if observation.hand.count(tile) < 2:
            return False
        if observation.missing_suit and tile_suit(tile) == observation.missing_suit:
            return False
        ranks = [tile_rank(item) for item in observation.hand if tile_suit(item) == tile_suit(tile)]
        rank = tile_rank(tile)
        connection_loss = sum(1 for delta in (-2, -1, 1, 2) if rank + delta in ranks)
        return self.duplicate_weight * 2 - connection_loss * self.connection_weight >= self.peng_threshold

    def should_kong(self, observation: AgentObservation, tile: str) -> bool:
        return self.kong_enabled and observation.hand.count(tile) == 4

    def _discard_score(self, tile: str, hand: tuple[str, ...]) -> float:
        rank = tile_rank(tile)
        ranks = [tile_rank(item) for item in hand if tile_suit(item) == tile_suit(tile)]
        connections = sum(1 for delta in (-2, -1, 1, 2) if rank + delta in ranks)
        duplicates = hand.count(tile) - 1
        terminal = self.terminal_weight if rank in (1, 9) else 0.0
        return terminal - connections * self.connection_weight - duplicates * self.duplicate_weight


@dataclass
class ValueAgent(FastRuleAgent):
    name: str = "value-agent"
    shanten_weight: float = 100.0
    ukeire_weight: float = 2.0

    def choose_discard(self, observation: AgentObservation) -> str:
        hand = list(observation.hand)
        best: tuple[float, str] | None = None
        for tile in legal_discards(hand, observation.missing_suit):
            after = list(hand)
            after.remove(tile)
            shanten = best_shanten(after, observation.open_melds)
            ukeire = effective_draw_count(after, observation.visible_counts, observation.open_melds)
            score = -shanten * self.shanten_weight + ukeire * self.ukeire_weight
            score += self._discard_score(tile, observation.hand)
            if best is None or score > best[0]:
                best = (score, tile)
        if best is None:
            raise RuntimeError("No legal discard")
        return best[1]


@dataclass
class RandomAgent(FastRuleAgent):
    name: str = "random"
    seed: int = 0

    def choose_discard(self, observation: AgentObservation) -> str:
        candidates = legal_discards(list(observation.hand), observation.missing_suit)
        state_seed = hash((self.seed, observation.player, observation.hand, observation.wall_remaining))
        return random.Random(state_seed).choice(candidates)

    def should_peng(self, observation: AgentObservation, tile: str) -> bool:
        return False

    def should_kong(self, observation: AgentObservation, tile: str) -> bool:
        return False
