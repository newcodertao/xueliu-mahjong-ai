from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from xueliu_ai.mahjong.rules_xueliu import XueliuRuleProfile, load_rule_profile, missing_suit_tiles
from xueliu_ai.mahjong.settlement import estimate_kong_score, estimate_win_settlement
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import TILE_NAMES
from xueliu_ai.selfplay.agents import AgentObservation, SelfPlayAgent


@dataclass
class SimPlayer:
    hand: list[str] = field(default_factory=list)
    missing_suit: str | None = None
    melds: list[tuple[str, ...]] = field(default_factory=list)
    discards: list[str] = field(default_factory=list)
    score: int = 0
    wins: int = 0


@dataclass(frozen=True)
class SimEvent:
    turn: int
    player: int
    action: str
    tile: str | None = None
    score_delta: int = 0
    targets: tuple[int, ...] = ()


@dataclass(frozen=True)
class GameResult:
    seed: int
    scores: tuple[int, ...]
    wins: tuple[int, ...]
    turns: int
    wall_remaining: int
    events: tuple[SimEvent, ...]

    @property
    def winners(self) -> tuple[int, ...]:
        best = max(self.scores)
        return tuple(index for index, score in enumerate(self.scores) if score == best)


class XueliuSelfPlayEnv:
    def __init__(
        self,
        agents: list[SelfPlayAgent],
        *,
        rules: XueliuRuleProfile | None = None,
        max_turns: int = 300,
        record_events: bool = False,
    ) -> None:
        if len(agents) != 4:
            raise ValueError("Xueliu self-play requires exactly four agents")
        self.agents = agents
        self.rules = rules or load_rule_profile()
        self.max_turns = max_turns
        self.record_events = record_events

    def play(self, seed: int) -> GameResult:
        rng = random.Random(seed)
        wall = [tile for tile in TILE_NAMES for _ in range(4)]
        rng.shuffle(wall)
        players = [SimPlayer() for _ in range(4)]
        for _ in range(13):
            for player in players:
                player.hand.append(wall.pop())
        for player in players:
            player.hand.sort()
        self._exchange_three(players, rng)
        for index, player in enumerate(players):
            player.missing_suit = self.agents[index].choose_missing_suit(player.hand)

        events: list[SimEvent] = []
        current = 0
        draw_required = True
        turn = 0
        while wall and turn < self.max_turns:
            turn += 1
            player = players[current]
            agent = self.agents[current]
            if draw_required:
                drawn = wall.pop()
                player.hand.append(drawn)
                self._event(events, turn, current, "draw", drawn)

            observation = self._observation(players, current, len(wall))
            kong_tile = next(
                (
                    tile
                    for tile, count in Counter(player.hand).items()
                    if count == 4 and agent.should_kong(observation, tile)
                ),
                None,
            )
            if kong_tile and wall:
                for _ in range(4):
                    player.hand.remove(kong_tile)
                player.melds.append((kong_tile,) * 4)
                kong_score = estimate_kong_score("an_kong", self.rules)
                for other in range(4):
                    if other == current:
                        continue
                    players[other].score -= kong_score
                    player.score += kong_score
                self._event(events, turn, current, "an_kong", kong_tile, kong_score * 3)
                player.hand.append(wall.pop())

            if self._can_hu(player):
                settlement = estimate_win_settlement(
                    player.hand,
                    missing_suit=player.missing_suit,
                    open_melds=len(player.melds),
                    self_draw=True,
                    rules=self.rules,
                )
                if settlement.legal:
                    for other in range(4):
                        if other == current:
                            continue
                        players[other].score -= settlement.base_score
                        player.score += settlement.base_score
                    player.wins += 1
                    self._event(events, turn, current, "self_draw_hu", score_delta=settlement.base_score * 3)

            observation = self._observation(players, current, len(wall))
            discard = agent.choose_discard(observation)
            if discard not in player.hand:
                raise RuntimeError(f"Agent {agent.name} selected unavailable tile {discard}")
            player.hand.remove(discard)
            player.discards.append(discard)
            self._event(events, turn, current, "discard", discard)

            hu_players = []
            for offset in range(1, 4):
                target = (current + offset) % 4
                candidate = players[target]
                candidate.hand.append(discard)
                can_hu = self._can_hu(candidate)
                candidate.hand.remove(discard)
                if can_hu:
                    hu_players.append(target)
            for target in hu_players:
                winner = players[target]
                winning_hand = [*winner.hand, discard]
                settlement = estimate_win_settlement(
                    winning_hand,
                    missing_suit=winner.missing_suit,
                    open_melds=len(winner.melds),
                    rules=self.rules,
                )
                if settlement.legal:
                    player.score -= settlement.base_score
                    winner.score += settlement.base_score
                    winner.wins += 1
                    self._event(events, turn, target, "discard_hu", discard, settlement.base_score, (current,))

            caller = None
            if not hu_players:
                for offset in range(1, 4):
                    target = (current + offset) % 4
                    candidate = players[target]
                    if candidate.hand.count(discard) < 2:
                        continue
                    obs = self._observation(players, target, len(wall))
                    if self.agents[target].should_peng(obs, discard):
                        caller = target
                        candidate.hand.remove(discard)
                        candidate.hand.remove(discard)
                        candidate.melds.append((discard,) * 3)
                        player.discards.pop()
                        self._event(events, turn, target, "peng", discard, targets=(current,))
                        break
            if caller is not None:
                current = caller
                draw_required = False
            else:
                current = (current + 1) % 4
                draw_required = True

            self._validate(players, wall)

        return GameResult(
            seed,
            tuple(player.score for player in players),
            tuple(player.wins for player in players),
            turn,
            len(wall),
            tuple(events),
        )

    def _exchange_three(self, players: list[SimPlayer], rng: random.Random) -> None:
        selections = [self.agents[index].exchange_three(player.hand, rng) for index, player in enumerate(players)]
        for player, selected in zip(players, selections):
            if len(selected) != 3 or len({tile[-1] for tile in selected}) != 1:
                raise RuntimeError("Agent returned illegal exchange-three selection")
            for tile in selected:
                player.hand.remove(tile)
        direction = rng.choice((1, 2, 3))
        for index, selected in enumerate(selections):
            players[(index + direction) % 4].hand.extend(selected)
        for player in players:
            player.hand.sort()

    def _observation(self, players: list[SimPlayer], player: int, wall_remaining: int) -> AgentObservation:
        visible = Counter()
        for item in players:
            visible.update(item.discards)
            for meld in item.melds:
                visible.update(meld)
        state = players[player]
        return AgentObservation(
            player,
            tuple(state.hand),
            state.missing_suit,
            len(state.melds),
            dict(visible),
            tuple(tuple(item.discards) for item in players),
            tuple(tuple(item.melds) for item in players),
            wall_remaining,
            tuple(item.score for item in players),
        )

    @staticmethod
    def _can_hu(player: SimPlayer) -> bool:
        return not missing_suit_tiles(player.hand, player.missing_suit) and best_shanten(
            player.hand, len(player.melds)
        ) < 0

    def _event(
        self,
        events: list[SimEvent],
        turn: int,
        player: int,
        action: str,
        tile: str | None = None,
        score_delta: int = 0,
        targets: tuple[int, ...] = (),
    ) -> None:
        if self.record_events:
            events.append(SimEvent(turn, player, action, tile, score_delta, targets))

    @staticmethod
    def _validate(players: list[SimPlayer], wall: list[str]) -> None:
        all_tiles = list(wall)
        for player in players:
            all_tiles.extend(player.hand)
            all_tiles.extend(player.discards)
            for meld in player.melds:
                all_tiles.extend(meld)
        counts = Counter(all_tiles)
        if any(count > 4 for count in counts.values()):
            raise RuntimeError("Tile conservation violated: more than four copies")
        if sum(counts.values()) != 108:
            raise RuntimeError(f"Tile conservation violated: expected 108, got {sum(counts.values())}")
