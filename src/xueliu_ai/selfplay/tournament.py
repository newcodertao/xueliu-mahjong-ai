from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import mean

from xueliu_ai.selfplay.agents import SelfPlayAgent
from xueliu_ai.selfplay.environment import GameResult, XueliuSelfPlayEnv


@dataclass(frozen=True)
class AgentStats:
    name: str
    games: int
    average_score: float
    win_rate: float
    average_hu_count: float


@dataclass(frozen=True)
class TournamentResult:
    games: int
    seed: int
    agents: tuple[AgentStats, ...]
    game_results: tuple[GameResult, ...]


def run_tournament(
    agents: list[SelfPlayAgent],
    *,
    games: int = 100,
    seed: int = 20260717,
    workers: int = 1,
    keep_game_results: bool = False,
) -> TournamentResult:
    if len(agents) != 4:
        raise ValueError("Tournament requires four agent entries")
    jobs = []
    for game in range(games):
        rotation = game % 4
        lineup = [agents[(seat + rotation) % 4] for seat in range(4)]
        jobs.append((lineup, seed + game * 7919, rotation))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(_play_job, jobs))
    else:
        completed = [_play_job(job) for job in jobs]

    scores = [[] for _ in agents]
    wins = [[] for _ in agents]
    hu_counts = [[] for _ in agents]
    results = []
    for rotation, result in completed:
        results.append(result)
        winner_seats = set(result.winners)
        for seat in range(4):
            agent_index = (seat + rotation) % 4
            scores[agent_index].append(result.scores[seat])
            wins[agent_index].append(1.0 if seat in winner_seats else 0.0)
            hu_counts[agent_index].append(result.wins[seat])
    stats = tuple(
        AgentStats(
            name=agents[index].name,
            games=len(scores[index]),
            average_score=mean(scores[index]) if scores[index] else 0.0,
            win_rate=mean(wins[index]) if wins[index] else 0.0,
            average_hu_count=mean(hu_counts[index]) if hu_counts[index] else 0.0,
        )
        for index in range(4)
    )
    return TournamentResult(
        games,
        seed,
        stats,
        tuple(results) if keep_game_results else (),
    )


def _play_job(job: tuple[list[SelfPlayAgent], int, int]) -> tuple[int, GameResult]:
    agents, seed, rotation = job
    return rotation, XueliuSelfPlayEnv(agents).play(seed)
