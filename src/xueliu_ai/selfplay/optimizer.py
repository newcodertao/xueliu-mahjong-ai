from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from xueliu_ai.selfplay.agents import FastRuleAgent
from xueliu_ai.selfplay.tournament import AgentStats, run_tournament


@dataclass(frozen=True)
class OptimizationCandidate:
    parameters: dict[str, float]
    stats: AgentStats


@dataclass(frozen=True)
class OptimizationResult:
    seed: int
    games_per_candidate: int
    candidates: tuple[OptimizationCandidate, ...]

    @property
    def champion(self) -> OptimizationCandidate:
        return max(
            self.candidates,
            key=lambda item: (item.stats.average_score, item.stats.win_rate),
        )


def optimize_fast_agent(
    *,
    candidates: int = 12,
    games_per_candidate: int = 40,
    seed: int = 20260717,
    workers: int = 1,
) -> OptimizationResult:
    rng = random.Random(seed)
    baseline = FastRuleAgent(name="baseline")
    results = []
    for index in range(candidates):
        parameters = {
            "connection_weight": rng.uniform(0.8, 4.0),
            "duplicate_weight": rng.uniform(1.0, 6.0),
            "terminal_weight": rng.uniform(0.0, 3.0),
            "peng_threshold": rng.uniform(-2.0, 8.0),
        }
        candidate = FastRuleAgent(name=f"candidate-{index}", **parameters)
        tournament = run_tournament(
            [candidate, baseline, baseline, baseline],
            games=games_per_candidate,
            seed=seed + index * 104729,
            workers=workers,
        )
        results.append(OptimizationCandidate(parameters, tournament.agents[0]))
    return OptimizationResult(seed, games_per_candidate, tuple(results))


def write_optimization_manifest(path: str | Path, result: OptimizationResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": result.seed,
        "games_per_candidate": result.games_per_candidate,
        "champion": asdict(result.champion),
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
