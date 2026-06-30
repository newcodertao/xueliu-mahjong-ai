from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:
    discard: str
    expected_value: float
    simulations: int


def simulate_discards(candidates: list[str], simulations: int = 0) -> list[SimulationResult]:
    return [SimulationResult(discard=tile, expected_value=0.0, simulations=simulations) for tile in candidates]
