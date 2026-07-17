from __future__ import annotations

from dataclasses import dataclass, field

from xueliu_ai.strategy.actions import StrategyAction


@dataclass(frozen=True)
class ScoreBreakdown:
    shanten: float = 0.0
    ukeire: float = 0.0
    wait_quality: float = 0.0
    two_step: float = 0.0
    expected_fan: float = 0.0
    continuation: float = 0.0
    immediate: float = 0.0
    risk: float = 0.0
    shape_loss: float = 0.0
    uncertainty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.shanten
            + self.ukeire
            + self.wait_quality
            + self.two_step
            + self.expected_fan
            + self.continuation
            + self.immediate
            - self.risk
            - self.shape_loss
            - self.uncertainty
        )


@dataclass(frozen=True)
class ActionEvaluation:
    action: StrategyAction
    score: float
    breakdown: ScoreBreakdown
    shanten_before: int | None = None
    shanten_after: int | None = None
    effective_draws: dict[str, int] = field(default_factory=dict)
    two_step_improvements: dict[str, float] = field(default_factory=dict)
    wait_type: str | None = None
    expected_fan: float = 0.0
    win_probability: float | None = None
    expected_net_value: float | None = None
    risk_by_player: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rank: int = 0
    gap_to_best: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.action_type.value,
            "tile": self.action.tile,
            "tiles": list(self.action.tiles),
            "score": self.score,
            "score_breakdown": self.breakdown.__dict__,
            "shanten_before": self.shanten_before,
            "shanten_after": self.shanten_after,
            "effective_draws": self.effective_draws,
            "two_step_improvements": self.two_step_improvements,
            "wait_type": self.wait_type,
            "expected_fan": self.expected_fan,
            "win_probability": self.win_probability,
            "expected_net_value": self.expected_net_value,
            "risk_by_player": self.risk_by_player,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "rank": self.rank,
            "gap_to_best": self.gap_to_best,
        }
