from __future__ import annotations

from dataclasses import dataclass, fields

from xueliu_ai.config import load_yaml


@dataclass(frozen=True)
class StrategyWeights:
    shanten: float = 100.0
    ukeire: float = 2.0
    wait_quality: float = 8.0
    two_step: float = 0.35
    expected_fan: float = 12.0
    blood_flow_continuation: float = 5.0
    immediate_score: float = 10.0
    deal_in_risk: float = 45.0
    shape_loss: float = 1.0
    uncertainty: float = 20.0

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "StrategyWeights":
        values = values or {}
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: float(value) for key, value in values.items() if key in allowed})


def load_strategy_weights(path: str = "configs/rule_xueliu.yaml") -> StrategyWeights:
    config = load_yaml(path)
    return StrategyWeights.from_mapping(config.get("strategy", {}).get("weights"))
