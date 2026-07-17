from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.evaluation import ActionEvaluation


@dataclass(frozen=True)
class StrategyTrainingRow:
    state_hash: str
    rules_version: str
    feature_version: str
    source: str
    hand: tuple[str, ...]
    missing_suit: str | None
    legal_actions: tuple[str, ...]
    recommended_action: str
    actual_action: str | None
    final_net_value: float | None
    recognition_quality: float
    inferred_tile_count: int


def make_training_row(
    context: StrategyContext,
    evaluations: list[ActionEvaluation],
    *,
    source: str,
    actual_action: str | None = None,
    final_net_value: float | None = None,
    feature_version: str = "strategy-v2",
) -> StrategyTrainingRow:
    if not evaluations:
        raise ValueError("At least one legal action evaluation is required")
    payload = {
        "hand": sorted(context.hand),
        "missing_suit": context.missing_suit,
        "melds": context.melds,
        "discards": context.discards,
        "phase": context.phase,
        "rules_version": context.rules_version,
    }
    state_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
    return StrategyTrainingRow(
        state_hash=state_hash,
        rules_version=context.rules_version,
        feature_version=feature_version,
        source=source,
        hand=context.hand,
        missing_suit=context.missing_suit,
        legal_actions=tuple(item.action.key for item in ranked),
        recommended_action=ranked[0].action.key,
        actual_action=actual_action,
        final_net_value=final_net_value,
        recognition_quality=context.recognition_quality,
        inferred_tile_count=context.inferred_tile_count,
    )


def append_training_row(path: str | Path, row: StrategyTrainingRow) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def is_training_quality_eligible(
    context: StrategyContext,
    *,
    minimum_quality: float = 0.95,
    max_inferred_tiles: int = 0,
) -> bool:
    return (
        context.recognition_quality >= minimum_quality
        and context.inferred_tile_count <= max_inferred_tiles
        and context.unknown_tile_count == 0
    )
