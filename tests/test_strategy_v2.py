import json
import time

from xueliu_ai.evaluation.strategy_replay import generate_strategy_report, summarize_strategy_log
from xueliu_ai.mahjong.rules_xueliu import load_rule_profile
from xueliu_ai.mahjong.settlement import estimate_win_settlement
from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.engine import StrategyEngine
from xueliu_ai.strategy.training_data import is_training_quality_eligible, make_training_row


HAND = ("1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B")


def test_rule_profile_loads_strategy_relevant_variants() -> None:
    rules = load_rule_profile()
    assert not rules.allow_chi
    assert rules.continue_after_hu
    assert rules.fan_cap >= 1


def test_strategy_engine_caches_unchanged_state() -> None:
    context = StrategyContext(hand=HAND, missing_suit="W")
    engine = StrategyEngine()
    first = engine.evaluate_discard(context)
    started = time.perf_counter()
    second = engine.evaluate_discard(context)
    assert second is first
    assert time.perf_counter() - started < 0.02
    assert first.advice.evaluations
    assert first.advice.evaluations[0].gap_to_best == 0


def test_forbidden_suit_blocks_settlement() -> None:
    result = estimate_win_settlement(list(HAND), missing_suit="W")
    assert not result.legal
    assert "forbidden_suit_remaining" in result.reasons


def test_strategy_replay_report_summarizes_decisions(tmp_path) -> None:
    path = tmp_path / "strategy.jsonl"
    rows = [
        {"event": "strategy_decision", "advice": {"recommended": "9W", "candidates": [{"score": 10}, {"score": 8}]}, "actual_action": "9W", "strategy_compute_ms": 12},
        {"event": "strategy_decision", "advice": None},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary = summarize_strategy_log(path)
    assert summary.decisions == 2
    assert summary.action_match_rate == 1.0
    assert "策略回放报告" in generate_strategy_report(summary)


def test_training_row_is_versioned_and_filters_uncertain_states() -> None:
    context = StrategyContext(hand=HAND, missing_suit="W")
    decision = StrategyEngine().evaluate_discard(context)
    row = make_training_row(context, decision.advice.evaluations or [], source="gold")
    assert row.state_hash
    assert row.feature_version == "strategy-v2"
    assert row.recommended_action.startswith("discard:")
    assert is_training_quality_eligible(context)
    uncertain = StrategyContext(
        hand=HAND,
        missing_suit="W",
        recognition_quality=0.8,
        inferred_tile_count=1,
    )
    assert not is_training_quality_eligible(uncertain)
