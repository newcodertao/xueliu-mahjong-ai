from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StrategyReplaySummary:
    decisions: int
    recommendations: int
    blocked: int
    matched_actual: int
    average_best_gap: float
    average_compute_ms: float

    @property
    def action_match_rate(self) -> float:
        return self.matched_actual / self.recommendations if self.recommendations else 0.0


def summarize_strategy_log(path: str | Path) -> StrategyReplaySummary:
    decisions = recommendations = blocked = matched = 0
    gaps: list[float] = []
    timings: list[float] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") not in {"strategy_decision", "realtime_ui_tick"}:
                continue
            decisions += 1
            advice = row.get("advice") or {}
            if not advice:
                blocked += 1
                continue
            recommendations += 1
            actual = row.get("actual_action")
            if actual and actual == advice.get("recommended"):
                matched += 1
            candidates = advice.get("candidates") or []
            if len(candidates) >= 2:
                gaps.append(float(candidates[0]["score"]) - float(candidates[1]["score"]))
            if row.get("strategy_compute_ms") is not None:
                timings.append(float(row["strategy_compute_ms"]))
    return StrategyReplaySummary(
        decisions,
        recommendations,
        blocked,
        matched,
        sum(gaps) / len(gaps) if gaps else 0.0,
        sum(timings) / len(timings) if timings else 0.0,
    )


def generate_strategy_report(summary: StrategyReplaySummary) -> str:
    return "\n".join(
        [
            "# 策略回放报告",
            "",
            f"- 决策帧：{summary.decisions}",
            f"- 已推荐：{summary.recommendations}",
            f"- 暂停推荐：{summary.blocked}",
            f"- 与实际动作一致率：{summary.action_match_rate:.1%}",
            f"- 第一、第二候选平均分差：{summary.average_best_gap:.2f}",
            f"- 平均策略耗时：{summary.average_compute_ms:.1f} ms",
        ]
    )
