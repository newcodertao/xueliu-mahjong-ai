from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.engine import StrategyEngine


@dataclass(frozen=True)
class StrategyGoldCase:
    case_id: str
    context: StrategyContext
    expected: str
    acceptable: tuple[str, ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class StrategyGoldSummary:
    total: int
    passed: int
    failures: tuple[str, ...]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_strategy_gold(path: str | Path) -> list[StrategyGoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        context_data = row["context"]
        context = StrategyContext(
            hand=tuple(context_data["hand"]),
            missing_suit=context_data.get("missing_suit"),
            melds={key: tuple(tuple(group) for group in groups) for key, groups in context_data.get("melds", {}).items()},
            discards={key: tuple(value) for key, value in context_data.get("discards", {}).items()},
            phase=context_data.get("phase", "my_turn"),
        )
        cases.append(
            StrategyGoldCase(
                str(row["id"]),
                context,
                str(row["expected"]),
                tuple(row.get("acceptable", ())),
                str(row.get("explanation", "")),
            )
        )
    return cases


def run_strategy_gold(cases: list[StrategyGoldCase]) -> StrategyGoldSummary:
    engine = StrategyEngine()
    failures = []
    for case in cases:
        actual = engine.evaluate_discard(case.context).advice.recommended
        allowed = {case.expected, *case.acceptable}
        if actual not in allowed:
            failures.append(f"{case.case_id}: expected {sorted(allowed)}, got {actual}")
    return StrategyGoldSummary(len(cases), len(cases) - len(failures), tuple(failures))
