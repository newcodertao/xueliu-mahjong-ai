from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    selected: int
    total_eligible: int
    errors: int
    coverage: float
    risk: float


def risk_coverage_curve(
    rows: Iterable[Mapping[str, object]],
    thresholds: Iterable[float] = range(0, 101),
) -> list[RiskCoveragePoint]:
    samples = [row for row in rows if bool(row.get("eligible", True))]
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        selected = [row for row in samples if float(row.get("core_score", 0.0)) >= threshold]
        errors = sum(not bool(row.get("core_state_correct", False)) for row in selected)
        selected_count = len(selected)
        points.append(
            RiskCoveragePoint(
                threshold=float(threshold),
                selected=selected_count,
                total_eligible=len(samples),
                errors=errors,
                coverage=selected_count / len(samples) if samples else 0.0,
                risk=errors / selected_count if selected_count else 0.0,
            )
        )
    return points


def select_threshold(
    rows: Iterable[Mapping[str, object]],
    *,
    maximum_risk: float = 0.01,
    thresholds: Iterable[float] = range(0, 101),
) -> RiskCoveragePoint | None:
    acceptable = [
        point
        for point in risk_coverage_curve(rows, thresholds)
        if point.selected and point.risk <= maximum_risk
    ]
    if not acceptable:
        return None
    return max(acceptable, key=lambda point: (point.coverage, -point.threshold))
