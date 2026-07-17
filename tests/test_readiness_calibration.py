from xueliu_ai.evaluation.readiness_calibration import risk_coverage_curve, select_threshold


def test_risk_coverage_curve_reports_coverage_and_error_risk() -> None:
    rows = [
        {"core_score": 95, "core_state_correct": True},
        {"core_score": 92, "core_state_correct": True},
        {"core_score": 80, "core_state_correct": False},
    ]
    point = risk_coverage_curve(rows, thresholds=[90])[0]
    assert point.selected == 2
    assert point.coverage == 2 / 3
    assert point.risk == 0


def test_select_threshold_maximizes_coverage_under_risk_limit() -> None:
    rows = [
        {"core_score": 95, "core_state_correct": True},
        {"core_score": 92, "core_state_correct": True},
        {"core_score": 80, "core_state_correct": False},
    ]
    selected = select_threshold(rows, maximum_risk=0, thresholds=[80, 90, 95])
    assert selected is not None
    assert selected.threshold == 90
    assert selected.coverage == 2 / 3
