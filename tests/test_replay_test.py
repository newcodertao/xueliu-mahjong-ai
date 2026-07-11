import json

from xueliu_ai.evaluation.replay_test import evaluate_expected, load_gold_cases


def test_evaluate_expected_counts_passes_and_failures() -> None:
    checked: dict[str, int] = {}
    passed: dict[str, int] = {}
    failures = evaluate_expected(
        {"phase": "my_turn", "my_hand_count": 14},
        {"phase": "waiting", "my_hand_count": 14},
        checked,
        passed,
    )

    assert checked == {"phase": 1, "my_hand_count": 1}
    assert passed == {"my_hand_count": 1}
    assert failures == [{"field": "phase", "expected": "my_turn", "actual": "waiting"}]


def test_load_gold_cases_accepts_wrapped_cases(tmp_path) -> None:
    path = tmp_path / "gold_cases.json"
    path.write_text(
        json.dumps({"cases": [{"id": "a", "image": "a.jpg", "expected": {"phase": "my_turn"}}]}),
        encoding="utf-8",
    )

    cases = load_gold_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "a"
    assert cases[0].expected["phase"] == "my_turn"

