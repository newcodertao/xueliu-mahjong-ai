from xueliu_ai.realtime_table import TableZones, diagnose_zones
from xueliu_ai.table.game_phase import GamePhase, PhaseContext, infer_game_phase, should_allow_recommend
from xueliu_ai.table.my_area import analyze_my_area


def _zones(hand: list[str], melds: list[str] | None = None) -> TableZones:
    return TableZones(
        hand=hand,
        bottom_melds=melds or [],
        left_melds=[],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=[*hand, *(melds or [])],
        my_discards=[],
        left_discards=[],
        top_discards=[],
        right_discards=[],
    )


def test_phase_waiting_and_my_turn_without_melds() -> None:
    waiting = _zones(["1W"] * 4 + ["2W"] * 4 + ["3W"] * 4 + ["4W"])
    my_turn = _zones([*waiting.hand, "5W"])

    assert infer_game_phase(PhaseContext(waiting, diagnose_zones(waiting), True, "W", 13)) == GamePhase.WAITING
    assert infer_game_phase(PhaseContext(my_turn, diagnose_zones(my_turn), True, "W", 14)) == GamePhase.MY_TURN


def test_phase_requires_missing_suit_before_recommend() -> None:
    zones = _zones(["1W"] * 4 + ["2W"] * 4 + ["3W"] * 4 + ["4W", "5W"])
    decision = should_allow_recommend(PhaseContext(zones, diagnose_zones(zones), True, None, 14))

    assert decision.phase == GamePhase.CHOOSE_MISSING_SUIT
    assert not decision.allow


def test_phase_allows_recommend_when_stable_my_turn() -> None:
    zones = _zones(["1W"] * 4 + ["2W"] * 4 + ["3W"] * 4 + ["4W", "5W"])
    decision = should_allow_recommend(PhaseContext(zones, diagnose_zones(zones), True, "W", 14))

    assert decision.phase == GamePhase.MY_TURN
    assert decision.allow


def test_my_area_analysis_reports_meld_count_and_drawn_tile() -> None:
    zones = _zones(["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W"], ["9W", "9W", "9W", "1T", "1T", "1T"])
    analysis = analyze_my_area(zones)

    assert analysis.meld_group_count == 2
    assert analysis.expected_counts == (7, 8)
    assert analysis.drawn_tile == "8W"
    assert analysis.legal_count

