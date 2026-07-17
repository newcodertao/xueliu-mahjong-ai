from __future__ import annotations

from dataclasses import replace

from xueliu_ai.realtime_table import TableZones
from xueliu_ai.table.game_phase import GamePhase
from xueliu_ai.table.recommendation_readiness import (
    RecommendationMode,
    evaluate_recommendation_readiness,
    with_robustness,
)
from xueliu_ai.table.structured_types import MeldGroup, MeldKind, StructuredTableState, ZoneTile


def _tile(label: str, x: float, y: float, zone: str = "hand") -> ZoneTile:
    return ZoneTile(label, 0.95, x, y, x + 36, y + 52, zone)


def _state(
    hand_count: int = 14,
    *,
    unknowns: list[ZoneTile] | None = None,
    meld: MeldGroup | None = None,
) -> StructuredTableState:
    hand_tiles = [_tile(f"{index % 9 + 1}W", 100 + index * 42, 700) for index in range(hand_count)]
    zone_tiles = list(hand_tiles)
    groups = [] if meld is None else [meld]
    meld_lists = {"bottom_melds": [], "left_melds": [], "right_melds": [], "top_melds": []}
    observed = {}
    logical = {}
    if meld is not None:
        zone_tiles.extend(meld.logical_tiles)
        meld_lists[meld.zone] = [tile.label for tile in meld.logical_tiles]
        for tile in meld.observed_only_tiles:
            observed[tile.label] = observed.get(tile.label, 0) + 1
        for tile in meld.logical_tiles:
            logical[tile.label] = logical.get(tile.label, 0) + 1
    unknowns = unknowns or []
    zone_tiles.extend(unknowns)
    zones = TableZones(
        hand=[tile.label for tile in hand_tiles],
        bottom_melds=meld_lists["bottom_melds"],
        left_melds=meld_lists["left_melds"],
        right_melds=meld_lists["right_melds"],
        top_melds=meld_lists["top_melds"],
        center_discards=[],
        unknown_tiles=[tile.label for tile in unknowns if tile.zone == "unknown_tiles"],
        event_tiles=[tile.label for tile in unknowns if tile.zone == "event_tiles"],
        all_tiles=[tile.label for tile in zone_tiles],
        zone_tiles=zone_tiles,
        meld_groups=groups,
    )
    own_meld = bool(meld and meld.zone == "bottom_melds")
    return StructuredTableState(
        zones,
        groups,
        int(bool(own_meld and meld and meld.is_confirmed)),
        int(bool(own_meld and meld and meld.is_suspected)),
        observed,
        logical,
    )


def _pong(zone: str = "bottom_melds", suspected: bool = False) -> MeldGroup:
    tiles = [
        replace(_tile("2T", 900 + index * 40, 610, zone), group_id="meld-1")
        for index in range(3)
    ]
    kind = MeldKind.SUSPECTED_PONG if suspected else MeldKind.PONG
    return MeldGroup("meld-1", zone, kind, "2T", observed_tiles=tiles, confidence=0.9)


def _evaluate(state: StructuredTableState, **kwargs):
    return evaluate_recommendation_readiness(
        state,
        phase=kwargs.pop("phase", GamePhase.MY_TURN),
        missing_suit=kwargs.pop("missing_suit", "B"),
        hand_stable=kwargs.pop("hand_stable", True),
        **kwargs,
    )


def test_complete_hand_allows_hand_only_recommendation_with_peripheral_unknowns() -> None:
    unknowns = [_tile("7T", 120, 150, "unknown_tiles"), _tile("8T", 220, 150, "unknown_tiles")]
    readiness = _evaluate(_state(14, unknowns=unknowns), table_context_enabled=False)
    assert readiness.allow_recommend
    assert readiness.mode == RecommendationMode.HAND_ONLY
    assert readiness.unknown_assessment.contextual_count == 2


def test_realistic_eleven_tile_hand_with_confirmed_pong_can_recommend() -> None:
    unknowns = [_tile("7T", 120, 150, "unknown_tiles"), _tile("8T", 220, 150, "unknown_tiles")]
    readiness = _evaluate(_state(11, unknowns=unknowns, meld=_pong()), table_context_enabled=False)
    assert readiness.allow_recommend
    assert readiness.core_score >= 90


def test_unknown_near_hand_is_critical_and_blocks() -> None:
    readiness = _evaluate(_state(14, unknowns=[_tile("7T", 160, 705, "unknown_tiles")]))
    assert not readiness.allow_recommend
    assert "critical_unknown_tiles:1" in readiness.hard_block_reasons


def test_own_suspected_meld_blocks_but_opponent_suspected_meld_does_not() -> None:
    own = _evaluate(_state(11, meld=_pong(suspected=True)))
    assert not own.allow_recommend
    opponent = _evaluate(_state(14, meld=_pong("left_melds", suspected=True)), table_context_enabled=False)
    assert opponent.allow_recommend


def test_missing_suit_turn_and_hand_stability_are_hard_gates() -> None:
    state = _state()
    assert not _evaluate(state, missing_suit=None).allow_recommend
    assert not _evaluate(state, phase=GamePhase.WAITING).allow_recommend
    assert not _evaluate(state, hand_stable=False).allow_recommend


def test_robust_high_context_promotes_to_enhanced() -> None:
    state = _state(11, meld=_pong())
    readiness = _evaluate(state, table_context_enabled=True)
    readiness = with_robustness(
        readiness,
        robust=True,
        table_context_enabled=True,
        trusted_context_available=True,
    )
    assert readiness.mode == RecommendationMode.ENHANCED


def test_event_tile_away_from_core_is_ignored() -> None:
    readiness = _evaluate(_state(14, unknowns=[_tile("3T", 600, 200, "event_tiles")]))
    assert readiness.allow_recommend
    assert readiness.unknown_assessment.ignorable_count == 1
