from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from xueliu_ai.table.game_phase import GamePhase
from xueliu_ai.table.structured_types import StructuredTableState, ZoneTile


class RecommendationMode(str, Enum):
    BLOCKED = "blocked"
    HAND_ONLY = "hand_only"
    TABLE_AWARE = "table_aware"
    ENHANCED = "enhanced"


class UnknownImpact(str, Enum):
    CRITICAL = "critical"
    CONTEXTUAL = "contextual"
    IGNORABLE = "ignorable"


@dataclass(frozen=True)
class RecommendationThresholds:
    minimum_core_score: float = 90.0
    table_aware_context_score: float = 50.0
    enhanced_context_score: float = 80.0
    max_safe_inferred_hand_tiles: int = 1
    minimum_safe_inferred_confidence: float = 0.40

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "RecommendationThresholds":
        values = values or {}
        return cls(
            minimum_core_score=float(values.get("minimum_core_score", 90.0)),
            table_aware_context_score=float(values.get("table_aware_context_score", 50.0)),
            enhanced_context_score=float(values.get("enhanced_context_score", 80.0)),
            max_safe_inferred_hand_tiles=int(values.get("max_safe_inferred_hand_tiles", 1)),
            minimum_safe_inferred_confidence=float(
                values.get("minimum_safe_inferred_confidence", 0.40)
            ),
        )


@dataclass(frozen=True)
class UnknownAssessment:
    critical: tuple[ZoneTile, ...] = ()
    contextual: tuple[ZoneTile, ...] = ()
    ignorable: tuple[ZoneTile, ...] = ()
    unlocated_contextual_count: int = 0

    @property
    def critical_count(self) -> int:
        return len(self.critical)

    @property
    def contextual_count(self) -> int:
        return len(self.contextual) + self.unlocated_contextual_count

    @property
    def ignorable_count(self) -> int:
        return len(self.ignorable)


@dataclass(frozen=True)
class RecommendationReadiness:
    mode: RecommendationMode
    allow_recommend: bool
    core_score: float
    context_score: float
    robust: bool
    hard_block_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unknown_assessment: UnknownAssessment = UnknownAssessment()

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "allow_recommend": self.allow_recommend,
            "core_score": round(self.core_score, 1),
            "context_score": round(self.context_score, 1),
            "robust": self.robust,
            "hard_block_reasons": list(self.hard_block_reasons),
            "warnings": list(self.warnings),
            "critical_unknown_count": self.unknown_assessment.critical_count,
            "contextual_unknown_count": self.unknown_assessment.contextual_count,
            "ignorable_unknown_count": self.unknown_assessment.ignorable_count,
        }


def evaluate_recommendation_readiness(
    state: StructuredTableState,
    *,
    phase: GamePhase,
    missing_suit: str | None,
    hand_stable: bool,
    table_context_enabled: bool = True,
    robust: bool = False,
    thresholds: RecommendationThresholds | None = None,
) -> RecommendationReadiness:
    thresholds = thresholds or RecommendationThresholds()
    zones = state.zones
    unknowns = assess_unknown_impacts(state)
    hard_reasons: list[str] = []
    warnings: list[str] = []

    phase_ready = phase == GamePhase.MY_TURN
    if not phase_ready:
        hard_reasons.append("not_my_turn")
    if missing_suit is None:
        hard_reasons.append("missing_suit_not_selected")

    expected_hand_count = 14 - state.confirmed_open_melds * 3
    hand_count_valid = len(zones.hand) == expected_hand_count
    if not hand_count_valid:
        hard_reasons.append(
            f"hand_count_invalid:expected_{expected_hand_count}_got_{len(zones.hand)}"
        )
    if not hand_stable:
        hard_reasons.append("hand_not_stable")

    bottom_suspected = [
        group for group in state.meld_groups if group.zone == "bottom_melds" and group.is_suspected
    ]
    if bottom_suspected:
        hard_reasons.append("own_meld_unconfirmed")
    bottom_conflicts = [
        group
        for group in state.meld_groups
        if group.zone == "bottom_melds" and group.conflicting_tiles
    ]
    if bottom_conflicts:
        hard_reasons.append("own_meld_label_conflict")

    inferred_hand = [
        tile for tile in zones.zone_tiles if tile.zone == "hand" and tile.inferred
    ]
    if len(inferred_hand) > thresholds.max_safe_inferred_hand_tiles:
        hard_reasons.append("too_many_inferred_hand_tiles")
    elif inferred_hand and any(
        tile.confidence < thresholds.minimum_safe_inferred_confidence
        for tile in inferred_hand
    ):
        hard_reasons.append("inferred_hand_confidence_too_low")
    elif inferred_hand:
        warnings.append(f"safe_inferred_hand_tiles:{len(inferred_hand)}")

    if unknowns.critical_count:
        hard_reasons.append(f"critical_unknown_tiles:{unknowns.critical_count}")
    if unknowns.contextual_count:
        warnings.append(f"contextual_unknown_tiles:{unknowns.contextual_count}")
    if unknowns.ignorable_count:
        warnings.append(f"ignored_event_tiles:{unknowns.ignorable_count}")

    consistency_errors = state.consistency_errors()
    if consistency_errors:
        hard_reasons.append(f"state_inconsistent:{consistency_errors[0]}")

    core_counts = Counter(zones.hand)
    for group in state.meld_groups:
        if group.zone == "bottom_melds":
            core_counts.update(tile.label for tile in group.observed_only_tiles)
    if any(count > 4 for count in core_counts.values()):
        hard_reasons.append("core_observed_tile_count_over_four")

    core_score = 0.0
    core_score += 35.0 if hand_count_valid else 0.0
    core_score += 25.0 if hand_stable else 0.0
    core_score += 20.0 if not bottom_suspected and not bottom_conflicts else 0.0
    core_score += 5.0 if phase_ready else 0.0
    core_score += 5.0 if missing_suit is not None else 0.0
    core_score += 10.0 if not consistency_errors and not unknowns.critical_count else 0.0
    if inferred_hand:
        core_score = max(0.0, core_score - 5.0)
    if core_score < thresholds.minimum_core_score:
        hard_reasons.append(
            f"core_score_below_threshold:{core_score:.1f}<{thresholds.minimum_core_score:.1f}"
        )

    context_score = _context_score(state, unknowns)
    hard_reasons = _dedupe(hard_reasons)
    warnings = _dedupe(warnings)
    if hard_reasons:
        return RecommendationReadiness(
            mode=RecommendationMode.BLOCKED,
            allow_recommend=False,
            core_score=core_score,
            context_score=context_score,
            robust=False,
            hard_block_reasons=tuple(hard_reasons),
            warnings=tuple(warnings),
            unknown_assessment=unknowns,
        )

    trusted_context_available = bool(state.observed_visible_counts)
    if not table_context_enabled or not trusted_context_available:
        mode = RecommendationMode.HAND_ONLY
    elif context_score >= thresholds.enhanced_context_score and robust:
        mode = RecommendationMode.ENHANCED
    elif context_score >= thresholds.table_aware_context_score:
        mode = RecommendationMode.TABLE_AWARE
    else:
        mode = RecommendationMode.HAND_ONLY
    if mode == RecommendationMode.HAND_ONLY and unknowns.contextual_count:
        warnings.append("peripheral_information_ignored")
    if mode == RecommendationMode.TABLE_AWARE and not robust:
        warnings.append("recommendation_sensitive_to_table_context")
    return RecommendationReadiness(
        mode=mode,
        allow_recommend=True,
        core_score=core_score,
        context_score=context_score,
        robust=robust,
        warnings=tuple(_dedupe(warnings)),
        unknown_assessment=unknowns,
    )


def with_robustness(
    readiness: RecommendationReadiness,
    *,
    robust: bool,
    table_context_enabled: bool,
    trusted_context_available: bool,
    thresholds: RecommendationThresholds | None = None,
) -> RecommendationReadiness:
    if not readiness.allow_recommend:
        return readiness
    thresholds = thresholds or RecommendationThresholds()
    if not table_context_enabled or not trusted_context_available:
        mode = RecommendationMode.HAND_ONLY
    elif readiness.context_score >= thresholds.enhanced_context_score and robust:
        mode = RecommendationMode.ENHANCED
    elif readiness.context_score >= thresholds.table_aware_context_score:
        mode = RecommendationMode.TABLE_AWARE
    else:
        mode = RecommendationMode.HAND_ONLY
    warnings = [
        warning
        for warning in readiness.warnings
        if warning != "recommendation_sensitive_to_table_context"
    ]
    if mode == RecommendationMode.TABLE_AWARE and not robust:
        warnings.append("recommendation_sensitive_to_table_context")
    return replace(
        readiness,
        mode=mode,
        robust=robust,
        warnings=tuple(_dedupe(warnings)),
    )


def assess_unknown_impacts(state: StructuredTableState) -> UnknownAssessment:
    zones = state.zones
    core_tiles = [
        tile
        for tile in zones.zone_tiles
        if tile.zone == "hand"
        or (tile.zone == "bottom_melds" and not tile.inferred)
    ]
    critical: list[ZoneTile] = []
    contextual: list[ZoneTile] = []
    ignorable: list[ZoneTile] = []
    for tile in zones.zone_tiles:
        if tile.zone not in {"unknown_tiles", "candidate_meld_tiles", "event_tiles"}:
            continue
        if _is_core_related(tile, core_tiles):
            critical.append(tile)
        elif tile.zone == "event_tiles":
            ignorable.append(tile)
        else:
            contextual.append(tile)
    represented_unknown = sum(1 for tile in zones.zone_tiles if tile.zone == "unknown_tiles")
    represented_candidates = sum(
        1 for tile in zones.zone_tiles if tile.zone == "candidate_meld_tiles"
    )
    unlocated = max(0, len(zones.unknown_tiles) - represented_unknown) + max(
        0, len(zones.candidate_meld_tiles) - represented_candidates
    )
    return UnknownAssessment(
        critical=tuple(critical),
        contextual=tuple(contextual),
        ignorable=tuple(ignorable),
        unlocated_contextual_count=unlocated,
    )


def recommendations_are_robust(hand_only_advice, table_aware_advice) -> bool:
    if hand_only_advice is None or table_aware_advice is None:
        return False
    if hand_only_advice.recommended != table_aware_advice.recommended:
        return False
    hand_best = hand_only_advice.candidates[0] if hand_only_advice.candidates else None
    table_best = table_aware_advice.candidates[0] if table_aware_advice.candidates else None
    return bool(
        hand_best
        and table_best
        and hand_best.shanten == table_best.shanten
    )


def _is_core_related(tile: ZoneTile, core_tiles: list[ZoneTile]) -> bool:
    text = f"{tile.group_id or ''} {tile.reason} {tile.source}".lower()
    if "bottom_meld" in text or "hand" in text or "drawn" in text:
        return True
    for core in core_tiles:
        width = max(tile.width, core.width)
        height = max(tile.height, core.height)
        if abs(tile.center_x - core.center_x) <= width * 1.6 and abs(
            tile.center_y - core.center_y
        ) <= height * 1.0:
            return True
    return False


def _context_score(state: StructuredTableState, unknowns: UnknownAssessment) -> float:
    opponent_suspected = sum(
        1
        for group in state.meld_groups
        if group.zone != "bottom_melds" and group.is_suspected
    )
    zones = state.zones
    penalty = min(48.0, unknowns.contextual_count * 12.0)
    penalty += min(30.0, opponent_suspected * 15.0)
    penalty += min(12.0, len(zones.hu_display_tiles) * 4.0)
    penalty += min(10.0, unknowns.ignorable_count * 2.0)
    return max(0.0, 100.0 - penalty)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
