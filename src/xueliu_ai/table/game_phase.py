from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from xueliu_ai.realtime_table import TableZones, ZoneDiagnostics


class GamePhase(str, Enum):
    UNKNOWN = "unknown"
    LOADING = "loading"
    DEALING = "dealing"
    PLAYING_PARTIAL = "playing_partial"
    CHOOSE_MISSING_SUIT = "choose_missing_suit"
    WAITING = "waiting"
    MY_TURN = "my_turn"
    SETTLEMENT = "settlement"


PHASE_TEXT = {
    GamePhase.UNKNOWN: "未知",
    GamePhase.LOADING: "加载中",
    GamePhase.DEALING: "发牌/动画中",
    GamePhase.PLAYING_PARTIAL: "牌局进行中（识别不完整）",
    GamePhase.CHOOSE_MISSING_SUIT: "等待定缺",
    GamePhase.WAITING: "等待摸牌",
    GamePhase.MY_TURN: "轮到我出牌",
    GamePhase.SETTLEMENT: "结算/非牌局",
}


@dataclass(frozen=True)
class PhaseContext:
    zones: TableZones
    diagnostics: ZoneDiagnostics
    stable: bool = False
    missing_suit: str | None = None
    detections: int = 0
    message: str = ""


@dataclass(frozen=True)
class RecommendDecision:
    phase: GamePhase
    allow: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def phase_text(self) -> str:
        return PHASE_TEXT[self.phase]

    def reason_text(self) -> str:
        return "；".join(self.reasons) if self.reasons else "可以推荐"


def infer_game_phase(context: PhaseContext) -> GamePhase:
    zones = context.zones
    hand_count = len(zones.hand)
    total_visible = len(zones.all_tiles)

    if context.detections <= 1 and total_visible <= 1:
        return GamePhase.LOADING

    if hand_count == 0 and total_visible >= 20:
        return GamePhase.SETTLEMENT

    if context.missing_suit is None and hand_count in context.diagnostics.expected_hand_counts:
        return GamePhase.CHOOSE_MISSING_SUIT

    if not context.diagnostics.valid:
        board_activity = bool(
            zones.center_discards
            or zones.meld_groups
            or zones.hu_display_tiles
            or zones.candidate_meld_tiles
        )
        expected_minimum = min(context.diagnostics.expected_hand_counts, default=13)
        if not board_activity and total_visible <= 14 and hand_count < expected_minimum:
            return GamePhase.DEALING
        return GamePhase.PLAYING_PARTIAL

    if _is_my_turn_count(hand_count, context.diagnostics.open_melds):
        return GamePhase.MY_TURN

    if _is_waiting_count(hand_count, context.diagnostics.open_melds):
        return GamePhase.WAITING

    return GamePhase.UNKNOWN


def should_allow_recommend(context: PhaseContext) -> RecommendDecision:
    phase = infer_game_phase(context)
    reasons: list[str] = []

    if phase != GamePhase.MY_TURN:
        reasons.append(f"当前阶段是{PHASE_TEXT[phase]}，不是出牌时机")
    if context.missing_suit is None:
        reasons.append("还没有选择定缺花色")
    if not context.stable:
        reasons.append("识别结果还没有连续稳定")
    if not context.diagnostics.valid:
        reasons.extend(context.diagnostics.warnings)

    return RecommendDecision(phase=phase, allow=not reasons, reasons=reasons)


def expected_hand_counts(open_melds: int) -> tuple[int, int]:
    concealed = max(1, 13 - open_melds * 3)
    drawn = max(2, 14 - open_melds * 3)
    return concealed, drawn


def _is_waiting_count(hand_count: int, open_melds: int) -> bool:
    return hand_count == expected_hand_counts(open_melds)[0]


def _is_my_turn_count(hand_count: int, open_melds: int) -> bool:
    return hand_count == expected_hand_counts(open_melds)[1]
