from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.mahjong.rules_xueliu import legal_discards
from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import tile_rank, tile_suit, validate_tiles
from xueliu_ai.strategy.actions import ActionType, StrategyAction
from xueliu_ai.strategy.context import StrategyContext
from xueliu_ai.strategy.evaluation import ActionEvaluation, ScoreBreakdown
from xueliu_ai.strategy.risk_model import discard_risk_by_player
from xueliu_ai.strategy.tile_efficiency import evaluate_tile_efficiency, wait_quality_score
from xueliu_ai.strategy.weights import StrategyWeights, load_strategy_weights


@dataclass(frozen=True)
class DiscardCandidate:
    tile: str
    score: float
    shanten: int
    ukeire: int
    reason: str
    effective_draws: dict[str, int] | None = None
    wait_type: str | None = None
    two_step_value: float = 0.0
    risk: float = 0.0
    score_breakdown: dict[str, float] | None = None


@dataclass(frozen=True)
class DiscardAdvice:
    recommended: str
    candidates: list[DiscardCandidate]
    explanation: str
    evaluations: list[ActionEvaluation] | None = None


def advise_discard(
    tiles: list[str],
    missing_suit: str | None = None,
    visible_counts: dict[str, int] | None = None,
    open_melds: int = 0,
    context: StrategyContext | None = None,
    weights: StrategyWeights | None = None,
) -> DiscardAdvice:
    validate_tiles(tiles)
    expected = 14 - open_melds * 3
    if len(tiles) != expected:
        raise ValueError(f"出牌建议需要摸牌后的暗手牌 {expected} 张，当前 {len(tiles)} 张")

    context = context or StrategyContext(hand=tuple(tiles), missing_suit=missing_suit)
    weights = weights or load_strategy_weights()
    candidates: list[DiscardCandidate] = []
    evaluations: list[ActionEvaluation] = []
    restricted = legal_discards(tiles, missing_suit)
    before_shanten = best_shanten(tiles, open_melds=open_melds)
    for tile in sorted(set(restricted)):
        after = tiles.copy()
        after.remove(tile)
        efficiency = evaluate_tile_efficiency(
            after,
            visible_counts,
            open_melds=open_melds,
            missing_suit=missing_suit,
        )
        shanten = efficiency.shanten
        ukeire = efficiency.ukeire
        seen = (visible_counts or {}).get(tile, 0)
        remaining = max(0, 4 - seen - tiles.count(tile))
        risk_by_player = discard_risk_by_player(tile, context)
        combined_risk = max(risk_by_player.values(), default=0.0)
        shape = _shape_score(after, tile, missing_suit) + seen * 1.5
        forced_bonus = 1000.0 if missing_suit and tile_suit(tile) == missing_suit.upper() else 0.0
        breakdown = ScoreBreakdown(
            shanten=-shanten * weights.shanten + forced_bonus,
            ukeire=ukeire * weights.ukeire,
            wait_quality=wait_quality_score(efficiency.wait_type) * weights.wait_quality,
            two_step=efficiency.two_step_value * weights.two_step,
            continuation=_continuation_value(after) * weights.blood_flow_continuation,
            risk=combined_risk * weights.deal_in_risk,
            shape_loss=max(0.0, -shape) * weights.shape_loss,
            uncertainty=(1.0 - context.recognition_quality) * weights.uncertainty,
        )
        score = breakdown.total + max(0.0, shape)
        reason = (
            f"打出后向听 {shanten}，有效进张 {ukeire}，"
            f"两步改良 {efficiency.two_step_value:.1f}，风险 {combined_risk:.0%}，"
            f"外面已见 {seen} 张，本手外剩余约 {remaining} 张"
        )
        if missing_suit and tile_suit(tile) == missing_suit.upper():
            reason = f"优先处理定缺花色；{reason}"
        candidates.append(
            DiscardCandidate(
                tile,
                score,
                shanten,
                ukeire,
                reason,
                effective_draws=efficiency.effective_draws,
                wait_type=efficiency.wait_type,
                two_step_value=efficiency.two_step_value,
                risk=combined_risk,
                score_breakdown=breakdown.__dict__,
            )
        )
        evaluations.append(
            ActionEvaluation(
                action=StrategyAction(ActionType.DISCARD, tile=tile),
                score=score,
                breakdown=breakdown,
                shanten_before=before_shanten,
                shanten_after=shanten,
                effective_draws=efficiency.effective_draws,
                two_step_improvements=efficiency.two_step_improvements,
                wait_type=efficiency.wait_type,
                risk_by_player=risk_by_player,
                reasons=(reason,),
            )
        )

    candidates.sort(key=lambda item: (item.score, item.ukeire, -_terminal_penalty(item.tile)), reverse=True)
    best = candidates[0]
    best_score = best.score
    ranked_evaluations = []
    evaluation_by_tile = {evaluation.action.tile: evaluation for evaluation in evaluations}
    for rank, candidate in enumerate(candidates, start=1):
        evaluation = evaluation_by_tile[candidate.tile]
        ranked_evaluations.append(
            ActionEvaluation(
                **{
                    **evaluation.__dict__,
                    "rank": rank,
                    "gap_to_best": best_score - candidate.score,
                }
            )
        )
    return DiscardAdvice(
        recommended=best.tile,
        candidates=candidates,
        explanation=f"推荐打 {best.tile}：{best.reason}。",
        evaluations=ranked_evaluations,
    )


def _shape_score(after: list[str], discarded: str, missing_suit: str | None) -> float:
    rank = tile_rank(discarded)
    score = 0.0
    if rank in (1, 9):
        score += 3
    if missing_suit and tile_suit(discarded) == missing_suit.upper():
        score += 5
    same_suit_ranks = [tile_rank(tile) for tile in after if tile_suit(tile) == tile_suit(discarded)]
    if rank - 1 in same_suit_ranks and rank + 1 in same_suit_ranks:
        score -= 8
    if rank - 1 in same_suit_ranks or rank + 1 in same_suit_ranks:
        score -= 3
    return score


def _terminal_penalty(tile: str) -> int:
    return 1 if tile_rank(tile) in (1, 9) else 0


def _continuation_value(tiles: list[str]) -> float:
    pairs = sum(1 for tile in set(tiles) if tiles.count(tile) >= 2)
    triplets = sum(1 for tile in set(tiles) if tiles.count(tile) >= 3)
    return min(1.0, pairs * 0.08 + triplets * 0.18)
