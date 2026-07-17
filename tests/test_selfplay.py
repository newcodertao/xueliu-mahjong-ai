from xueliu_ai.selfplay.agents import FastRuleAgent, RandomAgent
from xueliu_ai.selfplay.environment import XueliuSelfPlayEnv
from xueliu_ai.selfplay.optimizer import optimize_fast_agent
from xueliu_ai.selfplay.tournament import run_tournament


def _agents():
    return [FastRuleAgent(name=f"agent-{index}") for index in range(4)]


def test_selfplay_is_deterministic_and_zero_sum() -> None:
    first = XueliuSelfPlayEnv(_agents(), record_events=True).play(17)
    second = XueliuSelfPlayEnv(_agents(), record_events=True).play(17)
    assert first == second
    assert sum(first.scores) == 0
    assert first.turns > 0


def test_selfplay_preserves_all_108_tiles_in_eventful_game() -> None:
    result = XueliuSelfPlayEnv(_agents(), record_events=True).play(42)
    assert result.wall_remaining == 0
    assert any(event.action == "discard" for event in result.events)
    assert sum(result.scores) == 0


def test_tournament_rotates_seats_and_returns_agent_metrics() -> None:
    agents = [FastRuleAgent(name="candidate"), RandomAgent(name="r1"), RandomAgent(name="r2"), RandomAgent(name="r3")]
    result = run_tournament(agents, games=4, seed=9)
    assert result.games == 4
    assert [stats.games for stats in result.agents] == [4, 4, 4, 4]
    assert {stats.name for stats in result.agents} == {agent.name for agent in agents}


def test_multiple_seeds_remain_zero_sum() -> None:
    for seed in range(8):
        result = XueliuSelfPlayEnv(_agents()).play(seed)
        assert sum(result.scores) == 0
        assert result.turns <= 300


def test_optimizer_returns_reproducible_champion_manifest_data() -> None:
    first = optimize_fast_agent(candidates=2, games_per_candidate=4, seed=23)
    second = optimize_fast_agent(candidates=2, games_per_candidate=4, seed=23)
    assert first == second
    assert first.champion in first.candidates
