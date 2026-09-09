"""Tests for the rollout / evaluation harness.

Covers:
- run_episode produces a well-formed EpisodeResult
- evaluate_agent aggregates correctly over multiple seeds
- Determinism: same agent + same seed twice -> identical result
"""

from __future__ import annotations

import random

from jackdaw.env.agents import RandomAgent
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.rollout import EpisodeResult, evaluate_agent, run_episode

SEEDS = [f"ROLLOUT_{i}" for i in range(3)]


class TestRunEpisode:
    def test_returns_well_formed_result(self):
        result = run_episode(RandomAgent(), DirectAdapter, seed="ROLLOUT_SHAPE")
        assert isinstance(result, EpisodeResult)
        assert result.seed == "ROLLOUT_SHAPE"
        assert result.ante_reached >= 1
        assert result.length > 0
        assert isinstance(result.won, bool)

    def test_deterministic_for_fixed_agent_seed(self):
        """Same seeded, deterministic agent behavior + same env seed should
        reproduce the same episode length and ante."""

        random.seed(42)
        r1 = run_episode(RandomAgent(), DirectAdapter, seed="ROLLOUT_DETERMINISM")
        random.seed(42)
        r2 = run_episode(RandomAgent(), DirectAdapter, seed="ROLLOUT_DETERMINISM")
        assert r1.length == r2.length
        assert r1.ante_reached == r2.ante_reached
        assert r1.won == r2.won


class TestEvaluateAgent:
    def test_shape_and_aggregate_arithmetic(self):
        summary = evaluate_agent(RandomAgent(), SEEDS, DirectAdapter)
        assert summary["episodes"] == len(SEEDS)
        assert len(summary["results"]) == len(SEEDS)
        antes = [r.ante_reached for r in summary["results"]]
        assert summary["mean_ante"] == sum(antes) / len(antes)
        assert summary["max_ante"] == max(antes)
        assert summary["min_ante"] == min(antes)
        wins = [r.won for r in summary["results"]]
        assert summary["win_rate"] == sum(wins) / len(wins)
