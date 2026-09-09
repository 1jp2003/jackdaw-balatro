"""Reusable episode-rollout / evaluation harness.

Generic over any :class:`~jackdaw.env.agents.Agent` — not heuristic-specific
— so it can also evaluate RL-trained policies against the same frozen seed
set (:mod:`jackdaw.env.eval_seeds`) later. Fills the gap noted in
``CLAUDE.md``'s Conventions ("compare results on the frozen eval seed set,
not rollout statistics") and ``docs/RL_PLAN.md`` Phase 0 ("frozen eval
harness"): previously no code combined the :class:`Agent` protocol with
:class:`~jackdaw.env.balatro_env.BalatroEnvironment` to collect ante/win-rate
stats over many seeds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jackdaw.env.agents import Agent
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.game_interface import GameAdapter


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one rollout episode."""

    seed: str
    ante_reached: int
    won: bool
    length: int


def run_episode(
    agent: Agent,
    adapter_factory: Callable[[], GameAdapter],
    *,
    seed: str,
    back_key: str = "b_red",
    stake: int = 1,
    max_steps: int = 10_000,
) -> EpisodeResult:
    """Run one full episode of *agent* against a fresh :class:`BalatroEnvironment`.

    A fresh environment is constructed per call (``back_key``/``stake`` are
    constructor-only on :class:`BalatroEnvironment` — ``reset(**kwargs)``
    only reads ``seed``), which also keeps episodes fully independent.
    """
    env = BalatroEnvironment(
        adapter_factory=adapter_factory,
        back_keys=[back_key],
        stakes=[stake],
        max_steps=max_steps,
    )
    agent.reset()
    obs, mask, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        fa = agent.act(obs, mask, info)
        obs, terminated, truncated, mask, info = env.step(fa)

    return EpisodeResult(
        seed=seed,
        ante_reached=env.episode_ante,
        won=env.episode_won,
        length=env.episode_length,
    )


def evaluate_agent(
    agent: Agent,
    seeds: list[str],
    adapter_factory: Callable[[], GameAdapter],
    **episode_kwargs: object,
) -> dict:
    """Run *agent* over every seed in *seeds*, return summary stats.

    Keys: ``episodes``, ``mean_ante``, ``max_ante``, ``min_ante``,
    ``win_rate``, ``mean_length``, and ``results`` (the raw
    :class:`EpisodeResult` list).
    """
    results = [run_episode(agent, adapter_factory, seed=seed, **episode_kwargs) for seed in seeds]
    antes = [r.ante_reached for r in results]
    lengths = [r.length for r in results]
    wins = [r.won for r in results]
    n = len(results)
    return {
        "episodes": n,
        "mean_ante": sum(antes) / n,
        "max_ante": max(antes),
        "min_ante": min(antes),
        "win_rate": sum(wins) / n,
        "mean_length": sum(lengths) / n,
        "results": results,
    }


__all__ = ["EpisodeResult", "run_episode", "evaluate_agent"]
