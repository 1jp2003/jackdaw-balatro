"""Tests for HeuristicAgent.

Covers:
- Agent protocol compliance
- Episode completion / valid FactoredAction production
- The never-discard-on-last-hand correctness guard
- A loose regression check that it beats RandomAgent
"""

from __future__ import annotations

import random

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
    ActionType,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import Agent, RandomAgent
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.heuristic_agent import HeuristicAgent

BACK = "b_red"
STAKE = 1


def _run_agent_episode(agent: Agent, seed: str, max_steps: int = 5000) -> dict:
    """Run one episode with the given agent. Returns final raw_state."""
    adapter = DirectAdapter()
    agent.reset()
    adapter.reset(BACK, STAKE, seed)

    actions = 0
    for _ in range(max_steps):
        if adapter.done:
            break
        phase = adapter.raw_state.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        gs = adapter.raw_state
        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": legal}
        fa = agent.act({}, mask, info)

        engine_action = factored_to_engine_action(fa, gs)
        adapter.step(engine_action)
        actions += 1

    gs = adapter.raw_state
    gs["_actions_taken"] = actions
    gs["_won"] = adapter.won
    gs["_done"] = adapter.done
    return gs


class TestProtocol:
    def test_heuristic_agent_satisfies_protocol(self):
        assert isinstance(HeuristicAgent(), Agent)


class TestHeuristicAgent:
    def test_completes_episode(self):
        agent = HeuristicAgent()
        gs = _run_agent_episode(agent, seed="HEURISTIC_1")
        assert gs["_actions_taken"] > 0
        assert gs["_done"] or gs["_won"]

    def test_multiple_seeds(self):
        agent = HeuristicAgent()
        for i in range(5):
            gs = _run_agent_episode(agent, seed=f"HEURISTIC_{i}")
            assert gs["_actions_taken"] > 0

    def test_produces_valid_factored_actions(self):
        adapter = DirectAdapter()
        agent = HeuristicAgent()
        agent.reset()
        adapter.reset(BACK, STAKE, "HEURISTIC_FA")

        for _ in range(100):
            if adapter.done:
                break
            phase = adapter.raw_state.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            gs = adapter.raw_state
            legal = adapter.get_legal_actions()
            if not legal:
                break

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)

            assert isinstance(fa, FactoredAction)
            assert 0 <= fa.action_type < 21
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)

    def test_never_discards_on_last_hand(self):
        """Discarding scores nothing — on the last hand of a round the
        agent must always play, never discard, even if the best available
        hand is weak."""
        adapter = DirectAdapter()
        agent = HeuristicAgent()
        agent.reset()
        adapter.reset(BACK, STAKE, "HEURISTIC_LAST_HAND")

        # Get past blind select into SELECTING_HAND.
        gs = adapter.raw_state
        legal = adapter.get_legal_actions()
        mask = get_action_mask(gs)
        fa = agent.act({}, mask, {"raw_state": gs, "legal_actions": legal})
        adapter.step(factored_to_engine_action(fa, gs))

        gs = adapter.raw_state
        assert gs.get("phase") == GamePhase.SELECTING_HAND

        # Force hands_left to 1 (last hand) regardless of what it actually
        # is, so the guard is exercised deterministically.
        gs["current_round"]["hands_left"] = 1
        gs["current_round"]["discards_left"] = 1

        mask = get_action_mask(gs)
        legal = adapter.get_legal_actions()
        fa = agent.act({}, mask, {"raw_state": gs, "legal_actions": legal})
        assert fa.action_type == ActionType.PlayHand

    def test_lookahead_does_not_advance_real_game(self):
        """The core lookahead algorithm scores up to ~218 PlayHand
        candidates by deep-copying raw_state and running the real engine on
        each copy. This proves that isolation actually holds — that act()
        never mutates the real adapter state or advances its RNG — rather
        than relying only on reading the deepcopy call in the source. If a
        future change accidentally scored a candidate against the live
        state (e.g. a missing copy.deepcopy), this would catch it: the real
        hand/chips/hands_left/discards_left/RNG stream would change even
        though no real action was taken."""
        adapter = DirectAdapter()
        agent = HeuristicAgent()
        agent.reset()
        adapter.reset(BACK, STAKE, "HEURISTIC_ISOLATION")

        # Advance to SELECTING_HAND, where the lookahead runs.
        gs = adapter.raw_state
        legal = adapter.get_legal_actions()
        mask = get_action_mask(gs)
        fa = agent.act({}, mask, {"raw_state": gs, "legal_actions": legal})
        adapter.step(factored_to_engine_action(fa, gs))

        gs = adapter.raw_state
        assert gs.get("phase") == GamePhase.SELECTING_HAND

        cr = gs["current_round"]
        before = {
            "hand_len": len(gs.get("hand", [])),
            "chips": gs.get("chips", 0),
            "hands_left": cr.get("hands_left", 0),
            "discards_left": cr.get("discards_left", 0),
            "rng_state": dict(gs["rng"]._state),
        }

        mask = get_action_mask(gs)
        legal = adapter.get_legal_actions()
        agent.act({}, mask, {"raw_state": gs, "legal_actions": legal})

        gs_after = adapter.raw_state
        assert gs_after is gs  # same object — no swap-in of a copy
        cr_after = gs_after["current_round"]
        assert len(gs_after.get("hand", [])) == before["hand_len"]
        assert gs_after.get("chips", 0) == before["chips"]
        assert cr_after.get("hands_left", 0) == before["hands_left"]
        assert cr_after.get("discards_left", 0) == before["discards_left"]
        assert dict(gs_after["rng"]._state) == before["rng_state"]

    def test_heuristic_beats_random_on_fixed_seeds(self):
        """Loose regression guard: mean ante over a handful of fixed seeds
        should be higher for the heuristic than for RandomAgent. Aggregate,
        not per-seed, since RandomAgent samples from Python's unseeded
        global random module."""
        random.seed(1234)
        seeds = [f"HEURISTIC_VS_RANDOM_{i}" for i in range(5)]

        heuristic_antes = []
        for seed in seeds:
            gs = _run_agent_episode(HeuristicAgent(), seed=seed)
            heuristic_antes.append(gs.get("round_resets", {}).get("ante", 1))

        random_antes = []
        for seed in seeds:
            gs = _run_agent_episode(RandomAgent(), seed=seed)
            random_antes.append(gs.get("round_resets", {}).get("ante", 1))

        assert sum(heuristic_antes) / len(heuristic_antes) > sum(random_antes) / len(random_antes)
