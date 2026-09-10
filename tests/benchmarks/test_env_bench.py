"""Performance benchmarks for the env layer.

Run with: uv run pytest tests/benchmarks/test_env_bench.py -m benchmark
"""

from __future__ import annotations

import random
import time

import pytest

from jackdaw.engine.actions import Discard, PlayHand
from jackdaw.engine.game import IllegalActionError
from jackdaw.env.action_space import get_action_mask
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import encode_observation


def _resolve_action(action, gs: dict, rng: random.Random | None = None):
    """Fill in card indices for marker PlayHand/Discard actions."""
    rng = rng or random
    hand = gs.get("hand", [])
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = rng.randint(1, n)
        indices = tuple(sorted(rng.sample(range(len(hand)), count)))
        return PlayHand(card_indices=indices)
    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = rng.randint(1, n)
        indices = tuple(sorted(rng.sample(range(len(hand)), count)))
        return Discard(card_indices=indices)
    return action


def _make_adapter_at_hand_phase() -> tuple[DirectAdapter, dict]:
    """Create an adapter and advance to SELECTING_HAND phase."""
    from jackdaw.engine.actions import GamePhase, SelectBlind

    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_SEED_42")

    # Advance to selecting_hand
    for _ in range(10):
        gs = adapter.raw_state
        if gs.get("phase") == GamePhase.SELECTING_HAND:
            return adapter, gs
        legal = adapter.get_legal_actions()
        if not legal:
            break
        # Prefer SelectBlind to get to hand phase quickly
        select = [a for a in legal if isinstance(a, SelectBlind)]
        adapter.step(select[0] if select else legal[0])

    return adapter, adapter.raw_state


def _collect_game_states(n: int) -> list[dict]:
    """Collect diverse game states by playing random actions.

    Seeded, and tolerant of marker actions it cannot fill in. Previously it
    drew from the unseeded global ``random``, so each invocation walked a
    different action sequence — and whenever that walk opened a booster pack
    containing a targeted consumable, ``_resolve_action`` handed the engine a
    ``PickPackCard`` with no card targets and the benchmark died with
    ``IllegalActionError: c_sun requires between 1 and 3 target card(s)``.
    Measured on unchanged code, the suite failed 4 runs out of 5 that way,
    which makes every threshold in this file untrustworthy as a regression
    signal. Same class of problem as known issue #11.
    """
    rng = random.Random(20260909)
    adapter = DirectAdapter()
    adapter.reset("b_red", 1, "BENCH_COLLECT_0")
    states = []
    seed_counter = 0

    while len(states) < n:
        if adapter.done:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")

        states.append(adapter.raw_state)

        legal = adapter.get_legal_actions()
        if not legal:
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")
            continue
        try:
            adapter.step(_resolve_action(rng.choice(legal), adapter.raw_state, rng))
        except IllegalActionError:
            # A marker action this helper doesn't know how to fill in (e.g.
            # a pack card needing targets). Collecting states is the point,
            # not exercising every action type — move to a fresh game rather
            # than aborting the benchmark.
            seed_counter += 1
            adapter.reset("b_red", 1, f"BENCH_COLLECT_{seed_counter}")

    return states[:n]


@pytest.mark.benchmark
class TestEnvStepsPerSecond:
    """Assert that DirectAdapter env steps run fast enough for training."""

    def test_env_steps_per_second(self):
        """Full env step loop (engine + encode + mask) > 500 steps/sec.

        Note this drives `DirectAdapter` directly, so it does NOT build an
        action table — see `TestGymWrapperThroughput` for the path training
        actually runs, which is ~2x slower for exactly that reason.
        """
        rng = random.Random(20260909)
        adapter = DirectAdapter()
        adapter.reset("b_red", 1, "BENCH_SPS_0")
        n = 500
        done_count = 0

        t0 = time.perf_counter()
        for i in range(n):
            if adapter.done:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")

            gs = adapter.raw_state
            encode_observation(gs)
            get_action_mask(gs)

            legal = adapter.get_legal_actions()
            if not legal:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")
                continue
            try:
                adapter.step(_resolve_action(rng.choice(legal), adapter.raw_state, rng))
            except IllegalActionError:
                done_count += 1
                adapter.reset("b_red", 1, f"BENCH_SPS_{done_count}")

        elapsed = time.perf_counter() - t0
        sps = n / elapsed
        assert sps > 500, f"Only {sps:.0f} steps/sec (target: >500)"


@pytest.mark.benchmark
class TestEncodingLatency:
    """Assert encode_observation is fast enough."""

    def test_encoding_latency(self):
        """encode_observation < 500us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            encode_observation(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 500, f"encode_observation mean {mean_us:.1f}us (target: <500us)"


@pytest.mark.benchmark
class TestActionMaskLatency:
    """Assert get_action_mask is fast enough."""

    def test_action_mask_latency(self):
        """get_action_mask < 100us mean across diverse game states."""
        states = _collect_game_states(500)

        times = []
        for gs in states:
            t0 = time.perf_counter()
            get_action_mask(gs)
            times.append(time.perf_counter() - t0)

        mean_us = sum(times) / len(times) * 1e6
        assert mean_us < 100, f"get_action_mask mean {mean_us:.1f}us (target: <100us)"


@pytest.mark.benchmark
class TestGymWrapperThroughput:
    """Benchmark the path training actually runs.

    `TestEnvStepsPerSecond` above drives `DirectAdapter` directly, which
    never builds an action table — so it happily reported >500 steps/sec
    while real training ran at ~220, and the single most expensive component
    in the whole pipeline had no benchmark at all. These cover it.

    Reference points on the dev machine (6C/12T, single env, see
    docs/RUNS.md "Throughput"): enumeration was 4.26 ms/step before
    `hand_eval.group_by_rank` shared the rank grouping across sizes, and
    2.94 ms/step after, taking the wrapper from ~220 to ~309 steps/sec.
    Thresholds are set well below those so the test flags a real regression
    rather than ordinary machine-to-machine variation.
    """

    def _env(self):
        from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

        return BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=2_000)

    def test_gym_env_steps_per_second(self):
        """Full wrapper step (engine + encode + action table) > 120 steps/sec."""
        import numpy as np

        rng = np.random.default_rng(0)
        env = self._env()
        _, info = env.reset(seed=0)
        mask = info["action_mask"]

        n = 600
        t0 = time.perf_counter()
        for _ in range(n):
            action = int(rng.choice(np.nonzero(mask)[0]))
            _, _reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                _, info = env.reset()
                mask = info["action_mask"]
        sps = n / (time.perf_counter() - t0)

        assert sps > 120, f"Only {sps:.0f} gym steps/sec (target: >120, dev machine ~309)"

    def test_action_table_enumeration_latency(self):
        """_enumerate_actions < 8 ms mean — it is ~91% of env wall-clock.

        This is the hot path: a regression here scales the cost of every
        future training run, and nothing else in the suite would notice.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        env = self._env()
        original = env._enumerate_actions
        elapsed = 0.0
        calls = 0

        def measured(mask, info):
            nonlocal elapsed, calls
            start = time.perf_counter()
            out = original(mask, info)
            elapsed += time.perf_counter() - start
            calls += 1
            return out

        env._enumerate_actions = measured

        _, info = env.reset(seed=0)
        mask = info["action_mask"]
        for _ in range(400):
            action = int(rng.choice(np.nonzero(mask)[0]))
            _, _reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                _, info = env.reset()
                mask = info["action_mask"]

        mean_ms = elapsed / calls * 1000
        assert mean_ms < 8.0, f"_enumerate_actions mean {mean_ms:.2f}ms (target: <8ms, dev ~2.9ms)"
