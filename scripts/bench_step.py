#!/usr/bin/env python3
"""Decompose training throughput into env cost vs network cost.

SB3's ``time/fps`` folds together ``env.step()``, the policy forward pass
during rollout collection, and the amortized cost of ``train()``. When fps
moves between runs, this says which term moved — which matters, because on
this project's hardware the same code has measured 1.7x apart across runs
(see docs/RUNS.md "Throughput"), so an fps change is not by itself evidence
that a code change was expensive.

Requires the ``train`` optional dependency group.

Usage::

    uv run scripts/bench_step.py
    uv run scripts/bench_step.py --steps 5000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import CombinedExtractor

from jackdaw.env.feature_extractor import BalatroExtractor
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

# PPO update volume per env step: n_epochs * n_steps samples are replayed
# per n_steps collected, so each env step is paid for n_epochs times over
# in gradient work. Matches train_ppo.py's n_steps=4096 / batch_size=256.
_N_EPOCHS = 10


def bench_env(n_steps: int) -> float:
    """Env throughput under a uniformly-random legal action, splitting out
    the action-table enumeration from the engine transition itself."""
    rng = np.random.default_rng(0)
    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=2_000)

    elapsed = 0.0
    calls = 0
    original = env._enumerate_actions

    def timed(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal elapsed, calls
        start = time.perf_counter()
        out = original(*args, **kwargs)
        elapsed += time.perf_counter() - start
        calls += 1
        return out

    env._enumerate_actions = timed  # type: ignore[method-assign]

    _, info = env.reset(seed=0)
    mask = info["action_mask"]
    phases: dict[str, int] = {}
    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases_key = str(env._inner._adapter.raw_state.get("phase"))
        phases[phases_key] = phases.get(phases_key, 0) + 1
        action = int(rng.choice(np.nonzero(mask)[0]))
        _, _reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        if terminated or truncated:
            _, info = env.reset()
            mask = info["action_mask"]
    total = time.perf_counter() - t0

    print(f"env.step() only, random legal action: {n_steps / total:.0f} steps/sec")
    print(
        f"  action-table enumeration   {elapsed / calls * 1000:5.2f} ms/step  "
        f"({elapsed / total:.0%} of env time)"
    )
    print(f"  engine step + encoding     {(total - elapsed) / n_steps * 1000:5.2f} ms/step")
    print(
        "  step mix by phase: "
        + ", ".join(
            f"{p.split('.')[-1]} {n / n_steps:.0%}"
            for p, n in sorted(phases.items(), key=lambda kv: -kv[1])
        )
    )
    return n_steps / total


def _make_extractor(kind: str, space: object):  # type: ignore[no-untyped-def]
    if kind == "balatro":
        return BalatroExtractor(space, features_dim=256)  # type: ignore[arg-type]
    return CombinedExtractor(space)  # type: ignore[arg-type]


def bench_network(kind: str, rollout_iters: int, minibatch: int = 256) -> tuple[float, float]:
    """Return (rollout ms/step, train ms/step) attributable to the extractor.

    Both the policy and the value network run their own extractor, since
    train_ppo.py sets ``share_features_extractor=False`` (defect #4).
    """
    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=2_000)
    obs, _ = env.reset(seed=0)
    space = env.observation_space
    pi_ex, vf_ex = _make_extractor(kind, space), _make_extractor(kind, space)

    single = {k: torch.as_tensor(np.stack([v])) for k, v in obs.items()}
    single = preprocess_obs(single, space, normalize_images=False)
    with torch.no_grad():
        for _ in range(20):
            pi_ex(single)
        t0 = time.perf_counter()
        for _ in range(rollout_iters):
            pi_ex(single)
            vf_ex(single)
        rollout_ms = (time.perf_counter() - t0) / rollout_iters * 1000

    batch = {k: torch.as_tensor(np.stack([v] * minibatch)) for k, v in obs.items()}
    batch = preprocess_obs(batch, space, normalize_images=False)
    iters = 40
    if not any(p.requires_grad for p in pi_ex.parameters()):
        # SB3's CombinedExtractor over an all-Box Dict space is a
        # parameter-free flatten+concat: no weights, nothing to backprop.
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(iters):
                pi_ex(batch)
                vf_ex(batch)
            per_sample = (time.perf_counter() - t0) / iters / minibatch * 1000
        return rollout_ms, per_sample * _N_EPOCHS

    for _ in range(3):
        (pi_ex(batch).sum() + vf_ex(batch).sum()).backward()
    t0 = time.perf_counter()
    for _ in range(iters):
        pi_ex.zero_grad()
        vf_ex.zero_grad()
        (pi_ex(batch).sum() + vf_ex(batch).sum()).backward()
    per_sample = (time.perf_counter() - t0) / iters / minibatch * 1000
    return rollout_ms, per_sample * _N_EPOCHS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000, help="Env steps to time")
    args = parser.parse_args()

    torch.set_num_threads(1)
    env_fps = bench_env(args.steps)

    print()
    for kind in ("default", "balatro"):
        rollout_ms, train_ms = bench_network(kind, rollout_iters=2000)
        combined = 1.0 / (1.0 / env_fps + (rollout_ms + train_ms) / 1000)
        params = "parameter-free" if kind == "default" else "with embeddings + entity MLPs"
        print(f"{kind:>8} extractor ({params}):")
        print(f"  rollout fwd (pi+vf)        {rollout_ms:5.2f} ms/step")
        print(f"  train() fwd+bwd amortized  {train_ms:5.2f} ms/step")
        print(f"  => env + extractor ceiling {combined:5.0f} steps/sec")


if __name__ == "__main__":
    main()
