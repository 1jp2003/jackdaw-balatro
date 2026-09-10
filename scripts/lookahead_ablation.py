#!/usr/bin/env python3
"""Does a trained policy actually USE the lookahead observation block?

A feature can be wired correctly, train without error, and still be ignored —
which is indistinguishable from "the feature didn't help" if you only look at
the eval number. The joker embeddings were exactly this case for four runs
before anyone measured it (docs/RUNS.md).

Three conditions on one checkpoint and the same frozen seeds:

``intact``
    Normal eval, for reference.
``zeroed``
    The block forced to 0. Proves whether the wiring is live at all, but it
    is badly out-of-distribution — the policy has never seen the
    availability flag clear while a menu exists — so a collapse here does
    NOT show the content is used.
``permuted``
    The block replaced by one sampled from a *different* real state. This is
    the honest test: it preserves the block's marginal distribution and
    destroys only the correspondence between a state and its own features.
    A policy conditioning on these values degrades sharply; one ignoring
    them does not move.

Read zeroed and permuted together. Separately, each one misleads.

Requires the ``train`` optional dependency group.

Usage::

    uv run scripts/lookahead_ablation.py runs/balatro_ppo/balatro_ppo_run13.zip
    uv run scripts/lookahead_ablation.py <checkpoint> --episodes 200
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from sb3_contrib import MaskablePPO

from jackdaw.env.eval_seeds import EVAL_SEEDS
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


def _fresh_env(max_steps: int) -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=max_steps, lookahead_features=True
    )


def collect_blocks(model: MaskablePPO, seeds: list[str], max_steps: int) -> np.ndarray:
    """Pool real lookahead blocks to draw the permuted condition from.

    Only blocks from card-select states (availability flag set) are pooled,
    so the permutation stays in-distribution — swapping in a block whose
    flag disagrees with the state would make this the zeroed test again.
    """
    pool: list[np.ndarray] = []
    for seed in seeds:
        env = _fresh_env(max_steps)
        obs, info = env.reset(options={"game_seed": seed})
        mask = info["action_mask"]
        done = False
        while not done:
            if obs["lookahead"][0] == 1.0:
                pool.append(obs["lookahead"].copy())
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            mask = info["action_mask"]
            done = terminated or truncated
    if not pool:
        raise SystemExit("no card-select states reached — nothing to permute")
    return np.stack(pool)


def evaluate(
    model: MaskablePPO,
    seeds: list[str],
    mode: str,
    pool: np.ndarray,
    rng: np.random.Generator,
    max_steps: int,
) -> tuple[float, int]:
    antes: list[int] = []
    for seed in seeds:
        env = _fresh_env(max_steps)
        obs, info = env.reset(options={"game_seed": seed})
        mask = info["action_mask"]
        step_info: dict = {}
        done = False
        while not done:
            if mode == "zeroed":
                obs["lookahead"] = np.zeros_like(obs["lookahead"])
            elif mode == "permuted" and obs["lookahead"][0] == 1.0:
                obs["lookahead"] = pool[rng.integers(len(pool))].copy()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _reward, terminated, truncated, step_info = env.step(int(action))
            mask = step_info["action_mask"]
            done = terminated or truncated
        antes.append(int(step_info.get("balatro/ante_reached", 1)))
    return float(np.mean(antes)), sum(1 for a in antes if a > 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="A model trained with --lookahead")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=2_000)
    args = parser.parse_args()

    model = MaskablePPO.load(args.checkpoint, device="cpu")
    if "lookahead" not in getattr(model.observation_space, "spaces", {}):
        raise SystemExit(
            f"{args.checkpoint} was trained without the lookahead channel — "
            "there is nothing to ablate."
        )

    seeds = EVAL_SEEDS[: args.episodes]
    pool = collect_blocks(model, seeds[: min(40, len(seeds))], args.max_steps)
    print(f"{args.checkpoint}")
    print(f"{len(pool)} real lookahead blocks pooled for the permuted condition\n")

    print(f"{'condition':<12} {'mean ante':>10} {'past ante 1':>14}")
    rng = np.random.default_rng(0)
    for mode in ("intact", "zeroed", "permuted"):
        mean_ante, past = evaluate(model, seeds, mode, pool, rng, args.max_steps)
        print(f"{mode:<12} {mean_ante:>10.3f} {f'{past}/{len(seeds)}':>14}")

    print(
        "\nintact ~= permuted means the policy is not conditioning on these "
        "values.\nA collapse under 'zeroed' alone only shows the wiring is "
        "live — that condition is\nout-of-distribution by construction."
    )


if __name__ == "__main__":
    main()
