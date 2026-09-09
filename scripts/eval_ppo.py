#!/usr/bin/env python3
"""Evaluate a trained MaskablePPO checkpoint on the frozen eval seed set.

Drives ``BalatroGymnasiumEnv`` directly — the interface the model was
actually trained against — rather than ``jackdaw.env.rollout``'s
``Agent``-protocol harness, which targets the factored
``BalatroEnvironment`` interface instead. The two are different abstraction
levels in this codebase (see ``jackdaw/env/gymnasium_wrapper.py``'s module
docstring): a trained model's action indices only mean something against the
exact per-step action table ``BalatroGymnasiumEnv`` built when it queried
the policy, so evaluating a checkpoint has to happen at that same level.

Per CLAUDE.md's Conventions ("compare results on the frozen eval seed set,
not rollout statistics") — this is that comparison for a trained policy,
using the same ``EVAL_SEEDS`` the heuristic/random baselines in
``scripts/eval_agent.py`` use.

Requires the ``train`` optional dependency group (sb3-contrib, torch).

Usage::

    uv run scripts/eval_ppo.py --model runs/balatro_ppo/balatro_ppo --episodes 200
    uv run scripts/eval_ppo.py --model runs/balatro_ppo/checkpoints/balatro_ppo_500000_steps \
        --episodes 50 --out results/ppo_run5.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, ".")

from sb3_contrib import MaskablePPO

from jackdaw.env.eval_seeds import EVAL_SEEDS
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


@dataclass(frozen=True)
class PPOEpisodeResult:
    seed: str
    ante_reached: int
    rounds_beaten: int
    won: bool
    length: int


def run_ppo_episode(
    model: MaskablePPO,
    seed: str,
    *,
    back_key: str = "b_red",
    stake: int = 1,
    max_steps: int = 2_000,
    deterministic: bool = True,
) -> PPOEpisodeResult:
    """Run one episode of *model* against a fresh BalatroGymnasiumEnv.

    norm_obs=False in train_ppo.py's VecNormalize means the model was
    trained on raw (unnormalized) observations, so no VecNormalize stats
    need to be loaded/applied here for action selection.
    """
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        back_keys=[back_key],
        stakes=[stake],
        max_steps=max_steps,
    )
    obs, info = env.reset(options={"game_seed": seed})
    mask = info["action_mask"]

    length = 0
    step_info: dict = {}
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        obs, _reward, terminated, truncated, step_info = env.step(int(action))
        mask = step_info["action_mask"]
        length += 1

    return PPOEpisodeResult(
        seed=seed,
        ante_reached=int(step_info.get("balatro/ante_reached", 1)),
        rounds_beaten=int(step_info.get("balatro/rounds_beaten", 0)),
        won=bool(step_info.get("balatro/won", False)),
        length=length,
    )


def evaluate_ppo(model: MaskablePPO, seeds: list[str], **episode_kwargs: object) -> dict:
    results = [run_ppo_episode(model, seed, **episode_kwargs) for seed in seeds]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MaskablePPO checkpoint on the frozen eval seed set"
    )
    parser.add_argument("--model", required=True, help="Path to a saved model (.zip)")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of frozen seeds to use (<=200)"
    )
    parser.add_argument("--back", default="b_red")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking the argmax (deterministic is the default)",
    )
    parser.add_argument("--out", default=None, help="Optional path to write a JSON summary")
    args = parser.parse_args()

    seeds = EVAL_SEEDS[: args.episodes]
    model = MaskablePPO.load(args.model)

    t0 = time.time()
    summary = evaluate_ppo(
        model,
        seeds,
        back_key=args.back,
        stake=args.stake,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
    )
    elapsed = time.time() - t0

    print(f"Model:       {args.model}")
    print(f"Episodes:    {summary['episodes']}")
    print(f"Mean ante:   {summary['mean_ante']:.2f}")
    print(f"Max ante:    {summary['max_ante']}")
    print(f"Min ante:    {summary['min_ante']}")
    print(f"Win rate:    {summary['win_rate']:.1%}")
    print(f"Mean length: {summary['mean_length']:.1f}")
    eps_per_sec = summary["episodes"] / elapsed if elapsed > 0 else float("inf")
    print(f"Time:        {elapsed:.1f}s ({eps_per_sec:.2f} eps/sec)")

    if args.out:
        out = {k: v for k, v in summary.items() if k != "results"}
        out["model"] = args.model
        out["per_seed"] = [asdict(r) for r in summary["results"]]
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
