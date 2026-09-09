#!/usr/bin/env python3
"""Evaluate an agent on the frozen eval seed set.

Usage::

    uv run scripts/eval_agent.py --agent heuristic --episodes 200
    uv run scripts/eval_agent.py --agent random --episodes 50 --back b_red --stake 1
    uv run scripts/eval_agent.py --agent heuristic --episodes 200 --out results/heuristic_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from jackdaw.env.agents import RandomAgent
from jackdaw.env.eval_seeds import EVAL_SEEDS
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.heuristic_agent import HeuristicAgent
from jackdaw.env.rollout import evaluate_agent

AGENTS = {"random": RandomAgent, "heuristic": HeuristicAgent}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on the frozen eval seed set")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="heuristic")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of frozen seeds to use (<=200)"
    )
    parser.add_argument("--back", default="b_red")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--out", default=None, help="Optional path to write a JSON summary")
    args = parser.parse_args()

    seeds = EVAL_SEEDS[: args.episodes]
    agent = AGENTS[args.agent]()

    t0 = time.time()
    summary = evaluate_agent(
        agent,
        seeds,
        DirectAdapter,
        back_key=args.back,
        stake=args.stake,
        max_steps=args.max_steps,
    )
    elapsed = time.time() - t0

    print(f"Agent:       {args.agent}")
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
        out["agent"] = args.agent
        out["per_seed"] = [r.__dict__ for r in summary["results"]]
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
