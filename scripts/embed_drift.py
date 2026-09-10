#!/usr/bin/env python3
"""Measure how far a checkpoint's catalog embedding tables moved from init.

SB3 calls ``set_random_seed(seed)`` inside ``_setup_model()`` *before*
constructing the policy, so rebuilding a ``MaskablePPO`` with identical
arguments and the same ``--seed`` reproduces the run's initial weights
bit-for-bit. Diffing against that reconstruction is the decisive test for
"did this embedding table actually learn anything" — per-row *norm* checks
are not sufficient, since an untrained table's norm spread already matches
the chi distribution's natural spread.

A row that never received a nonzero gradient stays bit-identical to init
(Adam's moments stay 0 for it and there is no weight decay), so rows at
distance exactly 0 are an exact count of catalog entries never observed.

Two denominators matter and only one is meaningful. The catalog is a single
shared ID space over every key in ``centers.json``, but a row in the
``joker`` table can only ever be reached by a ``j_`` key — so coverage is
reported against the reachable subset, not the table height. Measuring
against table height understates coverage badly (see docs/RUNS.md).

**The reconstruction is only valid for checkpoints from the same code
revision.** Anything that changes the number or shape of tensors built
before the embeddings also changes how much RNG was consumed first. A shape
mismatch is caught and raises; a same-shape/different-stream mismatch is
not, and shows up as ~140% drift with *every* row marked seen. Read that
combination as "wrong init", never as a result — checkpoints from before
the run-11 code revision currently fail this way.

Requires the ``train`` optional dependency group.

Usage::

    uv run scripts/embed_drift.py runs/balatro_ppo/balatro_ppo_run11.zip 0
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jackdaw.env.feature_extractor import BalatroExtractor
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from jackdaw.env.observation import center_key_id, iter_center_keys

# Which center-key prefixes can ever land in each entity type's table. A
# shop slot can hold a joker, a consumable, a voucher or a booster pack.
_REACHABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "joker": ("j_",),
    "consumable": ("c_",),
    "shop_item": ("j_", "c_", "v_", "p_"),
}


def build_fresh(seed: int, max_steps: int = 2_000) -> MaskablePPO:
    """Rebuild the model exactly as scripts/train_ppo.py does.

    Must stay in sync with ``train_ppo.py``'s ``MaskablePPO(...)`` call: any
    parameter that changes the *number or shape* of tensors constructed
    before the embeddings also changes how much RNG is consumed first, which
    changes the reconstructed init.
    """

    def _make() -> Monitor:
        return Monitor(
            BalatroGymnasiumEnv(
                adapter_factory=DirectAdapter,
                max_steps=max_steps,
                seed_prefix=f"PPO_{seed}_w0",
                reward_shaping=True,
            )
        )

    env = VecNormalize(DummyVecEnv([_make]), norm_obs=False, norm_reward=True, gamma=0.99)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=lambda p: 3e-4 * p,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.15,
        clip_range_vf=0.2,
        target_kl=0.02,
        policy_kwargs={
            "features_extractor_class": BalatroExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "share_features_extractor": False,
        },
    )


def embedding_tables(model: MaskablePPO) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.policy.named_parameters()
        if ".embeddings." in name and name.endswith(".weight")
    }


def reachable_rows() -> dict[str, list[int]]:
    """Row indices each entity type's embedding table can actually reach."""
    keys = list(iter_center_keys())
    return {
        entity: [center_key_id(k) for k in keys if k.startswith(prefixes)]
        for entity, prefixes in _REACHABLE_PREFIXES.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a saved model (.zip)")
    parser.add_argument("seed", type=int, help="The --seed the run was trained with")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    init = embedding_tables(build_fresh(args.seed))
    trained = embedding_tables(MaskablePPO.load(args.checkpoint, device="cpu"))
    pools = reachable_rows()

    print(f"\n=== {args.label or args.checkpoint} (seed {args.seed}) ===")
    header = f"{'table':<38} {'drift':>7} {'rows':>11} {'reachable':>16} {'row move':>9}"
    print(header)
    for name, w0 in init.items():
        w1 = trained[name]
        if w0.shape != w1.shape:
            raise SystemExit(
                f"{name}: init {tuple(w0.shape)} != checkpoint {tuple(w1.shape)} — the "
                "checkpoint was trained under a different code revision, so its init "
                "cannot be reconstructed."
            )
        delta = w1 - w0
        rel = (delta.norm() / w0.norm()).item()
        # index 0 is padding_idx — frozen by construction, excluded
        row_norm = delta.norm(dim=1)
        moved = row_norm[1:] > 0
        n_seen, n_rows = int(moved.sum()), moved.numel()
        mean_move = float(row_norm[1:][moved].mean()) if n_seen else 0.0

        entity = name.split(".")[-2]
        pool = pools[entity]
        pool_seen = sum(1 for i in pool if float(row_norm[i]) > 0)
        cov = f"{pool_seen}/{len(pool)} ({pool_seen / len(pool):.0%})"
        short = name.replace("features_extractor.embeddings.", "").replace(".weight", "")
        print(f"{short:<38} {rel:>6.2%} {n_seen:>5}/{n_rows:<5} {cov:>16} {mean_move:>9.4f}")

    w0 = next(iter(init.values()))
    mean_norm = float(w0[1:].norm(dim=1).mean())
    print(
        f"\nmean init row norm: {mean_norm:.3f} — a 'row move' of x means a seen row "
        f"travelled {1 / mean_norm:.1%}·x of its own length."
    )


if __name__ == "__main__":
    main()
