"""Train MaskablePPO on the Balatro gymnasium environment.

Requires the ``train`` optional dependency group::

    uv pip install -e ".[train]"

Usage::

    python scripts/train_ppo.py --total-timesteps 500000
    python scripts/train_ppo.py --total-timesteps 50000 --log-dir runs/exp1 --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


class BalatroMetricsCallback(BaseCallback):
    """Log Balatro-specific episode metrics to tensorboard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._antes: list[int] = []
        self._rounds: list[int] = []
        self._wins: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "balatro/ante_reached" in info:
                self._antes.append(info["balatro/ante_reached"])
                self._rounds.append(info["balatro/rounds_beaten"])
                self._wins.append(info["balatro/won"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._antes:
            return
        self.logger.record("balatro/mean_ante_reached", np.mean(self._antes))
        self.logger.record("balatro/max_ante_reached", np.max(self._antes))
        self.logger.record("balatro/mean_rounds_beaten", np.mean(self._rounds))
        self.logger.record("balatro/win_rate", np.mean(self._wins))
        self._antes.clear()
        self._rounds.clear()
        self._wins.clear()


def make_env(seed: int = 0, max_steps: int = 10_000, worker_id: int = 0) -> BalatroGymnasiumEnv:
    # worker_id must be folded into seed_prefix, not just `seed` — with
    # n_envs > 1, every worker shares the same `seed` (SB3 doesn't vary it
    # per sub-env), so a bare f"PPO_{seed}" prefix makes every worker replay
    # the identical seed sequence: batch diversity drops to 1 while cost
    # stays N× and it fails silently (known issue #3).
    return BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"PPO_{seed}_w{worker_id}",
        reward_shaping=True,
    )


class EntCoefSchedule(BaseCallback):
    def __init__(self, initial: float, final: float, total_timesteps: int):
        super().__init__()
        self.initial, self.final = initial, final
        self.total = total_timesteps
        assert total_timesteps > 100_000, f"schedule denominator too small: {total_timesteps}"

    def _on_rollout_start(self) -> None:
        frac = min(self.num_timesteps / self.total, 1.0)
        self.model.ent_coef = self.initial + frac * (self.final - self.initial)
        self.logger.record("train/ent_coef", self.model.ent_coef)

    def _on_step(self) -> bool:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="runs/balatro_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save a checkpoint every N env steps (0 disables). A prior run "
        "crashed with NaN logits near 364k steps with no checkpointing, "
        "losing the entire run — see CLAUDE.md Conventions.",
    )
    parser.add_argument(
        "--extractor",
        choices=["balatro", "default"],
        default="balatro",
        help="'balatro' (default): BalatroExtractor — per-entity-type "
        "embeddings + masked-mean pooling (docs/RL_PLAN.md Sec 5.1/5.2), "
        "with share_features_extractor=False (known issue #4). 'default': "
        "SB3's flatten+concat CombinedExtractor, for an isolated "
        "architecture-only ablation against a run with everything else "
        "identical.",
    )

    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Monitor must wrap each sub-env directly: SB3's _wrap_env only inserts
    # Monitor when the env isn't already a VecEnv, so wrapping after
    # DummyVecEnv/VecNormalize silently drops rollout/ep_rew_mean and
    # ep_len_mean (known issue #6).
    def _make_worker_env(worker_id: int) -> Monitor:
        return Monitor(make_env(seed=args.seed, max_steps=args.max_steps, worker_id=worker_id))

    env = VecNormalize(
        DummyVecEnv([lambda i=i: _make_worker_env(i) for i in range(args.n_envs)]),
        norm_obs=False,
        norm_reward=True,
        gamma=0.99,
    )

    policy_kwargs: dict = {}
    if args.extractor == "balatro":
        from jackdaw.env.feature_extractor import BalatroExtractor

        policy_kwargs["features_extractor_class"] = BalatroExtractor
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": 256}
        # known issue #4: the default shared extractor let a critic
        # value-function blowup corrupt the policy through shared weights.
        policy_kwargs["share_features_extractor"] = False

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_path),
        learning_rate=lambda p: 3e-4 * p,
        ent_coef=0.005,
        n_steps=4096,
        batch_size=256,
        clip_range=0.15,
        clip_range_vf=0.2,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
    )
    print(f"Training for {args.total_timesteps} timesteps...")
    callbacks: list[BaseCallback] = [
        BalatroMetricsCallback(),
        EntCoefSchedule(initial=0.005, final=0.001, total_timesteps=args.total_timesteps),
    ]
    if args.checkpoint_freq > 0:
        # save_freq counts VecEnv.step() calls, not total env-steps, so it
        # must be divided by n_envs to checkpoint every checkpoint_freq
        # real env-steps regardless of how many workers are running.
        #
        # The checkpoint dir must be unique per invocation, not a flat
        # sibling of log_path shared by every run that uses the default
        # --log-dir: two separate `train_ppo.py` runs reaching the same
        # step count would otherwise write the identical filename
        # (balatro_ppo_<N>_steps.zip) and silently clobber each other's
        # crash-recovery checkpoint. A run that crashes only benefits from
        # checkpointing if a *later* run can't have overwritten its
        # recovery point first.
        checkpoint_dir = log_path / "checkpoints" / time.strftime("%Y%m%d_%H%M%S")
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="balatro_ppo",
                save_vecnormalize=True,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    save_path = log_path / "balatro_ppo"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
