"""Train MaskablePPO on the Balatro gymnasium environment.

Requires the ``train`` optional dependency group::

    uv pip install -e ".[train]"

Usage::

    python scripts/train_ppo.py --total-timesteps 500000
    python scripts/train_ppo.py --total-timesteps 50000 --log-dir runs/exp1 --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
)

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv


class ParamHealthCallback(BaseCallback):
    """Catch a non-finite policy parameter at the update that created it.

    ``VecCheckNan`` (``--check-nan``) only inspects observations, actions and
    rewards crossing the env boundary — it cannot see a NaN born *inside* the
    network. This is the other half: it checks every policy parameter once per
    rollout (i.e. right after the previous ``train()`` call) and reports the
    first tensor to go non-finite, which localizes the failure to a module
    instead of leaving only a ``Simplex()`` traceback to read backwards.

    Also logs ``diag/max_abs_param`` every rollout so a slow weight blowup is
    visible as a trend rather than only as an eventual crash — run 9 crashed
    with *no* precursor in explained_variance/value_loss, so the existing
    early-warning signs (docs/RL_PLAN.md §8) were not sufficient there.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._should_stop = False

    def _on_rollout_start(self) -> None:
        bad: list[str] = []
        max_abs = 0.0
        for name, param in self.model.policy.named_parameters():
            data = param.detach()
            if not bool(torch.isfinite(data).all()):
                bad.append(name)
            else:
                max_abs = max(max_abs, float(data.abs().max()))
        self.logger.record("diag/max_abs_param", max_abs)
        if bad:
            print("\n=== ParamHealthCallback: NON-FINITE PARAMETERS ===")
            print(f"step={self.num_timesteps}")
            for name in bad:
                print(f"  non-finite: {name}")
            print("=== stopping training (parameters are unrecoverable) ===\n")
            # _on_rollout_start cannot halt training (only _on_step's return
            # value is honored), so latch it and stop on the next step.
            self._should_stop = True

    def _on_step(self) -> bool:
        return not self._should_stop


class NanHunterCallback(BaseCallback):
    """Catch a non-finite activation *at the module that produced it*.

    The existing two probes both miss the failure this project keeps hitting
    (runs 3, 4, 9, 14 — docs/RUNS.md): ``VecCheckNan`` only inspects the env
    boundary, and ``ParamHealthCallback`` runs once per rollout, so a NaN
    born inside a ``train()`` minibatch update surfaces several updates later
    as a bare ``Simplex()`` traceback with every precursor metric healthy.

    This installs a forward hook on every leaf module of the policy. The
    first one to emit a non-finite output reports its own inputs alongside
    it, which separates "this module created the NaN" (finite in, non-finite
    out — e.g. an overflow or a 0/0) from "it was handed one" (non-finite
    in), and names the layer either way.

    Diagnostic only: hooks fire on every forward pass, including all
    ``n_epochs`` gradient passes. Cost is a finiteness reduction per module
    per pass — real but modest. Do not leave it on for a clean timing run.
    """

    def __init__(self, dump_dir: Path | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._handles: list[Any] = []
        self._fired = False
        self._dump_dir = dump_dir
        self._orig_masking: Any = None
        self._masking_cls: Any = None

    def _on_training_start(self) -> None:
        for name, module in self.model.policy.named_modules():
            # Leaf modules only: a container's output is just its last
            # child's, which would report the wrong layer.
            if list(module.children()):
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        self._patch_masking()
        if self.verbose:
            print(f"NanHunter: watching {len(self._handles)} modules + apply_masking", flush=True)

    def _patch_masking(self) -> None:
        """Also watch the step module hooks structurally cannot see.

        The first instrumented reproduction of this crash fired *no* module
        hook, which localizes the fault: every layer's output is finite, and
        the invalid distribution is produced between `action_net` and
        `MaskableCategorical` — inside `apply_masking` and the softmax, which
        are functional ops, not `nn.Module`s. This wraps that step and
        reports the three things that can make `Simplex()` reject a
        finite-looking tensor: a non-finite logit, a row with every action
        masked out, and probabilities that do not sum to 1.
        """
        from sb3_contrib.common.maskable.distributions import MaskableCategorical

        self._masking_cls = MaskableCategorical
        self._orig_masking = MaskableCategorical.apply_masking
        hunter = self

        def patched(self_dist: Any, masks: Any) -> Any:  # noqa: ANN401
            logits = getattr(self_dist, "_original_logits", None)
            result = hunter._orig_masking(self_dist, masks)
            if hunter._fired:
                return result
            problems: list[str] = []
            if logits is not None and not bool(torch.isfinite(logits).all()):
                bad = int((~torch.isfinite(logits)).sum())
                problems.append(f"{bad} non-finite logits entering apply_masking")
            if masks is not None:
                m = torch.as_tensor(masks, dtype=torch.bool).reshape(self_dist.logits.shape)
                empty = int((~m.any(dim=-1)).sum())
                if empty:
                    problems.append(f"{empty} row(s) with EVERY action masked out")
            probs = self_dist.probs.detach()
            if not bool(torch.isfinite(probs).all()):
                problems.append(f"{int((~torch.isfinite(probs)).sum())} non-finite probs")
            if float(probs.min()) < 0.0:
                problems.append(f"negative prob, min={float(probs.min()):.4g}")
            dev = float((probs.sum(dim=-1) - 1.0).abs().max())
            if dev > 1e-6:
                problems.append(f"probs sum deviates by {dev:.4g} (Simplex tolerance is 1e-6)")
            if problems:
                hunter._fired = True
                hunter._report_masking(problems, logits, masks, probs)
            return result

        MaskableCategorical.apply_masking = patched  # type: ignore[method-assign]

    def _make_hook(self, name: str):  # type: ignore[no-untyped-def]
        def hook(module: Any, inputs: Any, output: Any) -> None:
            if self._fired or not isinstance(output, torch.Tensor):
                return
            if torch.isfinite(output).all():
                return
            self._fired = True
            self._report(name, module, inputs, output)

        return hook

    def _report(self, name: str, module: Any, inputs: Any, output: Any) -> None:
        print("\n=== NanHunter: first non-finite activation ===")
        print(f"step={self.num_timesteps}  module={name}  ({type(module).__name__})")
        n_bad = int((~torch.isfinite(output)).sum())
        print(f"output: shape={tuple(output.shape)} non-finite={n_bad}/{output.numel()}")

        inputs_clean = True
        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor):
                continue
            finite = torch.isfinite(tensor)
            if not bool(finite.all()):
                inputs_clean = False
                print(f"input[{i}]: NON-FINITE {int((~finite).sum())}/{tensor.numel()}")
            else:
                print(f"input[{i}]: finite, max|x|={float(tensor.abs().max()):.4g}")

        for pname, param in module.named_parameters(recurse=False):
            data = param.detach()
            state = "finite" if bool(torch.isfinite(data).all()) else "NON-FINITE"
            print(f"param {pname}: {state}, max|w|={float(data.abs().max()):.4g}")

        print(
            "verdict: this module CREATED the non-finite value"
            if inputs_clean
            else "verdict: this module was HANDED a non-finite value (look upstream)"
        )

        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            target = self._dump_dir / f"nan_{self.num_timesteps}.pt"
            torch.save(
                {
                    "module": name,
                    "step": self.num_timesteps,
                    "inputs": [t.detach() for t in inputs if isinstance(t, torch.Tensor)],
                    "output": output.detach(),
                    "state_dict": self.model.policy.state_dict(),
                },
                target,
            )
            print(f"dumped to {target}")
        print("=== end NanHunter report ===\n")

    def _report_masking(self, problems: list[str], logits: Any, masks: Any, probs: Any) -> None:
        print("\n=== NanHunter: invalid action distribution ===", flush=True)
        print(f"step={self.num_timesteps}  (no module hook fired — fault is post-`action_net`)")
        for problem in problems:
            print(f"  ! {problem}")
        if logits is not None:
            finite = logits[torch.isfinite(logits)]
            if finite.numel():
                print(f"logits: max={float(finite.max()):.6g} min={float(finite.min()):.6g}")
        row_sums = probs.sum(dim=-1)
        worst = int(row_sums.sub(1.0).abs().argmax())
        print(f"worst row {worst}: sum={float(row_sums[worst]):.10f}")
        if masks is not None:
            m = torch.as_tensor(masks, dtype=torch.bool).reshape(probs.shape)
            print(f"row {worst}: {int(m[worst].sum())} of {m.shape[-1]} actions legal")
        if self._dump_dir is not None:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            target = self._dump_dir / f"dist_{self.num_timesteps}.pt"
            payload = {"step": self.num_timesteps, "probs": probs, "problems": problems}
            if logits is not None:
                payload["logits"] = logits.detach()
            if masks is not None:
                payload["masks"] = torch.as_tensor(masks, dtype=torch.bool)
            torch.save(payload, target)
            print(f"dumped to {target}")
        print("=== end NanHunter report ===\n", flush=True)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if getattr(self, "_orig_masking", None) is not None:
            self._masking_cls.apply_masking = self._orig_masking  # type: ignore[method-assign]
            self._orig_masking = None


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


def disable_distribution_validation() -> None:
    """Root-cause fix for the recurring "NaN in logits" crash (runs 3/4/9/14).

    It was never NaN. Instrumented reproduction of run 14's crash at step
    364,544 (docs/RUNS.md) caught the actual event: logits finite and healthy
    (range [-14.21, -0.29]), no non-finite value anywhere in the network, and
    **exactly one row of 256** whose softmax summed to 1.0000010729 — over
    ``Simplex()``'s fixed 1e-6 absolute tolerance by 7.3e-8.

    That is ordinary float32 error, not a training pathology. Converting the
    same float32 probabilities to float64 and re-summing still leaves them
    8.6e-7 from 1.0, so the values themselves cannot represent a simplex
    point that tightly — it is not a summation-order artifact that a better
    reduction would fix. A softmax over ``MAX_ACTIONS`` = 500 categories sits
    right at that tolerance, and each update evaluates 256 rows x 160
    minibatches, so over a 500k-step run the tail eventually gets hit. Which
    run dies, and when, is a dice roll — exactly matching the observed
    pattern of precursor-free crashes at unpredictable steps while
    ``explained_variance``, ``value_loss`` and ``max_abs_param`` all stayed
    healthy right up to the failure.

    Distribution argument validation is a debugging aid, off by default in
    PyTorch unless ``__debug__``. Disabling it costs nothing here and is
    standard for RL training loops. The genuine failure it might otherwise
    have caught — an actual NaN — is covered better by ``ParamHealthCallback``
    (always on) and ``--debug-nan``, both of which name the culprit instead of
    raising ``Simplex()`` several updates downstream.
    """
    torch.distributions.Distribution.set_default_validate_args(False)


def make_env(
    seed: int = 0,
    max_steps: int = 10_000,
    worker_id: int = 0,
    lookahead: bool = False,
) -> BalatroGymnasiumEnv:
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
        lookahead_features=lookahead,
    )


def derive_n_steps(rollout_steps: int, n_envs: int) -> int:
    """Per-env ``n_steps`` that keeps the TOTAL rollout at *rollout_steps*.

    SB3's ``n_steps`` is per environment: the rollout buffer holds
    ``n_steps * n_envs`` transitions. Raising ``--n-envs`` with ``n_steps``
    fixed therefore multiplies the rollout, which changes how many
    minibatches each update sees and how stale the oldest data in it is —
    that is a hyperparameter change, not a speedup, and it would make a
    parallel run incomparable to runs 11-17 for reasons having nothing to do
    with parallelism (docs/RL_PLAN.md §8 rule 1).
    """
    return max(rollout_steps // n_envs, 1)


def make_worker_env(seed: int, max_steps: int, worker_id: int, lookahead: bool) -> Monitor:
    """One training worker, Monitor-wrapped.

    Module-level and taking only plain arguments so it pickles under
    ``SubprocVecEnv``. Windows spawns rather than forks, which re-imports
    this module in each child — a closure over ``args`` would rely on
    cloudpickle's handling of local functions, and there is no reason to.

    Monitor must wrap each sub-env *here*, not around the VecEnv: SB3's
    ``_wrap_env`` only inserts Monitor when the env isn't already a VecEnv,
    so wrapping later silently drops ``rollout/ep_rew_mean`` and
    ``ep_len_mean`` (known issue #6).
    """
    return Monitor(
        make_env(seed=seed, max_steps=max_steps, worker_id=worker_id, lookahead=lookahead)
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
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
        help="'dummy' (default) steps every env in this process. 'subproc' "
        "runs each in its own process, which is the only way to parallelize "
        "the action-table ranking — ~91%% of env wall-clock (docs/RUNS.md, "
        "'Throughput'). Pointless at --n-envs 1, where it only adds IPC.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4096,
        help="TOTAL env steps per PPO update, across all workers. SB3's "
        "n_steps is per-env, so leaving it fixed while raising --n-envs "
        "silently multiplies the rollout (and thus the batch count per "
        "update) by n_envs — a different experiment, not a faster one. This "
        "holds the rollout constant instead: n_steps = rollout_steps // n_envs.",
    )
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
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Intra-op thread count for torch (default: torch's own choice, "
        "one per physical core). With --vec-env subproc the workers and the "
        "main process share cores, so this looks like it should be lowered — "
        "measured on a 6C/12T box at --n-envs 8 it should NOT be: 1 thread "
        "gives 451 fps, 2 gives 493, 6 (default) gives 506, and 4 gives 524. "
        "The network is not thread-starved and clamping it costs throughput. "
        "The optimum is machine-specific; re-measure before setting it.",
    )
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Add the lookahead observation channel (docs/RL_PLAN.md §5.3 "
        "Level 2): a summary of the ranked PlayHand/Discard menu the policy "
        "is choosing from — best offered value, whether it clears the blind "
        "now or across remaining hands, and how much the choice matters. "
        "Reuses the action table's existing scoring pass, so it is close to "
        "free. Off by default because it changes the observation space, "
        "which would make every existing checkpoint unloadable.",
    )
    parser.add_argument(
        "--debug-nan",
        action="store_true",
        help="Install forward hooks that name the module producing the first "
        "non-finite activation, and say whether it created the value or was "
        "handed one. This is the probe for the recurring NaN-in-logits crash "
        "(runs 3/4/9/14) that --check-nan and ParamHealthCallback both miss, "
        "because it is born inside a train() minibatch update. Has a per-"
        "forward cost; diagnosis only.",
    )
    parser.add_argument(
        "--check-nan",
        action="store_true",
        help="Wrap the env in VecCheckNan(raise_exception=True) to raise at "
        "the moment a NaN/inf appears in an observation, action or reward, "
        "naming which one — instead of surfacing later as a Simplex() "
        "traceback. Only sees the env boundary, NOT NaNs born inside the "
        "network (ParamHealthCallback covers that, and is always on). Has a "
        "per-step cost — use for diagnosis, drop for long runs.",
    )

    args = parser.parse_args()

    disable_distribution_validation()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    print(f"torch intra-op threads: {torch.get_num_threads()}")

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    n_steps = derive_n_steps(args.rollout_steps, args.n_envs)
    effective_rollout = n_steps * args.n_envs
    print(
        f"vec_env={args.vec_env} n_envs={args.n_envs} "
        f"n_steps={n_steps}/env -> rollout {effective_rollout} steps/update"
    )
    if effective_rollout != args.rollout_steps:
        print(
            f"  note: {args.rollout_steps} does not divide evenly by "
            f"{args.n_envs}; rollout is {effective_rollout}, not "
            f"{args.rollout_steps}. Powers of two divide cleanly."
        )
    if args.vec_env == "subproc" and args.n_envs == 1:
        print("  note: --vec-env subproc with one env only adds IPC overhead.")

    env_fns = [
        partial(make_worker_env, args.seed, args.max_steps, i, args.lookahead)
        for i in range(args.n_envs)
    ]
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    env = VecNormalize(
        vec_env_cls(env_fns),
        norm_obs=False,
        norm_reward=True,
        gamma=0.99,
    )
    if args.check_nan:
        # Outermost, so it inspects exactly what the model receives (i.e.
        # post-normalization values), not the raw pre-VecNormalize stream.
        env = VecCheckNan(env, raise_exception=True)

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
        n_steps=n_steps,
        batch_size=256,
        clip_range=0.15,
        clip_range_vf=0.2,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
    )
    # One id for every artifact this invocation produces, so a run's
    # checkpoints and its final model live together and can't be confused
    # with another run's. The pid matters: a bare second-resolution timestamp
    # collides when two runs are launched together (e.g. two seeds in
    # parallel), which would silently recreate the very clobbering that
    # defect #16 was about.
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = log_path / "checkpoints" / run_id

    print(f"Training for {args.total_timesteps} timesteps...")
    callbacks: list[BaseCallback] = [
        BalatroMetricsCallback(),
        ParamHealthCallback(),
        EntCoefSchedule(initial=0.005, final=0.001, total_timesteps=args.total_timesteps),
    ]
    if args.debug_nan:
        callbacks.append(NanHunterCallback(dump_dir=run_dir, verbose=1))
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
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(run_dir),
                name_prefix="balatro_ppo",
                save_vecnormalize=True,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # The authoritative save is per-invocation. `<log_dir>/balatro_ppo.zip`
    # was previously the *only* save, so every run silently overwrote the
    # previous run's finished model — the same collision the checkpoint
    # directory was already fixed for (defect #16), just at the final-save
    # path, and worse because the final model is the one that gets evaluated.
    # Run 13 overwrote run 12's before this was caught.
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "balatro_ppo_final"
    model.save(str(final_path))

    # Kept as a convenience pointer to the most recent run. Overwritten every
    # time by design — never cite it as a run's model, it does not stay put.
    latest_path = log_path / "balatro_ppo"
    model.save(str(latest_path))

    print(f"Model saved to {final_path}.zip")
    print(f"  (also copied to {latest_path}.zip, which the next run will overwrite)")


if __name__ == "__main__":
    main()
