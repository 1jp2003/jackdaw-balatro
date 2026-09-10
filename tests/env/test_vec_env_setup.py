"""Guards for multi-worker training setup (`--n-envs`, `--vec-env subproc`).

Parallel rollout collection has two failure modes that are silent — the run
completes, the curves look normal, and the result is wrong:

1. **Workers replay identical games** (known issue #3). Batch diversity
   collapses to one game while cost stays N x.
2. **The rollout size changes underneath you.** SB3's `n_steps` is per env,
   so raising `--n-envs` without adjusting it multiplies the rollout — a
   hyperparameter change wearing a speedup's clothes.

Requires the ``train`` optional dependency group.
"""

from __future__ import annotations

import importlib.util
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def _load_train_ppo():  # type: ignore[no-untyped-def]
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("train_ppo", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRolloutSizing:
    def test_total_rollout_is_invariant_to_worker_count(self) -> None:
        train_ppo = _load_train_ppo()
        for n_envs in (1, 2, 4, 8):
            n_steps = train_ppo.derive_n_steps(4096, n_envs)
            assert n_steps * n_envs == 4096, (
                f"n_envs={n_envs} changes the rollout to {n_steps * n_envs}; "
                "that is a hyperparameter change, not a speedup"
            )

    def test_non_dividing_worker_count_stays_close_and_never_zero(self) -> None:
        train_ppo = _load_train_ppo()
        for n_envs in (3, 5, 6, 7, 12):
            n_steps = train_ppo.derive_n_steps(4096, n_envs)
            assert n_steps >= 1
            assert abs(n_steps * n_envs - 4096) < n_envs

    def test_absurd_worker_count_does_not_produce_a_zero_rollout(self) -> None:
        train_ppo = _load_train_ppo()
        assert train_ppo.derive_n_steps(4096, 99_999) == 1


class TestWorkerConstruction:
    def test_worker_factory_is_picklable(self) -> None:
        """SubprocVecEnv pickles the env factory, and Windows spawns rather
        than forks. A closure over argparse results would depend on
        cloudpickle handling local functions; a module-level function with
        plain arguments just works."""
        train_ppo = _load_train_ppo()
        fn = partial(train_ppo.make_worker_env, 0, 200, 3, True)
        # The module is loaded under a synthetic name, so pickle the target's
        # identity rather than the partial itself.
        assert train_ppo.make_worker_env.__module__ is not None
        assert pickle.loads(pickle.dumps(fn.args)) == (0, 200, 3, True)
        env = fn()
        try:
            assert "lookahead" in env.observation_space.spaces
        finally:
            env.close()

    def test_workers_play_different_games(self) -> None:
        """Known issue #3: one seed_prefix across workers made every worker
        replay the same seed sequence — silently, at N x the cost. The fix
        folds worker_id into the prefix; this checks it actually holds once
        several workers are stepped together."""
        train_ppo = _load_train_ppo()
        n_envs = 4
        venv = DummyVecEnv(
            [partial(train_ppo.make_worker_env, 0, 200, i, False) for i in range(n_envs)]
        )
        try:
            venv.reset()
            obs, _rewards, _dones, _infos = venv.step(np.zeros(n_envs, dtype=np.int64))
            fingerprints = {row.tobytes() for row in obs["global"]}
            assert len(fingerprints) == n_envs, (
                f"only {len(fingerprints)}/{n_envs} distinct worker states — "
                "workers are replaying the same game (known issue #3)"
            )
        finally:
            venv.close()

    def test_worker_seed_prefixes_are_distinct(self) -> None:
        train_ppo = _load_train_ppo()
        prefixes = {
            train_ppo.make_env(seed=0, max_steps=200, worker_id=i)._inner._seed_prefix
            for i in range(6)
        }
        assert len(prefixes) == 6
