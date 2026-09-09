# Jackdaw Balatro — RL Fork

Fork of [jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro), a
bit-exact headless Balatro simulator. Goal: train an agent to play and win
Red Deck / White Stake, then generalize to beat more stakes and decks.

**Read `docs/RL_PLAN.md` before doing any RL work.** It has the full diagnosis,
architecture decisions, and phase plan. This file is the short version.

---

## Commands

```bash
uv sync --dev                                # install with dev deps (test, lint)
uv sync --extra train                        # add training deps (torch, sb3, tensorboard)
python scripts/train_ppo.py --total-timesteps 500000
tensorboard --logdir runs/balatro_ppo
pytest                                       # tests
pytest --cov=jackdaw                         # with coverage
pytest -m benchmark                          # perf benchmarks
pytest -m live                               # needs BalatroBot running
pytest -m slow                               # tests that take >10s
ruff check . && ruff format .                # lint + format
jackdaw validate                             # ~250 engine validation scenarios — needs
                                              # a live BalatroBot server, see below
```

Python 3.12+, `uv` for dependency management. `--dev` matches CI and covers
general dev work; `--extra train` adds torch/sb3-contrib/tensorboard, only
needed for `train_ppo.py`.

`jackdaw validate` connects to a live BalatroBot RPC server (default
`127.0.0.1:12346`) — it is not standalone. If unreachable it exits 1 and
prints `uvx balatrobot serve --fast --no-audio --love-path <path>`. Filter
with `--category <name>` / `--scenario <name>`.

---

## Layout

```
jackdaw/
  engine/          Deterministic simulator (30 modules) — see warning below
    game.py          step() — core state transition
    scoring.py       14-phase scoring pipeline
    jokers.py        150 joker effects
    rng.py           Bit-exact 3-layer PRNG (matches LuaJIT 2.1)
  env/
    observation.py       235-dim global + entity encoding
    action_space.py      21 action types, masking
    balatro_env.py       BalatroEnvironment (factored interface)
    gymnasium_wrapper.py Flat Discrete(500) + masks for SB3
    balatro_spec.py      GameSpec instance for Balatro
    game_spec.py         Game-agnostic protocols
    game_interface.py    GameAdapter protocol — DirectAdapter, BridgeAdapter
  bridge/          Live validation via BalatroBot (Windows only)
  cli/             `jackdaw` entry point — `validate` subcommand + scenarios
scripts/
  train_ppo.py     MaskablePPO training
  validate.py      Unified validation CLI (seed/crash/live/benchmark)
  lua_*_oracle.lua Lua reference oracles for RNG/scoring/hand-eval parity
docs/
  RL_PLAN.md       Full project plan — read this
  rl-guide.md      Gymnasium wrapper / observation / action / reward guide
  validation.md    Validation & scenario-writing guide
tests/
  engine/          Incl. oracle/cross-validation tests against scripts/lua_*
  env/, bridge/, cli/, benchmarks/
  fixtures/        Golden JSON fixtures for oracle/cross-validation tests
```

---

## Critical constraints

**The engine must stay bit-exact.** `jackdaw/engine/` reimplements Balatro's
LuaJIT PRNG exactly. Do not modify engine internals without running
`jackdaw validate` (~250 scenarios). Breaking bit-exactness silently invalidates
every result and breaks live-game replay.

**Live-bridge parity.** Any observation or action-space change must be
computable from what `BridgeAdapter` can see, so trained policies can be
validated against real Balatro. If you add an observation feature, verify the
bridge can supply it.

**Two runtimes.** Training happens on Linux (ROCm/CPU). Live validation happens
on Windows, because BalatroBot requires the Windows Balatro executable. Keep
code cross-platform.

---

## Current state

MaskablePPO on `MultiInputPolicy`, Red Deck / White Stake.

- **Scripted heuristic baseline (exact one-step lookahead): mean ante 2.75,
  win rate 0%** — `HeuristicAgent`, `uv run scripts/eval_agent.py --agent
  heuristic --episodes 200`. This is the number PPO has to beat.
- **PPO run 5 (500k steps, first run post Phase-0 fixes): mean ante 1.26,
  win rate 0%** — first *real* eval-seed number (`scripts/eval_ppo.py`,
  `results/ppo_run5.json`); runs 1-4's "~1.08" was a rollout statistic, not
  an eval-seed one, and sparse-mode metrics were broken anyway (known issue
  #7). **Does not beat the heuristic.**
- Run 5 completed all 500k steps with **no NaN crash**, unlike runs 3/4 —
  plausibly the Phase 0 fixes, not confirmed (one run, not an ablation).
- **Phase 1's highest-value fix (joker/consumable/shop identity embeddings)
  is implemented**: `BalatroExtractor` (`jackdaw/env/feature_extractor.py`),
  `--extractor balatro` (default) in `train_ppo.py`.
  **Run 6 (first attempt, pooled all 5 entity types) regressed**: mean ante
  1.00 (vs run 5's 1.26) despite healthy training curves (`explained_variance`
  0.59, no crash) — pooling `hand_card` destroyed per-card identity
  `PlayHand`/`Discard` need. **Fixed**: pooling now only applies to
  joker/consumable/shop_item; hand_card/pack_card are flattened
  (position-preserving). Regression test added. **Runs 7 and 8 (the fix,
  two seeds): mean ante 1.26 and 1.13** — both in the same 1.1-1.3 band as
  run 5's default-extractor 1.265. The run-to-run spread *within* the fixed
  extractor (0.13) is as large as the gap to run 5, so at 500k steps the
  two architectures are statistically indistinguishable — **confirmed not
  broken (unlike run 6), but no detectable benefit yet either.**
- **Run 9 (attempted 1M-step run) crashed at step 409,600 — the NaN-in-logits
  failure from runs 3/4 came back**, with `share_features_extractor=False`
  in place and *without* the gradual `explained_variance`/`value_loss`
  precursor pattern that flagged runs 3/4 in advance — a flat, healthy
  training signal right up to a sudden crash. Turned out not to be a clean
  step-count test: `learning_rate`/`clip_range` are scheduled on *fraction
  of declared* `--total-timesteps`, so run 9's effective LR was >3× run 7's
  at the same step count (a longer declared horizon decays slower) —
  plausibly the actual trigger, echoing runs 3/4's original
  reward-scale/critic-blowup mechanism rather than proving the new
  architecture itself is unstable. Also found and fixed while
  investigating: `EntCoefSchedule` was hardcoded to `total_timesteps=500_000`
  regardless of the CLI flag, and `CheckpointCallback` wrote to a flat
  `checkpoints/` folder shared across every run using the same `--log-dir`
  (silently collision-prone — now a timestamped subfolder per invocation).
  Checkpointing itself worked as intended: only ~10k steps were lost, not
  360k+. **Any future long run should be read as also having a different
  LR/clip_range trajectory, not a clean extension, until that's addressed.**
  Full diagnosis in `docs/RL_PLAN.md` §3 "Run 6" through "Run 9".
- 25% of run 5's eval episodes used to stall at ante 1 for the full
  2,000-step budget, spamming a no-op action (`SwapHandLeft`) instead of
  playing — **known issue #15, fixed**: a generic stall detector in
  `BalatroGymnasiumEnv` now force-truncates + penalizes N steps without
  chip/round/ante/hand/discard/dollar progress. Validated against the
  existing Run 5 checkpoint with no retraining: mean length 520.9 → 30.0,
  eval throughput 1.6 → 26.1 eps/sec, mean ante unchanged (1.26).
- Throughput: ~417-540 steps/sec single-env CPU (varies by hand/joker size
  now that entity max_counts are raised, issue #5).
- **Hyperparameter tuning is exhausted.** Runs 3 and 4 had very different policy
  dynamics and near-identical task performance. The bottleneck is the
  observation encoding and the action space.

---

## Known issues

Ordered by impact. Full detail with file:line in `docs/RL_PLAN.md` §4.

1. **`center_key` encoded as a normalized float** (`observation.py:420`) —
   destroys joker identity. *Highest-value fix* — **fixed**: raw integer IDs
   now also emitted (`observation.py::encode_catalog_ids`,
   `obs["{name}_ids"]`) and consumed by `jackdaw/env/feature_extractor.py::
   BalatroExtractor` (`nn.Embedding` + masked-mean pooling per entity
   type). Wire with `--extractor balatro` (default) in `train_ppo.py`. Not
   yet run at full 500k-step scale against the Run 5 baseline.
2. **`_enumerate_actions` randomly subsamples** when legal actions > 500
   (`gymnasium_wrapper.py:331`) — legal actions vanish nondeterministically.
3. **`seed_prefix` collides across parallel workers** (`balatro_env.py`) — all
   `SubprocVecEnv` workers play identical games. Fails silently.
4. **Shared features extractor** — a value-function blowup corrupts the policy.
   **Fixed**: `share_features_extractor=False` bundled into the
   `--extractor balatro` wiring above (train_ppo.py).
5. **Entity `max_count` too low** (`balatro_spec.py`: hand 8, jokers 5) — the
   agent can act on cards it never observed.
6. **`Monitor` lost when passing `VecNormalize`** — kills `ep_rew_mean` /
   `ep_len_mean` logging. Wrap inside the env lambda.
7. **Metrics broken when `reward_shaping=False`** — early return skips tracker
   updates.
8. **Stale reference to `action_heads`** in `balatro_spec.py`. No such module
   exists; there is no policy/encoder module in `jackdaw/env/`.
14. **`BalatroGymnasiumEnv._rng` unseeded on a `game_seed`-only reset —
    fixed.** Made eval on a "frozen" seed non-reproducible whenever
    action-table subsampling triggered. Doesn't affect the heuristic/random
    baselines (they use the factored `BalatroEnvironment` interface, never
    this RNG).
15. **Deterministic policy stalls on no-op action loops — fixed.**
    25% of run 5's eval episodes used to hit `max_steps=2000` stuck at
    ante 1, spamming `SwapHandLeft`. Generic no-progress stall detector
    in `BalatroGymnasiumEnv` now force-truncates + penalizes.

---

## Conventions

- **Never change reward shaping and hyperparameters in the same run.** Four
  existing runs are only partly comparable because of this.
- Checkpoint every 50k steps. A crash at 364k already cost a full run.
- Compare results on the frozen eval seed set, not rollout statistics —
  `uv run scripts/eval_agent.py --agent <random|heuristic> --episodes 200`,
  seeds from `jackdaw/env/eval_seeds.py::EVAL_SEEDS`.
- Pin `mean_ante_reached` charts to a `[1, 8]` axis.
- No new dependencies without asking.
- Add observation golden tests before refactoring `observation.py`.
- `ruff` for lint and format. Type hints throughout; `from __future__ import
  annotations` at the top of modules.

---

## Gotchas

- **`ent_coef` cannot be scheduled via the constructor.** SB3 only wraps
  `learning_rate`, `clip_range`, `clip_range_vf` with `get_schedule_fn`. Use a
  callback mutating `self.model.ent_coef`.
- **Reward scaling is mostly a no-op on the policy** — `normalize_advantage=True`
  standardizes advantages per minibatch. It still inflates value MSE
  quadratically. Use `VecNormalize(norm_reward=True)`.
- **No `MaskableRecurrentPPO` exists.** sb3-contrib ships `MaskablePPO` and
  `RecurrentPPO` separately; combining them means writing it.
- `_log_scale` is sign-symmetric and safe for negative dollars — not a NaN
  source.
- `max_steps=10_000` never binds; episodes are ~20 steps.
- **`spaces.MultiDiscrete`/`Discrete` get one-hot encoded by SB3 before a
  features extractor ever sees them** (`preprocess_obs`) — a raw-integer
  channel meant for `nn.Embedding` (e.g. `joker_ids`) must be `spaces.Box`
  instead, or it silently arrives as a huge one-hot tensor. A `forward()`
  unit test that hand-builds tensors won't catch this — it bypasses
  `preprocess_obs` entirely. Validate any new features-extractor input with
  a real `model.learn()` call.

---

## Where the leverage is

The engine is **deterministic, seeded, and a pure state transition**. That means
candidate actions can be evaluated exactly by copying state, stepping, and
reading the result. In the playing phase this makes hand selection a solved
problem — best-of-218 exact evaluation, no learning required.

This is the project's main advantage and it is currently unused. See
`docs/RL_PLAN.md` §5.3.
