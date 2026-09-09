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

- Best: **mean ante ~1.08, ~1.9 blinds beaten, win rate 0**
- The agent dies inside ante 1 — it cannot clear a 300-chip small blind by
  sampling hands
- Two runs (3, 4) crashed near 360k steps with NaN logits
- Throughput: ~417 steps/sec single-env CPU
- **Hyperparameter tuning is exhausted.** Runs 3 and 4 had very different policy
  dynamics and near-identical task performance. The bottleneck is the
  observation encoding and the action space.

---

## Known issues

Ordered by impact. Full detail with file:line in `docs/RL_PLAN.md` §4.

1. **`center_key` encoded as a normalized float** (`observation.py:420`) —
   destroys joker identity. Needs `nn.Embedding` + integer ID channels.
   *Highest-value fix.*
2. **`_enumerate_actions` randomly subsamples** when legal actions > 500
   (`gymnasium_wrapper.py:331`) — legal actions vanish nondeterministically.
3. **`seed_prefix` collides across parallel workers** (`balatro_env.py`) — all
   `SubprocVecEnv` workers play identical games. Fails silently.
4. **Shared features extractor** — a value-function blowup corrupts the policy.
   Set `share_features_extractor=False`.
5. **Entity `max_count` too low** (`balatro_spec.py`: hand 8, jokers 5) — the
   agent can act on cards it never observed.
6. **`Monitor` lost when passing `VecNormalize`** — kills `ep_rew_mean` /
   `ep_len_mean` logging. Wrap inside the env lambda.
7. **Metrics broken when `reward_shaping=False`** — early return skips tracker
   updates.
8. **Stale reference to `action_heads`** in `balatro_spec.py`. No such module
   exists; there is no policy/encoder module in `jackdaw/env/`.

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

---

## Where the leverage is

The engine is **deterministic, seeded, and a pure state transition**. That means
candidate actions can be evaluated exactly by copying state, stepping, and
reading the result. In the playing phase this makes hand selection a solved
problem — best-of-218 exact evaluation, no learning required.

This is the project's main advantage and it is currently unused. See
`docs/RL_PLAN.md` §5.3.
