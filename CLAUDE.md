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
python scripts/train_ppo.py --total-timesteps 500000 --lookahead   # + §5.3 L2 features
python scripts/train_ppo.py --total-timesteps 500000 --vec-env subproc --n-envs 8  # ~3x faster
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
  eval_ppo.py      Eval a checkpoint on the frozen seed set — the only real number
  eval_agent.py    Same, for the heuristic/random baselines
  bench_step.py    Where wall-clock goes: env vs network
  embed_drift.py   Did an embedding table actually learn anything
  lookahead_ablation.py  Does the policy actually use the lookahead features
  validate.py      Unified validation CLI (seed/crash/live/benchmark)
  lua_*_oracle.lua Lua reference oracles for RNG/scoring/hand-eval parity
docs/
  RL_PLAN.md       The plan: what to do next and why — read this
  RUNS.md          The lab notebook: every run, what it scored, what it taught
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

MaskablePPO on `MultiInputPolicy`, Red Deck / White Stake. Twenty runs;
per-run detail in `docs/RUNS.md`, the plan in `docs/RL_PLAN.md`.

| Agent | Mean ante | Max ante | Win rate |
|---|---|---|---|
| Random | 1.00 | — | 0% |
| **Heuristic** (exact one-step lookahead) | **2.75** | 7 | 0% |
| **Best PPO** (run 20: 1M steps, lookahead) | **1.87** | 6 | 0% |

The heuristic is `HeuristicAgent` (`uv run scripts/eval_agent.py --agent
heuristic --episodes 200`) and is the number PPO has to beat. It doesn't yet.

- **Three changes have produced confirmed gains, in order of discovery:**
  the deterministic top-K action table (issue #2, 1.26 → 1.42 across two
  seeds), the lookahead observation features (§5.3 L2, → ~1.52), and simply
  **training for 1M steps instead of 500k** (→ 1.74 group mean).
- **Only the last one moved the ceiling.** Every earlier change raised the
  ante-1 clear rate and left max ante at 3-4. Run 20 reaches **ante 6** with
  48/200 episodes at ante 3+ (vs 14-25 at 500k). Going deeper still needs
  jokers, economy and shop play, none of which has been worked on.
- **Stop tuning the embedding path.** Four `balatro`-extractor runs average
  ~1.14 vs the default extractor's 1.26: all-entity pooling (run 6, 1.00,
  broken), fixed pooling (runs 7/8, 1.26/1.13), small init (run 10, 1.07).
- **The embedding tables sit at ~97% of their random init — but exposure is
  *not* why.** The agent touches **95% of the reachable joker catalog** (142
  of 150 `j_` keys receive gradient in 500k steps), and did so before the
  action-table fix too. Each exposure just moves a row ~3% of its own length.
  The earlier "161 of 300 joker rows never seen" claim measured against the
  wrong denominator — the catalog is one shared 299-key ID space over all
  entity types, of which only 150 are jokers. **Corollary: more survival will
  not make the embeddings train.** The gap is credit assignment, not data.
  Reproduce with `uv run scripts/embed_drift.py <ckpt> <seed>`.
- **Hyperparameter tuning is exhausted.** Runs 3 and 4 had very different
  policy dynamics and near-identical task performance.
- **The seed-to-seed noise floor is ≈0.13 mean ante.** A single run landing
  0.1 above another means nothing; confirm every positive with a second seed.
- **The "NaN in logits" crash is solved, and it was never a NaN.** Four runs
  died to it (3, 4, 9, 14). Instrumented reproduction of run 14 caught the
  event: logits finite and healthy ([-14.21, -0.29]), and **one row of 256**
  whose float32 softmax summed to 1.0000010729 — past `Simplex()`'s fixed
  1e-6 tolerance by 7.3e-8. The trigger is an ordering bug in sb3-contrib's
  `MaskableCategorical.apply_masking`, which re-runs `Categorical.__init__`
  *before* refreshing `self.probs`, so PyTorch validates the stale probs from
  the previous **unmasked** 500-way parameterization. Fix:
  `Distribution.set_default_validate_args(False)` in `train_ppo.py` and
  `eval_ppo.py`; regression tests in `tests/env/test_action_distribution.py`.
  This retires "critic blowup" as the explanation for anything after run 4.
- **1M steps beats 500k, and it is the first change to move the ceiling.**
  Runs 19/20: 1.620/1.865 vs the 500k group's 1.460/1.535/1.565 (+0.22 on a
  0.13 noise floor, both above all three). Run 20 puts **48/200 episodes at
  ante 3+** against 14-25 before, and reaches ante 6. **Confounded with the LR
  schedule** — SB3 decays on fraction of *declared* total steps, so a 1M run
  is not a 500k run extended. It answers "is 1M with its natural schedule
  better", not "does step count alone help".
- **Lookahead features (§5.3 Level 2) give a real but inconsistent gain.**
  At 500k: 1.46/1.565 (lookahead) vs 1.420/1.420 (without) — non-overlapping
  on mean and clear rate, group gap +0.09, and run 15 is the first run to
  reach **ante 5** (the ceiling had been 3-4 since forever). The mean alone
  would be only suggestive; the **ablation** is what settles it — permuting
  the block costs run 15 **0.19 ante** (to 1.375, below the no-lookahead
  baseline), so its gain is causally attributable to the feature content.
  Run 13 barely reacts (−0.025). What varies across seeds is whether the
  policy *learns to use* the features, not whether they carry signal.
  All four lookahead runs (1.46, 1.565, 1.47, 1.45) beat both baselines
  (1.42, 1.42), so the direction is consistent even where margins are thin.
- **Rescaling the lookahead block's magnitude field did not help (runs
  16/17).** Dim 1 outweighs the decision-relevant ratio dims ~17× in raw
  pre-activation contribution, so dividing it by 10 looked obvious. Result:
  seed 1 fell 1.565 → 1.450, seed 0 moved 1.460 → 1.470. Ablation shows why —
  the block ended up mattering *less overall* (zeroed 1.02 → ~1.27) rather
  than differently. Default reverted to 1.0; knob kept.
  **This inference has now failed twice** (see also `embed_init_std`, run 10):
  "quiet the disproportionately loud input" reduces total contribution rather
  than rebalancing it. Treat it as suspect here. The mechanism that does what
  the intuition wants is `VecNormalize(norm_obs=True)` — global, and a
  different experiment.
- **`--total-timesteps` also changes the LR schedule.** SB3 schedules
  `learning_rate`/`clip_range` on fraction of *declared* total steps, so a
  longer run is not a clean extension of a shorter one.
- **Throughput: ~160 fps single-env, ~485 with `--vec-env subproc --n-envs 8`**
  (500k run: ~52 min → ~17 min). Two independent wins: `hand_eval.group_by_rank`
  shares the rank grouping across sizes (env 220 → 309 steps/sec), and parallel
  workers give a further 3.0x end-to-end. ~91% of env wall-clock is still
  `_enumerate_actions`. fps varies ±70% between runs on identical code — never
  read it as a code signal without `uv run scripts/bench_step.py`.
- **`n_steps` is per-env in SB3.** Use `--rollout-steps` (total) rather than
  raising `--n-envs` alone, or the rollout silently multiplies by worker count
  — a hyperparameter change, not a speedup.
- **Parallel collection is validated, not assumed.** Run 18 replicated run
  15's exact config with 8 workers: 1.535 vs 1.565 (inside the 0.13 noise
  floor) with a slightly *better* ante-1 clear rate, in 16.6 min instead of
  ~59. Parallel and serial runs are comparable.

---

## Known issues

Numbers are stable identifiers referenced from code comments and tests —
**never renumber, append**. Full detail with file:line in `docs/RL_PLAN.md` §4.

**Open:** #8 discard histogram is presence not counts (and covers the discard
pile, not the more useful draw pile) · #9 joker rarity is a `base_cost` proxy ·
#10 `ability_extra` sums unrelated fields into one float · #11 unseeded RNG on
the default (non-eval) paths · #12 `GameEnvironment.step` protocol arity drift ·
#17 `SubprocVecEnv` never enabled — the throughput lever.

**Fixed:** #1 `center_key` as a normalized float (raw IDs now emitted and
consumed by `BalatroExtractor`) · #2 random action-table subsampling
(deterministic top-K, recall@10 95%, 500 → ~33 entries) · #3 `seed_prefix`
collision across workers · #4 shared features extractor · #5 entity
`max_count` too low (now 20/10) · #6 `Monitor` lost behind `VecNormalize` ·
#7 metrics dead in sparse mode · #13 stale `action_heads` reference · #14
`_rng` unseeded on a `game_seed`-only reset (broke eval reproducibility) ·
#15 deterministic policy stalling on no-op loops (generic stall detector) ·
#16 `EntCoefSchedule` hardcoded denominator + `CheckpointCallback` writing to
a shared directory.

---

## Conventions

- **Never change reward shaping and hyperparameters in the same run.** This
  applies to architecture too — run 6 bundled embeddings *and* pooling and its
  negative result couldn't say which was responsible.
- **Confirm any positive result with a second seed.** The noise floor is ≈0.13
  mean ante; runs 6 and 10 both looked plausible and didn't survive.
- Checkpoint every 50k steps. A crash at 364k already cost a full run.
- Compare results on the frozen eval seed set, not rollout statistics —
  `uv run scripts/eval_agent.py --agent <random|heuristic> --episodes 200`,
  seeds from `jackdaw/env/eval_seeds.py::EVAL_SEEDS`.
- **Record every run in `docs/RUNS.md`** — the change, the eval number, the
  ante distribution, what it ruled in or out. `RL_PLAN.md` stays the plan.
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
- `max_steps=10_000` never binds; episodes are ~20-40 steps.
- **The catalog is one shared 299-key ID space** across all entity types (150
  `j_` jokers, 53 `c_` consumables, 32 vouchers, 32 packs, …). An embedding
  table sized `catalog_size + 1` can only ever be reached by its own prefix —
  use that subset as the denominator when reasoning about coverage, not the
  table height.
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

This is the project's main advantage, now used in all three places it can be:
the heuristic baseline (2.75) runs full exact lookahead; the action table uses
a cheap `evaluate_hand` ranking to decide which plays to offer; and
`obs["lookahead"]` (`--lookahead`, `docs/RL_PLAN.md` §5.3 Level 2) hands the
policy an 8-dim summary of that ranked menu — chiefly "does my best offered
play clear this blind now, or across the hands I have left". All three share
one scoring pass, so the observation channel costs +1.4% per step.

Exact lookahead costs 377 ms per decision (91% of it `deepcopy`), so anything
inside the training loop must use the cheap path.
