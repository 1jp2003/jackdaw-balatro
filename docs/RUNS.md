# Run log — Jackdaw Balatro RL

Every training run, what it changed, what it scored, and what was learned.
Split out of `RL_PLAN.md` so the plan can stay short and current while this
file grows: a run's detail stays useful as a record even after the problem
it found has been fixed, but it shouldn't clutter the document you read to
decide what to do next.

**The plan is `RL_PLAN.md`. This is the lab notebook.**

Conventions:
- The only comparable number is `mean_ante` from
  `uv run scripts/eval_ppo.py --model <path> --episodes 200` over the frozen
  `EVAL_SEEDS`. Rollout statistics from TensorBoard are directional only.
- One variable per run (`RL_PLAN.md` §8 rule 1).
- Per-seed data lives in `results/ppo_run<N>.json`.

---

## Run history

| Run | Change | Eval mean ante | Past ante 1 | fps | Outcome |
|---|---|---|---|---|---|
| 1-2 | Baseline (`ent_coef=0.05`, lr 1e-4) | ~1.05 (rollout) | — | — | 500k, stable |
| 3 | Reward ×10, flat blind reward, lr 3e-4, `ent_coef` 5e-4 (schedule bug) | ~1.07 (rollout) | — | — | **crashed 364k** |
| 4 | + `target_kl`, lr anneal, working `ent_coef` schedule | ~1.08 (rollout) | — | — | **crashed 360k** |
| 5 | Phase 0 defect fixes only (3/5/6/7); hyperparameters identical to run 4 | **1.265** | 31/200 | 524 | 500k, no crash — first real eval number |
| 6 | `--extractor balatro`: embeddings + masked-mean pooling on **all 5** entity types | **1.005** | 1/200 | 313 | 500k, no crash, **regression to random** |
| 7 | Pooling fix: pool only joker/consumable/shop_item, flatten hand_card/pack_card | **1.260** | 38/200 | 314 | 500k, no crash, parity with run 5 |
| 8 | Run 7, `--seed 1` | **1.130** | 21/200 | 187 | 500k, no crash — establishes the noise band |
| 9 | Run 7, `--total-timesteps 1000000` | — | — | 252 | **crashed 409,600** — and not a clean test |
| 10 | Run 7 + `BalatroExtractor(embed_init_std=0.1)` | **1.070** | 11/200 | 192 | 500k, no crash, worst non-buggy run |
| 11 | Deterministic top-K action table (defect #2 fix) | **1.420** | 73/200 | 172 | 500k, no crash — **first real gain** |
| 12 | Run 11, `--seed 1` | **1.420** | 70/200 | 171 | 500k, no crash — confirms run 11 |
| 13 | Run 11 + `--lookahead` (§5.3 Level 2 observation features) | **1.460** | 75/200 | 112 | 500k, no crash — inside the noise floor |
| 14 | Run 13, `--seed 1` | **1.580** @350k | 86/200 | 103 | **crashed ~364,544** — root-caused, see below |
| 15 | Run 14 retried after the `Simplex()` fix | **1.565** | 85/200 | 142 | 500k, no crash — first run to reach **ante 5** |
| 16 | Run 13 + `_LOOKAHEAD_VALUE_SCALE=10` (balance the block's field scales) | **1.470** | 79/200 | 129 | 500k, no crash — no gain |
| 17 | Run 16, `--seed 1` | **1.450** | 79/200 | 118 | 500k, no crash — **worse than run 15's 1.565** |
| 18 | Run 15 config, `--vec-env subproc --n-envs 8` | **1.535** | 89/200 | 505 | 500k in **16.6 min** — parallelism does not shift results |
| 19 | Run 18 at **1M steps** | **1.620** | 97/200 | 412 | 1M in 41 min, no crash |
| 20 | Run 19, `--seed 0` | **1.865** | 102/200 | 461 | 1M in 36 min — **best result so far**, max ante 6 |

Reference points: random 1.00, **heuristic 2.75** (the bar to beat), max
possible 8.

Runs 1-4 predate the eval harness; their numbers are training-rollout
averages and are not comparable to runs 5+. They are kept only because the
crash diagnosis below refers to them.

`fps` is whole-pipeline throughput and carries large exogenous variance —
see "Throughput" at the end of this file before reading anything into it.

---

## Run 5 — first eval-seed number (post Phase 0)

Runs 1-4's "ante ~1.08" was a *rollout* statistic (defect #7: sparse-mode
metrics were broken anyway, and even dense-mode rollout averages are exactly
what `CLAUDE.md`'s Conventions warn not to trust). Run 5 is the first run
with a real frozen-eval-seed number, via a harness this project didn't have
before.

| Agent | Mean ante | Max ante | Win rate |
|---|---|---|---|
| Random | 1.00 | — | 0% |
| **Heuristic** (`HeuristicAgent`, exact lookahead) | **2.75** | 7 | 0% |
| **PPO run 5** | **1.26** | 6 | 0% |

Training completed cleanly for the first time (no NaN crash — defects
3/5/6/7 plausibly helped, though this is one run, not an ablation).
`ep_rew_mean` reached 0.26 and `explained_variance` 0.54, so the policy was
learning *something*, just nowhere near one-step exact lookahead.

Two defects were found while building this eval: **#14** (`_rng` unseeded on
a string-only reset, making "frozen" eval seeds non-reproducible) and **#15**
(deterministic policy stalling in 25% of eval episodes). Both fixed — see
`RL_PLAN.md` §4.

Ante distribution: `{1:169, 2:15, 3:12, 4:3, 6:1}`.

---

## Run 6 — pooling everything: regression, diagnosed and fixed

Same 500k steps and hyperparameters as run 5, only change: `--extractor
balatro` (catalog-ID embeddings + masked-mean pooling on all 5 entity
types). No crash, `explained_variance` reached **0.59** (better than run 5's
0.54), `value_loss`/`loss` decreased smoothly — every training-time health
signal looked *good*.

Eval: **mean ante 1.00, max ante 2**, distribution `{1:199, 2:1}` — down to
random-baseline level. This is the canonical case behind `RL_PLAN.md` §8's
calibration note: healthy training curves + eval regression = an
architecture problem, not a training-dynamics or step-count problem.

**Root cause**: pooling applied uniformly to all 5 entity types, including
`hand_card`. Masked-mean pooling collapses N individual cards in specific
slots into one averaged vector — destroying exactly the per-slot identity
`PlayHand`/`Discard`'s `card_target` needs to select *which* cards to play.
Pooling only makes sense once something downstream consumes a per-entity
stream via attention/pointer lookup (`RL_PLAN.md` §5.4) instead of a flat
`Discrete(500)` linear layer reading one pooled vector.

Compounding methodology issue: run 6 bundled two representation changes
(embeddings *and* pooling) into one architecture tested as a unit, so a
negative result couldn't say which piece was responsible. §8 rule 1 applies
to architecture, not just to reward and hyperparameters.

**Fixed** in `feature_extractor.py`: pooling applies only to
`joker`/`consumable`/`shop_item`; `hand_card`/`pack_card` are flattened.
Regression test:
`tests/env/test_feature_extractor.py::test_hand_card_order_is_not_pooled_away`
swaps two hand-card slots and asserts the extractor's output changes.

---

## Runs 7 and 8 — the pooling fix: parity, and a measured noise band

Run 7 is run 6's fix, otherwise identical. No crash, `explained_variance`
0.52, `ep_rew_mean` 0.28. Run 8 is run 7 with `--seed 1` only.

| Run | Extractor | Seed | Mean ante | Distribution |
|---|---|---|---|---|
| 5 | default | 0 | 1.265 | `{1:169, 2:15, 3:12, 4:3, 6:1}` |
| 6 | balatro, all pooled (buggy) | 0 | 1.005 | `{1:199, 2:1}` |
| 7 | balatro, fixed | 0 | 1.260 | `{1:162, 2:25, 3:12, 4:1}` |
| 8 | balatro, fixed | 1 | 1.130 | `{1:179, 2:17, 3:3, 4:1}` |

Run 6 is an unambiguous outlier (the pooling bug, not noise). Runs 5/7/8 sit
in a tight 1.13-1.27 band, and **the seed-to-seed spread inside the identical
`balatro` config (7 vs 8: 0.13) is as large as the gap between either of them
and run 5's default-extractor point** — so at 500k steps they are
statistically indistinguishable.

That is an informative result, not a non-result: the fixed extractor is
**confirmed not broken (unlike run 6) and shows no detectable benefit** at
this budget. It also establishes the **≈0.13 ante noise floor** that every
later comparison is measured against.

---

## Run 9 — the crash came back, and the run wasn't a clean test

Intended as `--total-timesteps 1000000`, `--seed 0`, otherwise identical to
run 7. **Crashed at step 409,600** with the same failure as runs 3/4
(`ValueError: ... probs ... Simplex()` — NaN in the logits reaching
`MaskableCategorical`). Checkpointing worked: a 400k checkpoint survived, so
~10k steps were lost, not the 360k+ runs 3/4 lost.

Two things make this different from "runs 3/4's crash again":

1. **The precursor pattern is absent.** Runs 3/4 showed `explained_variance`
   sliding 0.9→0.40 and `value_loss` curving vertical for ~200k steps before
   crashing. Run 9's `explained_variance` (~0.55), `value_loss` (~0.7-0.9),
   `approx_kl` (~0.013) and `clip_fraction` (~0.16) were flat and healthy
   for the 60k+ steps immediately before the crash — a sudden, precursor-free
   event. Run 9 also used `share_features_extractor=False` (defect #4's fix)
   and still crashed, so that fix alone does not close this failure mode.
2. **It wasn't actually isolating step count.** `learning_rate` and
   `clip_range` are scheduled on SB3's `p` = fraction of *declared*
   `total_timesteps` remaining, not on absolute steps. At step 409,600 run 7
   was 82% through its schedule; run 9 was 41% through its — an effective
   learning rate **more than 3× higher** at the same step count. A held-longer,
   higher LR against a critic that still can't predict returns is exactly the
   mechanism runs 3/4 implicate.

**Two bugs found while investigating, both fixed:**
- `EntCoefSchedule` was hardcoded to `total_timesteps=500_000` regardless of
  `--total-timesteps`, so run 9 hit its floor at 500k and stayed there.
- `CheckpointCallback`'s `save_path` was a flat `<log_dir>/checkpoints/`
  shared by *every* invocation using the same `--log-dir`. Two runs reaching
  the same step count would silently overwrite each other's recovery
  checkpoint. Now a timestamped subfolder per invocation.

**Deliberately not fixed**: LR/`clip_range` remain
fraction-of-declared-`total_timesteps` schedules, which is SB3's own
convention. Changing that is itself a schedule design change needing its own
isolated validation — not something to do silently while chasing a crash.
**Any run with `--total-timesteps` ≠ 500,000 also has a different effective
LR trajectory, and is not a clean extension of prior runs.**

**Instrumentation added since**: `--check-nan` (`VecCheckNan`, env-boundary
NaNs only, has a per-step cost) and `ParamHealthCallback` (always on,
negligible — names the first non-finite parameter, halts training, and logs
`diag/max_abs_param` every rollout so a slow weight blowup shows as a
trend). These are complementary: `VecCheckNan` cannot see a NaN born inside
the network, which is where run 9's most likely originated — its weights
were finite and un-exploded at the 400k checkpoint (max |param| 4.56).

**Still open.** The NaN has been seen three times and root-caused once
(runs 3/4, below). Run 9's instance has instrumentation waiting for it but
no confirmed mechanism.

---

## Run 10 — small embedding init: hypothesis tested, not supported

Run 7 in every respect except `BalatroExtractor(embed_init_std=0.1)` instead
of `nn.Embedding`'s N(0,1) default. **Mean ante 1.07** (11/200 past ante 1) —
the lowest of any non-buggy run. Default reverted to 1.0.

The reasoning that motivated it: with N(0,1) init an untrained row is a fixed
random vector of norm ≈5.66 concatenated onto ~15 normalized feature dims, so
a joker's entity MLP sees 47 dims of which 32 are loud random constants — the
network has to learn to ignore two-thirds of its own input. Shrink the init
and an unseen row is ≈0, degrading gracefully to "continuous features only",
which is what the default extractor already scores 1.26 with.

**Why that was wrong**: an untrained N(0,1) row isn't only noise, it's also a
*random identity code* — a unique separable 32-dim signature per joker that
the downstream Linear can read to tell jokers apart with zero training (the
random-features effect). Shrinking init 10× removed that distinguishability
and made all jokers look alike, which cost more than the noise it removed.
The two framings predict opposite outcomes and the evidence favors the
latter. Worth remembering before reaching for the same fix elsewhere.

**Verdict on the embedding path after runs 6/7/8/10**: four `balatro`-extractor
runs averaging ~1.14 against a single default-extractor point at 1.26. Not
proof the architecture is harmful, but enough to stop tuning it.

---

## Runs 11 and 12 — deterministic top-K action table: first real progress

Defect #2 fixed. `PlayHand`/`Discard` menus are now a deterministic top-12
ranked by cheap joker-blind `evaluate_hand` value, with slots reserved per
hand type and for small-cardinality (setup) plays, instead of a random
subsample of ~218 candidates. Action table 500 → ~33 entries. Recall@10 of
that ranking against exact engine evaluation is **95%** (`RL_PLAN.md` §5.3).

Run 12 is run 11 with `--seed 1` only.

| | Mean ante | Range | Past ante 1 |
|---|---|---|---|
| Pre-fix (runs 5, 7, 8, 10) | 1.181 | 1.07-1.26 | 11-38 / 200 |
| **Top-K action table (runs 11, 12)** | **1.420** | **1.42-1.42** | **70-73 / 200** |

Run 11 distribution: `{1:127, 2:63, 3:9, 4:1}` — 63 episodes reached ante 2,
against 9-25 before. Training rollout stats moved in step
(`mean_ante_reached` 1.62, `ep_rew_mean` 0.605, `ep_len_mean` 40.4, all
roughly double their run-7/8 values), `explained_variance` 0.59, no NaN.

**The tightness is what makes this convincing, more than the size of the
gain.** Two independent seeds landed on 1.42/1.42 with clear rates of 73 and
70, against a pre-fix group scattered over 1.07-1.26 with clear rates from 11
to 38. The groups don't overlap on either metric and the within-group spread
collapsed. The mean gap alone (0.155 against a 0.13 measured noise floor)
would not have been conclusive. This is the first result to survive the
second-seed check that invalidated runs 6 and 10.

**What it did not fix**: the ceiling. Max ante is still 4, and slightly fewer
episodes reached ante 3. The agent got much better at *clearing the first
blind* and no better at going deep — consistent with having fixed
hand-selection availability while leaving scaling (jokers, economy, shop)
untouched. Still far from the heuristic's 2.75.

---

## Runs 13 and 14 — lookahead observation features: unresolved, and a crash

Run 11 plus `--lookahead` (§5.3 Level 2), same seed, nothing else changed.
Run 14 is run 13 with `--seed 1`.

Run 13 finished cleanly (`explained_variance` 0.519, `ep_len_mean` 37.9) at
**mean ante 1.460, 75/200 past ante 1** — against runs 11/12's 1.420/1.420
and 73/70. That is +0.04 on a **≈0.13 noise floor**, i.e. nothing yet.

**Run 14 crashed at ~step 364,500** (see below), so the confirming 500k seed
does not exist. Its 350k checkpoint survived, which makes a *same-step*
comparison across all four runs possible — and that comparison is the more
informative one, since every run is at the identical point in the LR
schedule:

| Run | Lookahead | Seed | @350k | Past ante 1 | Distribution @350k | @500k |
|---|---|---|---|---|---|---|
| 11 | no | 0 | 1.365 | 63/200 | `{1:137, 2:53, 3:10}` | 1.420 |
| 12 | no | 1 | 1.315 | 54/200 | `{1:146, 2:45, 3:9}` | 1.420 |
| 13 | **yes** | 0 | 1.410 | 72/200 | `{1:128, 2:62, 3:10}` | 1.460 |
| 14 | **yes** | 1 | **1.580** | **86/200** | `{1:114, 2:63, 3:18, 4:3, 5:2}` | *crashed* |

Run 15 is run 14 retried once the crash was root-caused and fixed. Same seed,
same config; it completed 500k cleanly, which is itself the practical
confirmation of that fix — the identical run had died at step 364,544 twice.
Its 350k checkpoint scores 1.580, matching crashed run 14's 350k exactly, as
determinism requires.

**Completed comparison at 500k:**

| Run | Lookahead | Seed | Mean ante | Past ante 1 | Max | Distribution |
|---|---|---|---|---|---|---|
| 11 | no | 0 | 1.420 | 73/200 | 4 | `{1:127, 2:63, 3:9, 4:1}` |
| 12 | no | 1 | 1.420 | 70/200 | 3 | `{1:130, 2:56, 3:14}` |
| 13 | **yes** | 0 | 1.460 | 75/200 | 4 | `{1:125, 2:61, 3:11, 4:3}` |
| 15 | **yes** | 1 | **1.565** | **85/200** | **5** | `{1:115, 2:60, 3:23, 4:1, 5:1}` |

Group means 1.420 vs 1.513, **+0.09**. The groups do not overlap on mean or on
ante-1 clear rate, and run 15 is the first 500k run to reach **ante 5** — the
ceiling had been stuck at 3-4 for every prior run. But the within-lookahead
spread is 0.105, close to the 0.13 noise floor, and run 13 sits only 0.04
above the non-lookahead pair. On the mean alone this would be *suggestive, not
established* — a much weaker pattern than the action-table result, which had a
0.24 gap with zero within-group spread.

### The ablation settles what the means cannot

Same three conditions on both lookahead checkpoints:

| Condition | Run 13 (seed 0) | Run 15 (seed 1) |
|---|---|---|
| intact | 1.460 (75/200) | **1.565 (85/200)** |
| **permuted** | 1.435 (66/200) | **1.375 (60/200)** |
| zeroed | 1.020 (3/200) | 1.020 (4/200) |

**Run 15's policy is genuinely conditioning on the features.** Swapping each
state's block for one from a *different* real state — same marginal
distribution, correspondence destroyed — costs it **0.19 ante** and drops it
to 1.375, i.e. to *below* the 1.420 non-lookahead baseline. Run 13's policy
barely reacts to the same treatment (−0.025).

That is the coherent reading of everything above: **the gain is causally
attributable to the feature content in the run that got a gain.** Both runs
land at roughly the baseline when permuted (1.435 and 1.375 vs 1.420); only
run 15 rises above it when the features are intact. So the feature works, and
what varies across seeds is whether the policy *learns to use it* — which is
also why the group spread is large relative to the group gap.

**Verdict: a real but inconsistently-realized gain.** Worth keeping. The
obvious lever for making it consistent is the scale imbalance below, which
is a cheap isolated change: dim 1 outweighs the decision-relevant dims by
~17x in raw pre-activation contribution, and it is the least useful dim in
the block.

### Run 14's crash — 4th occurrence, and the first with a real probe

Same `ValueError: ... probs ... Simplex()` as runs 3, 4 and 9: NaN in the
logits reaching `MaskableCategorical`, raised inside `train()`'s
`evaluate_actions`. Every precursor was **healthy**, and this time actively
improving:

| Metric | at 364,544 (last rollout before the crash) |
|---|---|
| `explained_variance` | 0.649 — *rising* |
| `value_loss` | 0.374 — *falling* |
| `approx_kl` | 0.0068 |
| `clip_fraction` | 0.103 |
| `diag/max_abs_param` | 4.48 — flat, no weight blowup |

That rules out the runs-3/4 mechanism (critic blowup inflating encoder
weights until float32 overflows): the weights never inflated. It matches run
9's precursor-free signature instead. Crash step is also suspiciously close to
runs 3 and 4 (364k, 360k, now ~364.5k), though runs 5/7/8/10/11/12/13 all
passed that point without incident, so it is not a hard threshold.

**Why the existing probes missed it, again.** `VecCheckNan` only inspects the
env boundary. `ParamHealthCallback` runs at `_on_rollout_start`, so a NaN born
during a minibatch update inside `train()` is never visible to it — by the
time the next rollout starts, the run has already died.

**New probe: `--debug-nan` (`NanHunterCallback`).** Forward hooks on all 44
leaf modules of the policy; the first to emit a non-finite output reports its
own inputs beside it, separating *created here* (finite in, non-finite out)
from *handed one*, and naming the layer either way. Verified by poisoning a
weight and confirming it fires and identifies the module correctly — an
instrument that silently fails to install would turn "no data" into "we
looked and found nothing".

Training is seeded and deterministic, and the crash **reproduced exactly** on
a rerun — the first time this failure has been available on demand. That is
what makes it tractable at all.

#### Root cause — found, and it was never a NaN

The instrumented reproduction caught the exact event at step 364,544:

```
! probs sum deviates by 1.073e-06 (Simplex tolerance is 1e-6)
logits: max=-0.289463 min=-14.2149
worst row 118: sum=1.0000010729
```

From the dumped tensor: logits **finite and healthy**, all 500 probabilities
non-zero in every row, and **exactly one row of 256** outside tolerance —
over the line by 7.3e-8. Re-summing those same float32 values in float64
still leaves them 8.6e-7 from 1.0, so this is inherent representation error,
not a summation-order artifact a better reduction would fix.

**The mechanism is an ordering bug in sb3-contrib**, in
`MaskableCategorical.apply_masking`:

```python
# Reinitialize with updated logits
super().__init__(logits=logits)            # validates the STALE cached probs
# self.probs may already be cached, so we must force an update
self.probs = logits_to_probs(self.logits)  # refresh — one line too late
```

`Categorical.__init__` runs PyTorch's `Simplex()` check against the `probs`
left over from the *previous, unmasked* parameterization — a float32 softmax
over all `MAX_ACTIONS` = 500 categories. That is why the tensor in the
traceback has 500 non-zero entries even though only ~40 actions were legal:
**the tensor being validated is not the distribution being constructed.**

A float32 softmax at that width sits right at the tolerance — over 200,000
random draws the worst drift measured was 9.5e-7, just *under* 1e-6. The
margin is that thin, and a 500k-step run evaluates on the order of 10⁷ rows
(256 per minibatch x 160 minibatches per update), so the tail is reached
eventually. Which run dies, and when, is a dice roll.

**This accounts for every observation** that made the crash so confusing:
no precursor pattern (nothing pathological precedes it), healthy
`explained_variance`/`value_loss`/`max_abs_param`, no non-finite value
anywhere, unpredictable crash steps, and why runs 5/7/8/10-13 passed the same
step counts untouched. It also retires the runs-3/4 "critic blowup" story as
the explanation for anything after run 4.

**Fix**: `Distribution.set_default_validate_args(False)` in `train_ppo.py`
and `eval_ppo.py`. Validation is a debugging aid (off by default in PyTorch
outside `__debug__`) and here it was rejecting a distribution that was
numerically fine. Genuine NaNs remain covered by `ParamHealthCallback`
(always on) and `--debug-nan`, both of which name the culprit instead of
surfacing `Simplex()` several updates downstream.

Regression tests: `tests/env/test_action_distribution.py` reproduces the
upstream ordering bug against the real `MaskableCategorical`, asserts it
raises with validation on, and asserts the fix both prevents it and keeps
masking correct (illegal actions still ~0).

**Worth revisiting separately**: `MAX_ACTIONS` is 500, but the action table
now peaks at **52** legal actions (median 40, p99 46, measured over 6,000
steps). Shrinking it would cut this drift at the source and narrow a policy
head that is ~90% unused — but it changes the action space and invalidates
every checkpoint, so it is its own isolated change, not a bundled one.

#### What the instrumented reproduction established

**No module hook fired.** Every one of the 44 leaf modules, `action_net`
included, produced finite output — and the distribution built from those
finite logits was still rejected. That localizes the fault to the step
between `action_net` and `MaskableCategorical`: `apply_masking` and the
softmax, which are functional ops and therefore structurally invisible to
module hooks. A second probe now wraps that step (see below).

**The anomaly in the traceback.** The rejected `probs` tensor has *non-zero*
entries in its tail (~1.4e-3, ~5.6e-4). With `MAX_ACTIONS=500` and the
sentinel `HUGE_NEG=-1e8` applied to every illegal action, those entries
should be exactly 0.

**Measured, to test the obvious explanation:** the action table never gets
large enough for this to be a many-categories effect. Over 6,000 steps the
legal-action count is min 1, median 40, p99 46, **max 52 — never above 100**.
So ~460 of 500 entries are masked at every step, only ~40 probabilities are
non-zero, and float32 summation error over 40 terms (~1e-7) cannot breach
`Simplex()`'s 1e-6 tolerance. **That hypothesis is dead.**

The hypothesis at this point was the failure mode this repo already warned
about in its own gotchas — *the `-1e8` sentinel sits near float32's working
range*, so finite-but-huge logits would leak probability onto illegal
actions. **The probe killed that too**: the captured logits were
[-14.21, -0.29]. The real answer was that the validated tensor was stale, not
that the live one was bad. See "Root cause" above.

### How to read the ablation conditions

The embedding path burned four runs before anyone checked whether the feature
was reaching the policy at all, so both lookahead runs were ablated three
ways. The two ablations mislead separately and only work together:

- **Zeroing collapses both policies to random-baseline** (1.02). That proves
  the wiring is live — not a repeat of "the feature was never connected". It
  proves nothing about whether the *content* is used, because zeroing is badly
  out-of-distribution: the policy has never seen the availability flag clear
  while a menu exists.
- **Permuting is the honest test.** It preserves the block's marginal
  distribution and destroys only the correspondence between a state and its
  own features, so it stays in-distribution. A policy conditioning on these
  values makes wrong decisions on nearly every step and degrades; one ignoring
  them does not move.

Run 13 did not move (−0.025) and run 15 moved a lot (−0.19). Reporting only
run 13 would have concluded "wired but unused" — which is what the first pass
here did conclude, before run 15 existed.

## Runs 16 and 17 — balancing the block's field scales: rejected

`_LOOKAHEAD_VALUE_SCALE = 10.0`, both seeds, nothing else changed. Since
training is deterministic, this is as clean an A/B as this project gets: the
same two seeds, one variable.

| Variant | Seed 0 | Seed 1 | Group mean | Permutation Δ |
|---|---|---|---|---|
| no lookahead (11, 12) | 1.420 | 1.420 | 1.420 | — |
| **unscaled** (13, 15) | 1.460 | **1.565** | **1.513** | −0.025, **−0.19** |
| rescaled (16, 17) | 1.470 | 1.450 | 1.460 | −0.015, +0.075 |

**The hypothesis was wrong.** Seed 1 lost 0.115 (1.565 → 1.450); seed 0
gained 0.01. Group mean fell 1.513 → 1.460. Distributions confirm it is not
a mean artifact: run 15 reached ante 5 and put 23 episodes at ante 3, while
both rescaled runs top out at ante 3 with 11-15 there.

**The ablation says why, and it is the interesting part.** Permuting the
block costs unscaled seed 1 **0.19 ante**, but costs either rescaled run
about nothing (−0.015 and *+0.075*, i.e. noise). Meanwhile the *zeroed*
condition rose from 1.02 to ~1.27. Read together: after rescaling, the block
matters **less overall**, not differently. Shrinking the loud field did not
hand its influence to the quiet ones — it made the whole block quieter, and
the network never grew compensating weights on the ratio fields within 500k
steps (the LR anneals to ~0 by then).

Default reverted to `1.0`; the knob is kept for study, and
`test_value_scale_knob_is_honored` keeps it working.

### The generalizable lesson: this reasoning has now failed twice

The prediction was: *"index 1 is 17x louder than the most decision-relevant
field purely because of its units, so quieting it will let the useful signal
through."* It is the same prediction, and the same failure, as run 10's
`embed_init_std=0.1`: *"an untrained N(0,1) embedding row is a loud random
vector drowning the real features, so shrinking it will help."*

Both times, reducing an input's magnitude reduced its **total contribution**
rather than **rebalancing** contributions within the group. Both times the
measured result was worse than leaving it alone. Two independent instances in
this codebase is enough to treat "quiet the loud input" as a suspect
inference here rather than an obvious improvement.

If field scale is genuinely worth attacking, the mechanism that actually does
what the intuition wants is `VecNormalize(norm_obs=True)`, which standardizes
every observation dimension by running statistics instead of hand-dividing
one of them. That is a different, global experiment — it would touch all 235
global dims plus every entity feature, and invalidate every checkpoint.

### Mechanism behind the imbalance (measured, still true)

First-layer weights of `global_mlp` on the trained checkpoint, mean |w| per
input column:

| | global columns (235) | lookahead columns (8) | ratio |
|---|---|---|---|
| policy | 0.0767 | 0.0849 | 1.11× |
| value | 0.0762 | 0.0796 | 1.04× |

The network weights these dims slightly *above* the average global dim, and
the two largest are #2 `best/need` (0.104) and #3 `best×hands/need` (0.090) —
exactly the two most decision-relevant. The rank order is right.

The problem is raw scale. `VecNormalize` runs with `norm_obs=False`, so
observations reach the network unnormalized, and **dim 1 is `log2(best)` with
a mean of 7.16 while every other dim is a ratio in [0,1]** (mean 0.23-0.79).
Contribution to the pre-activation is weight × value:

- dim 1: 0.082 × 7.16 ≈ **0.59**
- dim 2: 0.104 × 0.33 ≈ **0.034**

A ~17× gap, in favour of the one dimension carrying the least decision-
relevant information.

The measurement above is solid and still stands. **The inference drawn from
it did not survive contact with runs 16/17** — "the block is a magnitude
channel with seven vestigial ratios attached, so rescaling dim 1 should let
the informative dims get traction" was tested directly and came out worse.
See "Runs 16 and 17" above before reaching for this again.

Reproduce: `scripts/lookahead_ablation.py <checkpoint> 200`.

---

## Post-mortem: the joker embeddings barely move, and it isn't exposure

Measured by loading a checkpoint and diffing it against its exact
reconstructed init (SB3 calls `set_random_seed(seed)` inside `_setup_model()`
*before* building the policy, so rebuilding a `MaskablePPO` with identical
args and the same `--seed` reproduces the init tensors bit-for-bit). A row
that never received a nonzero gradient stays bit-identical to init — Adam's
moments stay 0 for it and there is no weight decay — so "rows at distance
exactly 0" is an *exact* count of catalog entries the agent never observed.

**Run 11, 500k steps** (policy-side tables; the value-side tables move ~1.5×
further but tell the same story):

| Table | Rows with gradient | …of the rows it can *ever* reach | Table drift `‖W−W₀‖_F/‖W₀‖_F` | Mean row movement |
|---|---|---|---|---|
| `joker` | 142 / 299 | **142 / 150 (95%)** | 2.19% | 0.169 |
| `consumable` | 31 / 299 | 31 / 53 (58%) | 0.79% | 0.131 |
| `shop_item` | 222 / 299 | 222 / 267 (83%) | 3.21% | 0.173 |

Run 12 (seed 1) reproduces this: 143/150, 31/53, 223/267.

**The denominator matters and an earlier version of this analysis got it
wrong.** The catalog is a *single shared ID space* built from every key in
`centers.json` — 299 keys, of which 150 are jokers (`j_`), 53 consumables
(`c_`), 32 vouchers, 32 boosters, and the rest decks/enhancements/editions.
A row in the `joker` table can only ever be touched by a `j_` key, so the
reachable denominator is 150, not 299. The earlier claim that "161 of 300
joker rows were never seen" measured against the wrong denominator and
concluded there was a data-availability problem. **There isn't one: the agent
touches 95% of the joker catalog.**

**What is actually small is the update per exposure.** A seen row moves 0.169
in L2 against a mean init row norm of 5.62 — **3% of its own length over
500k steps**. The tables are ~97% random init not because rows are unseen but
because each exposure contributes almost nothing.

**This also survives the action-table fix, which is the decisive part.**
Run 9 (400k, *before* the fix) touched **139 joker rows = 93%** of the
reachable 150. Run 11 (500k, *after*) touches 142 = 95%. The fix roughly
doubled how often the agent clears ante 1 and reaches a shop, and joker-row
coverage moved by three rows. Coverage was never the binding constraint.

Caveat on that 139: it was measured against run 9's checkpoint at the time,
with the same reconstruct-and-diff method, and is quoted here re-divided by
the corrected denominator — it is not a fresh measurement. **Pre-run-11
checkpoints no longer reconstruct**: `scripts/embed_drift.py` on run 7 or
run 9 with the seed each was trained under now reports ~140% drift and
"100% of rows seen", the signature of comparing against an unrelated random
init. Some change since then shifted how much RNG is consumed before the
embeddings are built. The reconstruction is only valid against checkpoints
from the same code revision, and the script raises on a shape mismatch but
cannot detect this subtler case — treat ~140% drift with full coverage as
"wrong init", never as a result.

Two corrections to earlier reasoning follow:

1. **"The agent dies in ante 1, so it never sees jokers, so the embeddings
   can't learn" was wrong.** Deterministic *eval* rollouts do look like that
   (run 11's policy reaches a shop in 90/100 eval episodes; run 7's in
   13/100), but training uses a *stochastic* policy that explores far more,
   and it was already seeing nearly every joker before the fix. Never infer
   training-time exposure from deterministic eval behavior.
2. **More survival will not, by itself, make the embeddings train.** The
   remaining gap is credit assignment — "did holding this joker help?" is a
   very noisy signal spread over a whole run — not data availability. Any
   future attempt at the joker path should target the signal (auxiliary
   losses, shorter-horizon joker credit, a value head conditioned on joker
   set), not more exposure.

Reproduce: `scripts/embed_drift.py <checkpoint.zip> <seed>`.

---

## Post-mortem: crash diagnosis (runs 3, 4)

All crashes so far raise `ValueError: ... probs ... to satisfy the constraint
Simplex()` — NaN in the logits reaching `MaskableCategorical`. This is the
runs-3/4 diagnosis; run 9's crash matches the exception but *not* the
precursor pattern, so don't assume a shared root cause.

Run 3 looked like policy divergence (`approx_kl` 0.029 rising,
`clip_fraction` 0.33). Run 4 had **healthy** policy stats (`approx_kl` 0.0042
*falling*, `clip_fraction` 0.068) and crashed 4k steps earlier. That rules
out policy divergence.

Shared signature: `value_loss` ~10× run 2 and curving vertical,
`explained_variance` sliding 0.9 → 0.40, both crashing at `ep_rew_mean` ≈ 0.17.

**Mechanism**: episodes lengthen → returns grow → value targets grow → MSE
grows quadratically → gradients through the *shared* features extractor grow
→ encoder weights inflate → float32 overflow → `inf` reaches policy logits →
softmax gives NaN. Masked categoricals are extra sensitive because the `-1e8`
mask sentinel already sits near float32's working range.

The policy was collateral damage from a critic blowup through a shared
encoder. The reward ×10 (100× MSE, `vf_coef=0.5`) was the proximate trigger.

**Root cause underneath it**: the critic can't predict returns from the
observation. `explained_variance` decay is that limit showing up as a crash.

**Early warning signs** (both crashes were visible ~200k steps ahead):
`clip_fraction` sustained above ~0.3; `approx_kl` rising monotonically;
`value_loss` curving vertical; `explained_variance` declining while
`value_loss` climbs.

---

## Throughput: where the wall-clock goes

> **Superseded in part.** The decomposition below is what motivated the
> `hand_eval.group_by_rank` optimization; see "Optimizing the hot path"
> underneath it for the post-fix numbers (enumeration 4.26 → 2.94 ms/step,
> env 220 → 309 steps/sec).

Training fps fell from run 5's 524 to runs 11/12's ~171. Decomposed by direct
measurement (`uv run scripts/bench_step.py`, single-threaded torch, this
machine; ±10% between invocations):

| Component | ms/step | Share of the ~5.7 ms budget |
|---|---|---|
| **Action-table enumeration** | **4.26** | **75%** |
| `BalatroExtractor` forward ×2 (rollout, unshared pi/vf) | 0.63 | 11% |
| `BalatroExtractor` fwd+bwd amortized into `train()` | 0.43 | 8% |
| Engine `step()` + observation encoding | 0.35 | 6% |

The model predicts **177 steps/sec** from env + extractor alone against runs
11/12's observed 171-172 fps, so those four terms account for essentially the
entire budget — there is no missing overhead to hunt for.

Three things follow:

1. **The environment is the whole story; the network is nearly free.** Raw
   `env.step()` under a random legal action runs at ~220 steps/sec, and
   **92% of that wall-clock is `_enumerate_actions`** — the engine transition
   and observation encoding together are 0.35 ms. The `BalatroExtractor`
   costs 10× the default extractor and that buys back only ~19%. (SB3's
   `CombinedExtractor` over an all-`Box` Dict space is a parameter-free
   flatten+concat — it has *no* weights, so there is nothing to backprop
   through either; all the default config's capacity lives in the policy
   MLP.) 82% of steps are in `SELECTING_HAND`, which is exactly the phase
   enumeration is expensive in.
2. **The 524 → 171 drop is bought, not regressed.** Run 5 had a cheap
   random-subsample action table and a parameter-free extractor. Runs 11/12
   pay ~4.3 ms/step to rank candidate plays, and that is precisely the change
   that moved eval mean ante 1.26 → 1.42. ~50 minutes per 500k-step run
   instead of ~16 is the price of the only confirmed gain so far.
3. **fps carries large exogenous variance — don't read it as a code signal.**
   Runs 7 and 8 are the same code and differ by 1.7× (314 vs 187 fps). The
   logs show no overlap between training runs, and `max_count` last changed
   before run 5, so this is background load on the machine, not the
   configuration. Only treat an fps change as real if it reproduces or a
   direct benchmark confirms it.

If throughput becomes the constraint, the target is unambiguous: cheapen or
cache `_enumerate_actions`, or run `SubprocVecEnv` workers so the enumeration
parallelizes. Optimizing the network would recover at most ~16%.

---

## Optimizing the hot path: `hand_eval.group_by_rank`

The section above said the target was unambiguous. Profiling inside
`_enumerate_actions` (`cProfile`; py-spy is not installed and has never
attached in this sandbox) found it:

| Function | calls | tottime | cumtime |
|---|---|---|---|
| **`get_x_same`** | 533,668 | **2.12 s (21%)** | **3.28 s (33%)** |
| `evaluate_hand` | 133,417 | 1.06 s | 8.34 s (84%) |
| `Card.is_suit` | 826,900 | 0.63 s | 1.17 s |
| enum `__get__` | 2,639,721 | 0.56 s | 0.81 s |

`evaluate_poker_hand` calls `get_x_same` **four times** per hand (sizes 5, 4,
3, 2), and each call rebuilt the same rank grouping from scratch with an
O(n²) scan. Fix: group once (`group_by_rank`), filter four times.

| | Before | After |
|---|---|---|
| `_enumerate_actions` | 4.26 ms/step | **2.94 ms/step** |
| Env throughput (single, no network) | 220 steps/sec | **309 steps/sec** |

A second pass mattered nearly as much as the first: the initial rewrite still
walked `range(14, 0, -1)` per size, i.e. 56 dict probes for a hand holding at
most 5 distinct ranks. Sorting the handful of real groups once instead took
enumeration from 3.52 → 2.94 ms.

### Doing this safely against the bit-exactness rule

`hand_eval.py` is engine code, where `CLAUDE.md` requires `jackdaw validate`
— which needs a live BalatroBot server. Three offline steps stood in for it:

1. **Derived** the equivalence. The old code scanned `i` from `len-1` down to
   0 overwriting `vals[card_id]`, so the surviving group for a rank was the
   one built at the *smallest* matching `i` — which is just the matching
   cards in ascending index order.
2. **Differential-tested** the claim: 0 mismatches over 80,000 comparisons.
3. **Generated a golden fixture first** —
   `tests/fixtures/hand_eval_refactor_golden.json`, 4,011 randomized hands
   biased toward what a grouping rewrite breaks (empty hands, five of a kind,
   flush houses, wheel straights, wild/stone), recording detected hand,
   scoring-card *positions*, and full `poker_hands` decomposition including
   group order. `tests/engine/test_hand_eval_refactor_golden.py`.

The Lua oracle test says "matches the source"; the golden says "unchanged by
the refactor". **`jackdaw validate` is still the outstanding gate.**

**Next targets, not taken:** `Card.is_suit` (~8%) and enum attribute access
(~7%). Both have far more semantic surface than the grouping change (debuff,
wild, smeared, stone) for a fraction of the payoff. Do them one at a time,
each with its own validate cycle — stacking unvalidated engine changes makes
a bit-exactness failure impossible to bisect.

### The benchmarks did not cover any of this

`tests/benchmarks/test_env_bench.py::test_env_steps_per_second` drives
`DirectAdapter` directly, so it never builds an action table — it asserted
">500 steps/sec" while real training ran at 220. The most expensive component
in the pipeline had no benchmark at all.

Worse, the benchmark suite was **flaky on unchanged code**: five consecutive
runs gave 5-pass, 2-fail, 1-fail, 1-fail, 1-fail. `_collect_game_states` drew
from the unseeded global `random`, so each run walked a different action
sequence, and any walk that opened a booster pack containing a targeted
consumable handed the engine a `PickPackCard` with no targets
(`IllegalActionError: c_sun requires between 1 and 3 target card(s)`). This
briefly produced a false signal — stashing the engine change made the suite
pass, which looked like evidence the change was at fault. It wasn't; five
repeat runs showed it was luck. Same class as known issue #11.

Now seeded and tolerant of unfillable marker actions (6/6 deterministic), with
two new benchmarks covering the real path: `test_gym_env_steps_per_second`
(>120/sec) and `test_action_table_enumeration_latency` (<8 ms mean).

---

## Parallel rollout collection (`--vec-env subproc`)

`n_envs > 1` had never actually been run in this project — defect #3 (every
worker replaying the same seed sequence) was fixed defensively, never
exercised. It holds: **4/4 distinct worker observations** under both
`DummyVecEnv` and `SubprocVecEnv`.

Raw env throughput, no network (`scratchpad/bench_subproc.py`):

| Workers | steps/sec | vs 1 |
|---|---|---|
| 1 (dummy) | 284 | 1.00x |
| 4 | 699 | 2.46x |
| 6 | 914 | 3.21x |
| 8 | 1089 | 3.83x |

End-to-end training fps, 120k steps each, back-to-back on an idle machine:

| Config | fps | vs baseline | 500k run |
|---|---|---|---|
| dummy, 1 | 160 | 1.00x | ~52 min |
| subproc, 4 | 355 | 2.22x | ~23 min |
| subproc, 6 | 428 | 2.68x | ~20 min |
| **subproc, 8** | **485** | **3.03x** | **~17 min** |

End-to-end gain (3.0x) is below raw env scaling (3.8x) because the policy
forward/backward runs in the main process and is not parallelized — it is
the serial fraction. 8 workers still beat 6 on a 6C/12T box, and 4096
divides evenly by 8, so there is no rollout rounding.

### The trap: `n_steps` is per environment

SB3's rollout buffer holds `n_steps * n_envs` transitions. Raising
`--n-envs` with `n_steps` fixed multiplies the rollout — with 8 workers,
4096 becomes 32,768, changing the minibatch count per update and the
staleness of the oldest data in it. That is a hyperparameter change
disguised as a speedup, and it would have made every parallel run
incomparable to runs 11-17 for reasons unrelated to parallelism.

`--rollout-steps` (total, default 4096) now derives
`n_steps = rollout_steps // n_envs`, so worker count buys wall-clock and
nothing else. `tests/env/test_vec_env_setup.py` pins the invariant.

### Holding the rollout constant is necessary, not sufficient

A parallel run is still **not** bit-identical to a serial one:

- each worker contributes a shorter contiguous trajectory (512 steps at
  `n_envs=8`, versus 4096), so GAE bootstrapping at rollout boundaries
  differs;
- batch composition changes — 8 concurrent games instead of one sequential
  stream. That is usually *better* for PPO, but it is a change.

So the speedup got its own validation: **run 18 replicates run 15's exact
configuration (seed 1, `--lookahead`) with `--n-envs 8`.**

| Run | Collection | Mean ante | Past ante 1 | Distribution | Wall clock |
|---|---|---|---|---|---|
| 15 | serial, 1 env | 1.565 | 85/200 | `{1:115, 2:60, 3:23, 4:1, 5:1}` | ~59 min |
| 18 | **subproc, 8 envs** | **1.535** | **89/200** | `{1:111, 2:71, 3:18}` | **16.6 min** |

**Parallel collection does not shift results.** The means differ by 0.03
against a 0.13 noise floor, and run 18 actually clears ante 1 slightly *more*
often (89 vs 85). Run 15's higher max ante (5 vs 3) is two single episodes in
the tail, not a distributional difference. Parallel runs are comparable to
serial ones, and `--vec-env subproc --n-envs 8` is the default worth using.

One caveat for anyone reading the ablation column: run 18's policy conditions
on the lookahead block only weakly (permuted costs it 0.045, versus run 15's
0.19) despite scoring nearly the same. Across all five lookahead runs, strong
conditioning has shown up exactly once. Whatever makes a run learn to exploit
these features, it is not something any change so far has controlled.


---

## Runs 19 and 20 — 1M steps: the ceiling finally moves

Runs 18's configuration at `--total-timesteps 1000000`, both seeds, 8
parallel workers. No crashes. 41 and 36 minutes respectively — the same work
would have taken ~3.5 hours before this session's throughput fixes.

| Run | Steps | Seed | Mean ante | Past ante 1 | **Ante 3+** | Max | Distribution |
|---|---|---|---|---|---|---|---|
| 13 | 500k | 0 | 1.460 | 75/200 | 14 | 4 | `{1:125, 2:61, 3:11, 4:3}` |
| 15 | 500k | 1 | 1.565 | 85/200 | 25 | 5 | `{1:115, 2:60, 3:23, 4:1, 5:1}` |
| 18 | 500k | 1 | 1.535 | 89/200 | 18 | 3 | `{1:111, 2:71, 3:18}` |
| **19** | **1M** | 1 | **1.620** | 97/200 | 24 | 4 | `{1:103, 2:73, 3:21, 4:3}` |
| **20** | **1M** | 0 | **1.865** | **102/200** | **48** | **6** | `{1:98, 2:54, 3:31, 4:12, 5:4, 6:1}` |

Group means: **1.743 at 1M against 1.520 at 500k, +0.22** on a 0.13 noise
floor, and both 1M runs beat all three 500k runs.

**The ante 3+ column is the real story.** Every earlier change moved the
ante-1 clear rate and left the ceiling alone — the agent got better at the
first blind and no better at going deep. Run 20 puts **48 episodes at ante 3
or beyond** against 14-25 before, reaches ante 6, and is the first run since
run 5 to see ante 5+ more than once. That is the ceiling moving, not just
the floor.

Seed ordering also flipped: seed 0 was the weaker seed at 500k (1.460 vs
1.565) and the stronger at 1M (1.865 vs 1.620). Treat per-seed rankings as
noise, not character.

### This is confounded with the learning-rate schedule

**A 1M run is not "500k plus more."** SB3 schedules `learning_rate` and
`clip_range` on fraction of *declared* `total_timesteps`, so runs 19/20 had a
higher LR than runs 15/18 at every matching step. Run 9 was previously
misread as evidence about step count for exactly this reason.

So what these runs answer is *"is a 1M run with its natural schedule better
than a 500k run with its natural schedule?"* — practically the question worth
asking, and the answer is a clear yes. What they do **not** isolate is step
count alone. Anyone wanting that needs an absolute-step LR schedule, which is
its own change requiring its own validation (§7).

### Ablation: the best run is also the one using the features

| Run | intact | permuted | Δ | zeroed |
|---|---|---|---|---|
| 19 (1M, seed 1) | 1.620 | 1.560 | −0.06 | 1.275 |
| **20 (1M, seed 0)** | **1.865** | **1.645** | **−0.22** | 1.205 |

Run 20 conditions on the lookahead block more strongly than any run so far
(run 15's −0.19 was the previous high). Note its permuted score (1.645) still
beats every 500k run, so its advantage is partly extra training and partly
learned feature use — the two are separable here and both are real.

Across seven lookahead runs, strong conditioning has now appeared in the two
best runs (15 and 20) and weakly elsewhere. That is suggestive, not
established — run 18 scored 1.535 with a −0.045 ablation, so it is not a
clean rule.

### Found while doing this: the Simplex guard was missing from the ablation script

`scripts/lookahead_ablation.py` never called
`disable_distribution_validation()` — the fix had gone into `train_ppo.py`
and `eval_ppo.py` only. It ran three 200-episode sweeps against run 20 and
died on the same float32 tolerance failure this session root-caused. Fixed.
Worth noting the shape of the mistake: the diagnostic that proved the bug was
itself vulnerable to it, and only a checkpoint that happened to sit near the
tolerance exposed the gap.


---

## Where the remaining wall-clock goes (post-parallelism)

With 8 workers the picture inverts — the env is no longer the bottleneck:

```
wall clock per step      2.06 ms
env work per step        3.24 ms  -> /8 workers = 0.40 ms of critical path
main-process remainder   1.66 ms  (80% of critical path)
```

Which is also why 6 -> 8 workers bought only 13%. More workers now buy very
little; the serial main process gates everything.

**Torch thread contention: hypothesis tested, rejected.** 8 workers plus
torch's default 6 intra-op threads on a 6C/12T box looks like textbook
oversubscription, so clamping torch should help. Measured at `--n-envs 8`,
120k steps each:

| `OMP_NUM_THREADS` | fps |
|---|---|
| 1 | 451 |
| 2 | 493 |
| 4 | **524** |
| 6 (torch default) | 506 |

Clamping to 1 costs **11%**. The network is not thread-starved and the
default is already near-optimal; 4 threads is the peak and buys 3.5%.
Exposed as `--torch-threads` (default: leave torch alone) since the optimum
is machine-specific, but do not expect much from it.

**On moving to a GPU.** Not worth it for this workload, and the parallel
numbers do not change that despite the main process now dominating:

- The threading result shows the network is not compute-bound — throwing
  more parallelism at it barely moves throughput, so a faster device will not
  either.
- The model is tiny (256-dim features, small MLPs), and rollout inference
  runs at **batch 8**. Kernel-launch and host-device transfer overhead at
  that size routinely exceeds the compute, so a GPU can be *slower* for the
  rollout half.
- Of the 1.66 ms main-process budget, ~1.06 ms is the features extractor
  (`scripts/bench_step.py`); the rest is SB3 buffer bookkeeping,
  `VecNormalize` and IPC deserialization — none of which a GPU touches.

Linux is still mildly preferable (`fork` instead of Windows `spawn` for
worker startup, generally cheaper IPC), but that is a constant factor on
setup, not a multiplier on training.

**The real remaining lever is `MAX_ACTIONS`.** It is 500 while the action
table peaks at 52 (median 40). That inflates the policy head 8x, every mask
allocation, and every softmax — all of it in the serial main process — and it
is the same width that puts the float32 softmax at the `Simplex()` tolerance
in the first place. Shrinking it to ~64 attacks throughput and that
numerical fragility together. It changes the action space, so it invalidates
existing checkpoints and needs its own baseline pair.
