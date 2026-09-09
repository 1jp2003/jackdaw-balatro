# RL Plan — Jackdaw Balatro

Working plan for training an agent to play Balatro via the Jackdaw simulator.
Written after analysis of `observation.py`, `gymnasium_wrapper.py`,
`action_space.py`, `balatro_env.py`, `balatro_spec.py`, `game_spec.py`, and four
training runs.

---

## 1. Goal and scope

**Immediate target:** an agent that reliably clears ante 4–6 on Red Deck /
White Stake (`back_keys=["b_red"]`, `stakes=[1]`), with occasional wins.

**Not the target (yet):** consistent ante-8 wins, or all decks and stakes.
Those are follow-ons once the pipeline works.

**Secondary goal:** a documented, reproducible process suitable for a portfolio
writeup. This means experiment discipline matters — see §8.

**Constraint:** any change to the observation or action space must remain
computable from what `BridgeAdapter` can see, so the trained policy can be
validated against live Balatro through BalatroBot.

### Success bar

A scripted heuristic using exact one-step lookahead is expected to reach ante
3–4 unaided. **The RL agent is not interesting until it beats that heuristic.**
Build the heuristic first and treat its number as the project baseline.

---

## 2. Hardware and compute budget

| Resource | Spec | Notes |
|---|---|---|
| CPU | Intel i5-12400F, 6C/12T | **The binding constraint.** |
| GPU | Radeon RX 6750 XT, 12 GB | gfx1031, not officially ROCm-supported |
| RAM | 32 GB | Not a limit |
| OS | Train on Linux; validate on Windows | LiveBackend/BalatroBot is Windows-only |

Measured throughput: **~417 steps/sec** single-env on Windows CPU
(500k steps in ~20 min). With 6 `SubprocVecEnv` workers on Linux, expect
**~2,000–2,500 steps/sec**, i.e. ~9M steps/hour. 10⁸ steps is an overnight run.

**Do not rent cloud compute.** The arithmetic doesn't justify it. If throughput
later becomes the bottleneck, rent **CPU** (e.g. Hetzner CCX/AX, ~€40–60/mo for
16–32 cores), not GPU — a GPU instance would sit idle behind the same Python
simulator bottleneck.

**ROCm setup (optional):** Ubuntu 22.04/24.04, PyTorch ROCm wheel, and
`HSA_OVERRIDE_GFX_VERSION=10.3.0` to present gfx1031 as gfx1030. Worth doing
because it's free upside, but the networks here are small MLPs and embeddings —
the GPU is close to irrelevant. Do not treat it as a blocker.

---

## 3. Current state

Four MaskablePPO runs completed. Best result: **mean ante ~1.08, ~1.9 blinds
beaten, `ep_len_mean` ~20, win rate 0.**

The agent dies inside ante 1. Small blind at ante 1 needs 300 chips; a
near-random 5-card play from ~218 options scores 20–60, so four hands reaches
~150. It cannot clear the first blind by sampling hands.

### Run history

| Run | Change | Result |
|---|---|---|
| 2 | Baseline (`ent_coef=0.05`, lr 1e-4) | ante 1.05, stable, 500k completed |
| 3 | Reward ×10, flat blind reward, lr 3e-4, `ent_coef` 5e-4 (schedule bug) | ante 1.07, **crashed 364k** |
| 4 | + `target_kl`, lr anneal, working `ent_coef` schedule | ante 1.08, **crashed 360k** |
| 5 | Same hyperparameters as run 4, no reward/hparam changes — only defects 3/5/6/7 fixed (Phase 0) | 500k completed, **no crash**; see below |
| 6 | `--extractor balatro`: embeddings + masked-mean pooling on **all 5** entity types (Phase 1, first attempt) | 500k completed, no crash, **ante 1.00 — regression**; see below |
| 7 | `--extractor balatro`, pooling fix: only pool joker/consumable/shop_item, flatten hand_card/pack_card | 500k completed, no crash, **ante 1.26 — parity with run 5**, different distribution shape; see below |
| 8 | Same as run 7, `--seed 1` (second seed, isolating run-to-run noise) | 500k completed, no crash, **ante 1.13**; see below |
| 9 | Same as run 7 (`--seed 0`), `--total-timesteps 1000000` — intended to test the embedding cold-start hypothesis | **crashed at step 409,600** — same NaN-in-logits failure as runs 3/4. Not a clean step-count test — see below |

### Run 5 — first eval-seed-based number (post Phase 0)

Runs 1-4's "ante ~1.08" was a *rollout* statistic (defect #7: sparse-mode
metrics were broken anyway, and even dense-mode rollout averages are exactly
what CLAUDE.md's Conventions warn not to trust). Run 5 is the first run with
a real frozen-eval-seed-set number, via the eval harness this project didn't
have before (`scripts/eval_ppo.py`, built the same session after finding and
fixing defect #14's determinism bug — see below):

`uv run scripts/eval_ppo.py --model runs/balatro_ppo/balatro_ppo --episodes 200`
(deterministic policy, all 200 `EVAL_SEEDS`, `results/ppo_run5.json`):

| Agent | Mean ante | Max ante | Win rate |
|---|---|---|---|
| Random | 1.00 | — | 0% |
| **Heuristic** (`HeuristicAgent`, exact lookahead) | **2.75** | 7 | 0% |
| **PPO run 5** (500k steps, post Phase 0) | **1.26** | 6 | 0% |

**PPO does not beat the heuristic** — not close. §5.3/§6's framing ("the RL
agent is not interesting until it beats the heuristic," Phase 1 exit is
"PPO approaches the heuristic") is not yet satisfied. Training did complete
cleanly for the first time (no NaN crash — defects 3/5/6/7 plausibly helped
stability, though this is one run, not a controlled ablation), and
`rollout/ep_rew_mean` climbed to 0.26 / `explained_variance` to 0.54 by the
end (both logged correctly now that defects 6/7 are fixed), so the policy is
learning *something* — just not enough to compete with one-step exact
lookahead. Consistent with §5.4/Phase 1's diagnosis: the bottleneck is
representation and action space, not infra.

Also found while building this eval, already fixed: defect #14
(`BalatroGymnasiumEnv._rng` wasn't reseeded on a string-only reset, making
eval on a "frozen" seed non-reproducible whenever action-table subsampling
triggered) and identified, not yet fixed: defect #15 (deterministic policy
stalls on cosmetic no-op action loops in 25% of eval episodes) — both
detailed in §4.

### Run 6 — Phase 1 embeddings, first attempt: regression, diagnosed and fixed

Same 500k timesteps, same hyperparameters as run 5, only change:
`--extractor balatro` (§5.1/5.2 — catalog-ID embeddings + masked-mean
pooling on all 5 entity types). No crash, `explained_variance` reached
**0.59** (better than run 5's 0.54) and `value_loss`/`loss` decreased
smoothly — every training-time health signal in §8 looked *good*.

But `scripts/eval_ppo.py --episodes 200` (`results/ppo_run6.json`): **mean
ante 1.00, max ante 2** — worse than run 5 (1.26, max 6), i.e. down to
random-baseline level. Ante distribution: 199/200 episodes never left ante
1 (run 5: 169/200). This is the exact case §8's "how to judge a run"
calibration note now describes: healthy training curves + eval regression
= an architecture problem, not a training-dynamics or step-count problem.

**Root cause**: pooling was applied uniformly to all 5 entity types,
including `hand_card`. Masked-mean pooling collapses N individual cards in
specific slots into one averaged vector — destroying exactly the per-slot
identity `PlayHand`/`Discard`'s `card_target` needs to select *which*
cards to play, which is the single dominant factor in even clearing ante 1
(§3's own diagnosis: "near-random 5-card play... cannot clear the first
blind"). Pooling only makes sense once something downstream consumes a
per-entity stream via attention/pointer lookup (§5.4, "later") instead of
today's flat `Discrete(500)` linear layer reading one pooled vector.

Compounding methodology issue: run 6 bundled two representation changes
(embeddings *and* pooling) into one architecture tested as a unit —
violating this project's own "isolate one variable" discipline (§8) just
applied to architecture instead of reward/hyperparameters. A negative
result couldn't tell us which piece was responsible.

**Fixed** (`jackdaw/env/feature_extractor.py::BalatroExtractor`): pooling
now applies only to `joker`/`consumable`/`shop_item` (catalog entity
types); `hand_card`/`pack_card` are flattened (per-slot identity
preserved, matching the default extractor's behavior for those two types,
just still routed through a per-entity-type MLP). Regression test added
(`tests/env/test_feature_extractor.py::test_hand_card_order_is_not_pooled_away`
— swaps two hand-card slots and asserts the extractor's output changes)
so this can't silently regress again. Run 7 (table above) is this fix,
training now — when it lands, compare `results/ppo_run7.json`'s mean ante
against run 5's 1.26: if it's clearly higher, embeddings-for-catalog-types
+ positional-hand-cards is validated and Phase 1's exit criterion
("PPO approaches the heuristic") is closer; if it's still ~1.0-1.3, the
next hypothesis to test is embedding cold-start (jokers are rarely seen
before ante 1 is lost, so the embedding table gets few gradient updates in
500k steps) — try a longer run or a second seed before concluding the
representation fix itself doesn't help.

### Run 7 — pooling fix result: parity, not (yet) a win

Same 500k timesteps/hyperparameters, `--extractor balatro` with the
hand_card/pack_card pooling fix from run 6. No crash;
`explained_variance` 0.52, `ep_rew_mean` 0.28 — healthy, similar to run 5.

`results/ppo_run7.json`: **mean ante 1.26** — identical to run 5's 1.26 to
two decimal places. Max ante 4 (run 5: 6). Ante distribution:
`{1: 162, 2: 25, 3: 12, 4: 1}` vs run 5's `{1: 169, 2: 15, 3: 12, 4: 3, 6: 1}`
— run 7 clears ante 1 slightly *more often* (38/200 vs 31/200 episodes)
but with a lower ceiling (no run reached ante 5+). Net: a different
distribution shape landing on the same mean — a wash, not a regression
(the fix did resolve run 6's collapse to random-baseline) and not a proven
win either. Squarely the "still ~1.0–1.3" outcome anticipated above.

**Interpretation**: this is one run each at 500k steps — not enough to
distinguish "the representation fix doesn't help at this scale" from "it
needs more steps to pay off" from "it's genuinely neutral and the next
bottleneck is elsewhere (action space, §5.4)." Runs 3 vs 4 already
demonstrated this project's single-run comparisons carry real noise (very
different `approx_kl`/`clip_fraction` trajectories, near-identical final
performance). **Not run yet, worth doing before drawing a firm
conclusion**: a longer run (e.g. 1-2M steps, isolating step count as the
only changed variable — §8 rule 1) or a second seed at 500k to see if the
"same mean, different shape" pattern holds. Whoever picks this up next
should treat Phase 1's embedding item as *implemented and not regressing*,
not yet as *validated* — the honest state is "inconclusive after one
comparison run in each direction."

### Run 8 — second seed: noise band established

Same as run 7, `--seed 1` only (isolates run-to-run noise from the run
5-vs-7 comparison — §8 rule 1). No crash, healthy curves
(`explained_variance` 0.54, no NaN). `results/ppo_run8.json`: **mean ante
1.13**, max ante 4.

All four runs together (ante distributions, `results/ppo_run{5,6,7,8}.json`):

| Run | Extractor | Seed | Mean ante | Distribution |
|---|---|---|---|---|
| 5 | default | 0 | 1.265 | `{1:169, 2:15, 3:12, 4:3, 6:1}` |
| 6 | balatro, all pooled (buggy) | 0 | 1.005 | `{1:199, 2:1}` |
| 7 | balatro, fixed | 0 | 1.260 | `{1:162, 2:25, 3:12, 4:1}` |
| 8 | balatro, fixed | 1 | 1.130 | `{1:179, 2:17, 3:3, 4:1}` |

**Reading this**: run 6 is a clear, unambiguous outlier (1/200 episodes
past ante 1 — the pooling bug, not noise). Runs 5/7/8 all sit in a tight
1.13-1.27 band. Critically, **the seed-to-seed spread within the identical
`balatro` config (runs 7 vs 8: 1.26 vs 1.13, a 0.13 gap) is about as large
as the gap between either of them and run 5's single default-extractor
point (1.265)** — meaning at 500k steps and this sample size, runs 5, 7,
and 8 are statistically indistinguishable from each other. This is a
genuine, informative result, not a non-result: **the fixed `balatro`
extractor is confirmed not broken (unlike run 6), but shows no detectable
benefit over the default extractor at this training budget.**

This does not resolve whether embeddings would help given more training
(the cold-start hypothesis is still untested) or whether the real
bottleneck has moved elsewhere (§5.4's action-space redesign). The
originally-planned next step — a longer run (1-2M steps) — now tests that
specific question against a known noise floor (~0.13 ante) rather than a
guess: if a 1-2M-step `balatro` run clears that band by a comfortable
margin (say, >1.5), that's a real signal the representation fix pays off
with more data. If it's still in the 1.1-1.3 band, that's reasonably
strong evidence — two seeds at 500k plus one long run all landing in the
same place — that representation isn't the bottleneck right now, and
Phase 2's action-space work (§5.4) is more likely where the next real gain
is.

### Run 9 — the crash came back, and the run wasn't a clean test

Intended as `--total-timesteps 1000000`, `--seed 0`, otherwise identical to
run 7, to test the embedding cold-start hypothesis by extending the exact
same trajectory further. **Crashed at step 409,600** with the identical
failure mode as runs 3/4 (`ValueError: ... probs ... Simplex()` — NaN in
the logits reaching `MaskableCategorical`, see "Crash diagnosis" below).
Checkpointing (Phase 0's defect-#3-adjacent fix, `--checkpoint-freq`)
worked exactly as intended here — a checkpoint at 400,000 steps survived,
so this run lost ~10k steps, not the 360-364k runs 3/4 lost with no
checkpointing at all.

**Two things make this a materially different finding than "runs 3/4's
crash, again":**

1. **The precursor pattern is absent.** Runs 3/4 showed `explained_variance`
   sliding 0.9→0.40 and `value_loss` curving vertical for ~200k steps
   before crashing (see "Crash diagnosis" below) — a gradual blowup with
   warning. Run 9's `explained_variance` (~0.55), `value_loss` (~0.7-0.9),
   `approx_kl` (~0.013), and `clip_fraction` (~0.16) were all flat and
   healthy for the 60k+ steps immediately before the crash. This looks
   like a sudden, precursor-free event, not a gradual drift — a different
   failure signature than the one already diagnosed and (partially)
   attributed to `share_features_extractor=True`. Run 9 used
   `share_features_extractor=False` (the known-issue-#4 fix) and still
   crashed, so that fix alone does not close this failure mode.
2. **Run 9 wasn't actually isolating step count.** `learning_rate=lambda p:
   3e-4 * p` and `clip_range=0.15` are both scheduled on SB3's internal
   `p` = fraction of *declared* `total_timesteps` remaining — not on
   absolute step count. Run 7 (`total_timesteps=500_000`) had decayed to
   ~18% of its starting LR by step 409,600 (82% of the way through its
   schedule); run 9 (`total_timesteps=1_000_000`) was only 41% of the way
   through its schedule at that same step count, i.e. its effective
   learning rate was **more than 3× higher** than run 7's would have been
   at the identical point. A separate, smaller instance of the same bug
   was also found and fixed: `EntCoefSchedule` was hardcoded to
   `total_timesteps=500_000` regardless of `--total-timesteps`, so for
   run 9 it hit its floor at step 500k and stayed there instead of
   scheduling over the declared 1M (now fixed — uses `args.total_timesteps`).
   A held-longer, higher learning rate interacting with a critic that
   still can't perfectly predict returns is exactly the mechanism runs
   3/4's diagnosis already implicates (§ below) — so this may not be new
   evidence about the *embedding architecture* being unstable so much as
   confirmation that *this reward scale / learning-rate combination* is
   still marginal, now triggered by an inadvertently-extended high-LR
   window rather than by the reward ×10 change that triggered it in runs
   3/4.

**Fixed**: the checkpoint-collision bug this investigation surfaced (see
"Existing code changes" note below) and the `EntCoefSchedule` hardcoding.
**Not fixed, deliberately left as a decision point**: the LR/clip_range
schedules are still fraction-of-declared-`total_timesteps`-based, which is
SB3's own convention (`get_schedule_fn`) and changing it to an
absolute-step-based schedule would itself be a hyperparameter/schedule
design change needing its own isolated validation (§8 rule 1) — not
something to do silently while investigating a crash. **Any future
`--total-timesteps` value other than 500,000 should be read as "also has a
different effective LR/clip_range trajectory than prior runs," not as a
clean single-variable extension**, until/unless that's addressed.

**Checkpoint-collision bug found and fixed**: `CheckpointCallback`'s
`save_path` was a flat `<log_dir>/checkpoints/` folder shared by *every*
invocation using the same `--log-dir` (the default for every run so far).
Two separate runs reaching the same step count would silently overwrite
each other's checkpoint file — run 9's 400k-step recovery checkpoint
happened to be intact only because no later run had reached that filename
yet, not because collisions can't happen. Fixed: `scripts/train_ppo.py`
now writes to a timestamped subfolder,
`<log_dir>/checkpoints/<YYYYMMDD_HHMMSS>/`, unique per invocation.

### Crash diagnosis (runs 3, 4 — see "Run 9" above for a third, differently-shaped crash)

All three crashes so far: `ValueError: ... probs ... to satisfy the
constraint Simplex()`. NaN in the logits reaching `MaskableCategorical`.
This subsection is the original runs-3/4 diagnosis; run 9's crash (above)
matches the same exception but *not* the precursor pattern documented
here — read both before assuming they share a root cause.

Run 3 looked like policy divergence (`approx_kl` 0.029 rising,
`clip_fraction` 0.33). Run 4 had **healthy** policy stats (`approx_kl` 0.0042
*falling*, `clip_fraction` 0.068) and crashed 4k steps earlier. That rules out
policy divergence.

Shared signature in both: `value_loss` ~10× run 2 and curving vertical,
`explained_variance` sliding 0.9 → 0.40, both crashing at `ep_rew_mean` ≈ 0.17.

**Mechanism:** episodes lengthen → returns grow → value targets grow → MSE
grows quadratically → gradients through the *shared* features extractor grow →
encoder weights inflate → float32 overflow → `inf` reaches policy logits →
softmax gives NaN. Masked categoricals are extra sensitive because the
`-1e8` mask sentinel already sits near float32's working range.

The policy was collateral damage from a critic blowup through a shared encoder.
The reward ×10 (100× MSE, `vf_coef=0.5`) was the proximate trigger.

**Root cause underneath it:** the critic cannot predict returns from the current
observation, because joker identity — the strongest predictor of future scoring
in Balatro — is compressed into one normalized float. `explained_variance`
decay is that limit showing up as a training crash.

### Key finding

Runs 3 and 4 had very different policy dynamics and near-identical task
performance (1.75 vs 1.86 blinds beaten). **Hyperparameter tuning is exhausted.**
The bottleneck is representation and action space.

---

## 4. Known defects

Ordered by impact. File references are to the unmodified fork.

### Critical

1. **Joker identity destroyed.** `observation.py:420` —
   `center_key_id(card.center_key) / NUM_CENTER_KEYS`. All ~299 centers
   compressed to one float in [0,1]. Tells the network joker #47 ≈ #48, and the
   ordering is arbitrary. Same bug for boss blind key at `observation.py:745`.
   *Highest-value fix in the project.*

2. **Legal actions randomly vanish.** `gymnasium_wrapper.py:331-332` — global
   `_subsample` when `len(actions) > MAX_ACTIONS`. Playing phase with an 8-card
   hand: 218 PlayHand + 218 Discard + swaps/sorts/sells/consumables routinely
   exceeds 500. Randomly deletes actions across all types, so the winning play
   may not be offered, and identical states give different menus.

3. **Parallel workers play identical games.** `balatro_env.py` —
   `seed = f"{self._seed_prefix}_{self._episode_count}"`. With one
   `seed_prefix` across `SubprocVecEnv` workers, all workers generate the same
   seed sequence. Batch diversity drops to 1 while cost stays N×. **Fails
   silently.** Fix: `f"{prefix}_w{worker_id}"`.

4. **Shared features extractor.** `share_features_extractor` defaults True.
   Verified present in sb3-contrib 2.9.0 `MaskableMultiInputActorCriticPolicy`
   (`policies.py:469`). Set False so a critic blowup can't corrupt the policy.

### Important

5. **Entity truncation.** `balatro_spec.py` — `hand_card` `max_count=8`,
   `joker` `max_count=5`. Hand size exceeds 8 in normal play (Juggler, Turtle
   Bean, vouchers); jokers exceed 5 with Negative editions. `_build_obs`
   truncates, but `card_mask` still marks those cards legal — **the agent can
   act on cards it never observed.** Raise to ~14 and ~8.

6. **Monitor wrapper lost.** SB3's `_wrap_env` only inserts `Monitor` when the
   env is not already a `VecEnv` (verified in `base_class.py`). Passing
   `VecNormalize` skips it, so `rollout/ep_rew_mean` and `ep_len_mean` disappear.
   Wrap inside the lambda: `Monitor(make_env(...))`.

7. **Metrics broken in sparse mode.** `gymnasium_wrapper.py:210-213` —
   `_compute_reward` returns early when `reward_shaping=False`, so the tracker
   updates never run. `balatro/ante_reached` reports 1 and `rounds_beaten`
   reports 0 forever.

8. **Discard histogram is presence, not counts.** `observation.py:827` sets
   `= 1.0`. Loses duplicates. Also covers the discard pile rather than the
   remaining draw pile, which is the more useful quantity.

### Minor

9. **Joker rarity is a cost proxy.** `observation.py:424` uses
   `base_cost / 20` despite the docstring claiming rarity ordinal. Read real
   rarity from `centers.json`.

10. **`ability_extra` is mush.** `observation.py:438-447` sums unrelated
    numeric fields into one float. Drop it or encode per-joker.

11. **Unseeded RNG.** `gymnasium_wrapper.py:143` `default_rng()` unseeded;
    `balatro_env.py` uses global `random` for deck/stake. Breaks reproducibility.

12. **Protocol drift.** `game_spec.py` types `GameEnvironment.step` as
    returning 6 values including a reward float; `BalatroEnvironment.step`
    returns 5. `runtime_checkable` only checks method existence, so this passes
    silently.

13. **Stale comment.** `balatro_spec.py` references an `action_heads` module
    that does not exist. There is no policy/encoder module in `jackdaw/env/`.

14. **`BalatroGymnasiumEnv._rng` unseeded on a string-only reset — FIXED.**
    Found while building the Run 5 baseline eval (`scripts/eval_ppo.py`):
    `_enumerate_actions`'s subsampling RNG (`self._rng`, defect #2's
    mechanism) was only reseeded when the Gymnasium `seed: int` param was
    passed. The new `options["game_seed"]` reset path (added the same
    session, for passing a string eval seed like `"EVAL_020"` through the
    int-typed `seed` API) left `self._rng` on OS entropy, so identical
    `game_seed` resets could enumerate *different* action tables whenever
    subsampling kicked in — confirmed empirically: the same trained model,
    same seed, `deterministic=True` gave 9 steps on one process run and
    2000 (max-steps truncation) on another. This silently broke eval
    reproducibility for any policy eval built on `BalatroGymnasiumEnv`
    (not the heuristic/random baselines in `scripts/eval_agent.py` —
    those drive the factored `BalatroEnvironment` interface directly and
    never touch this RNG). Fixed in `gymnasium_wrapper.py::reset`: a
    `game_seed`-only reset now also reseeds `self._rng` from a stable
    (non-`hash()`) int derived from the same string.

15. **Deterministic policy stalls on no-op action loops — FIXED.**
    Found evaluating the Run 5 baseline (`scripts/eval_ppo.py`, 500k
    timesteps, post defects 3/5/6/7/14 fixes): with `deterministic=True`,
    **25% of the 200 frozen-eval episodes (50/200) hit the `max_steps=2000`
    cap while still stuck at ante 1** — traced one directly
    (`EVAL_004`/`EVAL_011`): the policy spams `SwapHandLeft` for 1992 of
    2000 steps after only 3 `PlayHand`s and 4 `Discard`s near the start.
    `SwapHandLeft`/`SwapHandRight`/`SortHandRank`/`SortHandSuit` are
    always-legal, near-zero-cost cosmetic actions (hand reordering only,
    no scoring effect) that the -0.001/-0.002 per-step reward penalty
    apparently doesn't disincentivize enough once the policy is uncertain
    about committing to a real play. This drags the mean episode length to
    520 despite a median of 24 (most episodes still just lose ante 1
    quickly) — a heavy-tailed distribution hidden by rollout-average
    metrics during training.

    **Fix** (`gymnasium_wrapper.py::BalatroGymnasiumEnv`): a generic stall
    detector, not an exclusion list of "cosmetic" action types (robust to
    a policy finding some other non-progressing loop later, e.g.
    alternating between two different actions). `_progress_fingerprint`
    tracks `(chips, round, ante, hands_left, discards_left, dollars)`; if
    unchanged for `_STALL_STEPS_LIMIT=20` consecutive steps, the step is
    forced `truncated=True` and an extra `_STALL_PENALTY=-1.0` is added on
    top of the normal terminal-loss reward, so stalling is a strictly
    worse outcome than losing quickly (not just an equal one).

    **Validated against the existing Run 5 checkpoint with no retraining**
    (the fix lives in the environment, so it applies transparently to any
    rollout through it): re-running `scripts/eval_ppo.py` on the same 200
    seeds post-fix gives **mean length 520.9 → 30.0, eval throughput 1.6 →
    26.1 eps/sec (16×), max single-episode length 2000 → 160** — and
    critically **mean ante is unchanged at 1.26 (max 6)**, confirming the
    fix only cuts off wasted non-progress, it doesn't alter what the
    existing policy actually achieves. `results/ppo_run5_poststall.json`
    has the corrected per-seed data.

### Not a defect

`_log_scale` (`observation.py:295-299`) is sign-symmetric and safe for negative
dollars. Ruled out as a NaN source.

`max_steps=10_000` never binds — episodes are ~20 steps and ~280 complete per
rollout, which is good for credit assignment. Set it to 2,000 as a loop guard,
not for throughput. **Update**: 2,000 does bind, but only stops the bleeding
late — see defect #15. It caps how long a stall can run, it doesn't prevent
one; 25% of Run 5's eval episodes used the full 2,000-step budget stalled at
ante 1.

---

## 5. Architecture decisions

### 5.1 Catalog embeddings

`center_key` is a **token**, not a quantity. Replace the normalized float with
`nn.Embedding(NUM_CENTER_KEYS, 32)`. Each ID gets an independent learned vector;
nothing is shared unless training discovers it should be. Same treatment at
8–16 dims for rank, suit, enhancement, edition, seal, card_set, boss blind key.

**Prerequisite:** the env must emit **raw integer IDs** as separate observation
channels (`joker_ids: MultiDiscrete([299]*5)`, etc.), populated from
`center_key_id()`. Do *not* recover them via `round(v[0] * 299)` — lossy and
fragile. Requires changes to `observation.py` and `_build_obs`.

**Implemented**: `observation.py::encode_catalog_ids` + `Observation.joker_ids`/
`consumable_ids`/`shop_ids` (populated in `encode_observation`, flowed through
`GameObservation.entity_ids` in `game_spec.py`), exposed as
`obs["{name}_ids"]` by `BalatroGymnasiumEnv._build_obs`. Column 0 of the
existing `jokers`/`consumables`/`shop_cards` float arrays is left as-is
(still the normalized `center_key_id` float) — `BalatroExtractor` (§5.2)
just doesn't read it, no dimension change needed anywhere downstream.

**Gotcha, not in the original sketch**: the channels are `spaces.Box`, not
`spaces.MultiDiscrete` as sketched above. SB3's `preprocess_obs` **one-hot
encodes every `Discrete`/`MultiDiscrete` space before a features extractor
ever sees it** (`stable_baselines3/common/preprocessing.py`) — a
`MultiDiscrete([300]*10)` arrives at `forward()` as a *3000-dim one-hot
tensor*, not the 10 raw integers `obs["joker_ids"].long()` expects. This
surfaced as a shape-mismatch crash the first time an actual `model.learn()`
call exercised the real SB3 preprocessing pipeline — extractor-only unit
tests that feed `forward()` hand-built tensors directly never call
`preprocess_obs`, so they passed regardless and didn't catch it. `Box`
passes through as a plain float tensor holding the integer value (SB3 only
special-cases `Box` for image normalization, which doesn't apply here), and
`.long()` in `forward()` recovers the exact index for `nn.Embedding`. If you
add another integer/categorical observation channel later, use `Box`, not
`MultiDiscrete`/`Discrete`, unless you actually want the one-hot expansion —
and validate any new features-extractor logic with a real (even tiny)
`model.learn()` call, not just direct `forward()` unit tests.

### 5.2 Set pooling

Jokers arrive as `(5, 15)` zero-padded. `MultiInputPolicy` flattens to 75 floats
and concatenates. Two problems: the network learns each slot independently (one
lesson must be relearned five times), and padding zeros are indistinguishable
from real zero-valued features.

Fix: shared per-entity MLP, then masked mean/max/attention pooling using
`entity_counts`. Order-invariant, count-invariant, one lesson per joker.

```python
class BalatroExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)
        self.center_emb = nn.Embedding(NUM_CENTER_KEYS, 32)
        self.joker_mlp  = nn.Sequential(nn.Linear(32 + 14, 64), nn.ReLU())
        self.global_mlp = nn.Sequential(nn.Linear(235, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 64 + ..., features_dim), nn.ReLU())

    def forward(self, obs):
        ids   = obs["joker_ids"].long()                    # (B, 5)
        feats = torch.cat([self.center_emb(ids), obs["joker"][..., 1:]], -1)
        h     = self.joker_mlp(feats)                      # (B, 5, 64)
        n     = obs["entity_counts"][:, 1:2]
        valid = (torch.arange(5, device=h.device) < n).unsqueeze(-1)
        jokers = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        ...
```

Wire via:
```python
policy_kwargs=dict(
    features_extractor_class=BalatroExtractor,
    share_features_extractor=False,
)
```

**Implemented**: `jackdaw/env/feature_extractor.py::BalatroExtractor` —
generalized from the joker-only sketch to all five entity types (uniform
per-entity-type embedding + MLP + masked-mean-pool, catalog types get the
embedding concatenated in, non-catalog types — hand_card, pack_card —
skip it). Wired into `scripts/train_ppo.py` via `--extractor
{balatro,default}` (default: `balatro`), bundled with
`share_features_extractor=False` (known issue #4) exactly as sketched
above, since that's what makes the separate extractor meaningful. `default`
keeps SB3's flatten+concat `CombinedExtractor` for a clean architecture-only
ablation against a run with reward/hyperparameters otherwise identical.
Not in `jackdaw/env/__init__.py`'s import list — it needs torch, which the
base package must stay importable without (CI's `uv sync --dev` doesn't
install `train` extras). Round-tripped through `MaskablePPO.save`/`.load`
and `scripts/eval_ppo.py` successfully in a short smoke run; a real 500k+
comparison run against the Run 5 baseline (mean ante 1.26) hasn't been done
yet — that's the natural next step.

### 5.3 Exact lookahead ("search")

The engine is deterministic, seeded, and a pure state transition. This is the
project's main advantage and it is currently unused.

**Level 1 — exact one-step evaluation.** For each legal `PlayHand`, copy the
state, apply it, read resulting chips, discard the copy. Pick the max. This is
*exact*, not heuristic: the true chip value of every available play, jokers and
hand levels included.

```python
def best_play(action_table, raw_state):
    best, best_score = None, -1
    for i, a in enumerate(action_table):
        if a.action_type != ActionType.PlayHand:
            continue
        s = copy.deepcopy(raw_state)
        result = engine.step(s, factored_to_engine_action(a, s))
        if result["chips"] > best_score:
            best, best_score = i, result["chips"]
    return best
```

Cost warning: ~218 deep copies per step. Profile it. If `deepcopy` dominates,
score hands directly through `scoring.py` without a full state copy.

**Implemented**: `jackdaw/env/heuristic_agent.py::HeuristicAgent`, exactly
this algorithm (`copy.deepcopy` + `engine.step`, reading
`gs["last_score_result"]`), plus a hand-written discard heuristic (§6 Phase
0) and trivial defaults for every other action type. Evaluate with
`uv run scripts/eval_agent.py --agent heuristic`. Confirmed: `deepcopy`
does dominate — see §10.

**Level 2 — lookahead as observation features.** Feed "best achievable chips
this hand", "best chips after one optimal discard", and their ratios to blind
target into the global vector. Turns an inference problem into a lookup.

**Level 3 — MCTS / expert iteration.** Out of scope for now. Balatro is
stochastic (shop, draws, packs), so it needs determinization or Stochastic
MuZero. Noted as a door that exists.

### 5.4 Factored action head (later)

Flat `Discrete(500)` has unstable semantics: index 37 means different things at
different timesteps, and the network can't infer the enumeration from the
observation. Combined with defect #2 (random subsampling), the mapping is also
non-stationary across identical states.

Target architecture, matching the native `FactoredAction`:
- action-type head over 21 types
- pointer head attending over entity embeddings for `entity_target`
- card selection either autoregressive, or delegated entirely to exact
  evaluation (§5.3)

`GameSpec` already provides the right contract: `entity_type_for_action()`,
`needs_entity_set`, `catalog_size`. Building this means filling a well-shaped
hole, not designing from scratch.

**Interim cheap fix:** make the enumeration canonical and deterministic — fixed
sort order, fixed index blocks per action type, deterministic subsampling seeded
on state. May be worth more than any architecture change on its own.

### 5.5 On the LSTM

Wanted for "memory / long-term planning". Two honest caveats:

- **No `MaskableRecurrentPPO` exists.** sb3-contrib ships `MaskablePPO` and
  `RecurrentPPO` separately. Combining them means writing it yourself. The
  common workaround (hard-masking inside the env) throws away correct logit
  zeroing and log-probs.
- **Balatro is closer to fully observed than it looks.** Jokers, vouchers, hand
  levels, money, ante, blind are all in the observation. The genuinely hidden
  state is mainly the remaining draw pile — and adding a 52-dim remaining-deck
  count histogram solves that directly, rather than making an LSTM learn card
  counting through BPTT.
- Building toward a joker synergy across antes is a **credit assignment**
  problem, not a memory problem. The current joker set is already observable.

**Decision: keep the LSTM as a controlled ablation after the factored head, not
as a foundation.** Since the factored head already requires a custom policy, the
marginal cost of adding recurrence drops a lot at that point. If cheap memory is
wanted sooner, add explicit history features (last N actions, cards played this
round, shop items passed on).

---

## 6. Phases

### Phase 0 — Instrument (weekend)
- [x] Fix defects 3, 5, 6, 7 (seed collision, truncation, Monitor, metrics).
      3: `scripts/train_ppo.py::make_env` now takes `worker_id`, folded into
      `seed_prefix`; `main()` builds one `Monitor`-wrapped env per
      `--n-envs` worker with a distinct id. 5: `balatro_spec.py` — hand_card
      max_count 8→20, joker 5→10 (tuned up further after the initial
      8→14/5→8 estimate; see `balatro_spec.py` for current values — treat
      as a tunable estimate, not a fixed contract). 6: each sub-env is
      wrapped in `Monitor(...)`
      before `DummyVecEnv`/`VecNormalize`. 7: `gymnasium_wrapper.py::
      _compute_reward` now updates `_episode_max_ante`/`_episode_max_round`
      unconditionally, before the sparse-mode early return, not only in the
      dense-shaping branch.
- [x] Profile. Confirm whether `_enumerate_actions` dominates. **Confirmed,
      yes.** `py-spy` couldn't attach a target process in this sandboxed
      Windows session (`Error: Failed to find python version from target
      process` even with `--nonblocking`), so used stdlib `cProfile` instead
      — same investigative goal. Over a short single-env training run,
      `_enumerate_actions` (`gymnasium_wrapper.py:279`) is **60.9% of
      `BalatroGymnasiumEnv.step`'s cumulative time** (14.4s of 23.7s
      cumulative across 12288 steps), and its hottest line by self-time is
      the int-cast generator expression inside `_card_combos`
      (`gymnasium_wrapper.py:78`) building combination-index tuples — 21.4M
      calls, more self-time than any other function in the profile besides
      torch's own C internals. Directly corroborates known issue #2 (random
      subsampling in `_enumerate_actions`) and motivates Phase 2's
      "canonical deterministic enumeration" item — not fixed here, that's
      explicitly Phase 2 scope, this item was only to confirm the
      diagnosis.
- [x] **Heuristic agent** with exact one-step play evaluation (§5.3 Level 1) —
      `jackdaw/env/heuristic_agent.py::HeuristicAgent`. `flush_bot.py` doesn't
      exist in this repo, so *discards* use a small hand-written near-flush /
      near-straight heuristic instead (lookahead still can't solve discards —
      discarding scores nothing; value appears next draw)
- [x] Frozen eval harness: 200 held-out seeds, deterministic policy, reports
      ante distribution + win rate — `jackdaw/env/eval_seeds.py::EVAL_SEEDS`,
      `jackdaw/env/rollout.py`, `scripts/eval_agent.py`
- [x] Observation golden tests — snapshot encoded observations for fixed states
      and assert. Needed before refactoring `observation.py` while preserving
      live-bridge parity — `tests/env/test_observation_golden.py` +
      `tests/fixtures/observation_golden.json` (two fixed scenarios: right
      after reset at `BLIND_SELECT`, and after `SelectBlind` at
      `SELECTING_HAND` with a dealt hand — both cheaply reproducible,
      deliberately not depending on any hand-selection outcome)

**Exit:** known steps/sec, known heuristic mean ante, working eval harness.
Phase 0 complete.
Heuristic over the full 200-seed `EVAL_SEEDS` set (`results/heuristic_v1.json`,
`uv run scripts/eval_agent.py --agent heuristic --episodes 200`): **mean ante
2.75, max ante 7, win rate 0%, ~0.09 episodes/sec (~2272s total)** — the
"reach ante 3-4 unaided" prediction in §5.3 lands close but on the low side
of mean; individual seeds range from ante 1 up to 7. RandomAgent baseline
over the same 20-seed prefix: mean ante 1.00. Zero wins for either agent —
"beat Red Deck / White Stake" is not solved by hand-selection lookahead
alone; shop/economy/joker-synergy play (deliberately left trivial in
`HeuristicAgent`, see its module docstring) is the likely next lever, along
with the win condition itself requiring surviving ante 8, well past what a
PlayHand-only heuristic reaches.

### Phase 1 — Representation (1–2 weeks)
- [x] Emit integer catalog IDs from `observation.py` — §5.1
- [~] `BalatroExtractor` with embeddings + masked set pooling — §5.2.
      Implemented; run 6 (pooling all 5 entity types) regressed to
      random-baseline (mean ante 1.00); fixed by pooling only
      joker/consumable/shop_item and flattening hand_card/pack_card — see
      §3 "Run 6"/"Run 7". **Run 7 (the fix) result: mean ante 1.26, exact
      parity with run 5** (different ante-distribution shape, same mean) —
      not a regression, not yet a proven win. Leave this unchecked
      (implemented-but-inconclusive) until a longer run or second seed
      resolves it one way or the other — see the Exit note below.
- [x] `share_features_extractor=False` — bundled into `train_ppo.py`'s
      `--extractor balatro` wiring (§5.2)
- [ ] 52-dim remaining-deck **count** histogram — blocked on §10's open
      `BridgeAdapter` deck-composition question (live-bridge parity)
- [ ] Lookahead features (§5.3 Level 2)
- [ ] Real joker rarity; drop `ability_extra`
- [ ] `SubprocVecEnv` × 6 on Linux with distinct per-worker seed prefixes —
      `make_env`'s `worker_id` param (Phase 0, defect #3) is already
      collision-safe for this; only the `DummyVecEnv` → `SubprocVecEnv`
      swap itself remains

**Exit:** `explained_variance` stops decaying; no NaN crashes; PPO approaches
the heuristic. **Not yet met** — run 7 landed at "still ~1.0-1.3" (mean ante
1.26, identical to run 5). **Next action for whoever picks this up**: this
needs a longer run (e.g. 1-2M steps — embeddings for jokers/shop/consumables
may need more than 500k steps to warm up, since they're rarely observed
while the agent is still usually losing at ante 1) or a second seed at 500k,
*before* concluding the representation fix helps, hurts, or is neutral.
Don't change reward shaping or hyperparameters in that same follow-up run —
only step count or seed, so the comparison stays isolated (§8 rule 1). If a
longer/repeated run is still ~1.0-1.3, that's real evidence the bottleneck
has moved to the action space (§5.4/Phase 2) rather than representation —
worth reconsidering priority order at that point. If it's clearly higher,
check off this item and move to the remaining Phase 1 items above (joker
rarity is probably the next cheapest win; the deck histogram needs the §10
BridgeAdapter question resolved first).

### Phase 2 — Action space (2–3 weeks)
- [ ] Canonical deterministic enumeration; eliminate random subsampling
- [ ] Factored action head with entity pointer (§5.4)
- [ ] Curriculum: success = "clear ante N", auto-advance on eval threshold

**Exit:** beats the heuristic.

### Phase 3 — Polish
- [ ] LSTM ablation
- [ ] Optuna hyperparameter sweep (only now — sweeping before the
      representation is fixed just burns compute)
- [ ] Live validation via `BridgeAdapter` on Windows
- [ ] Writeup

---

## 7. Hyperparameters

Current working configuration:

```python
MaskablePPO(
    "MultiInputPolicy", env,
    learning_rate=lambda p: 3e-4 * p,   # schedules DO work for lr
    ent_coef=0.005,                     # constructor default; callback overrides
    n_steps=4096,
    batch_size=256,
    clip_range=0.15,
    clip_range_vf=0.2,
    target_kl=0.02,
    policy_kwargs=dict(share_features_extractor=False),
)
```

Notes learned the hard way:

- **`ent_coef` cannot be scheduled through the constructor.** SB3 wraps only
  `learning_rate`, `clip_range`, `clip_range_vf` with `get_schedule_fn`.
  `ent_coef` is a plain float used directly as `self.ent_coef * entropy_loss`.
  Pass a callable and you get a `TypeError`. Use a callback that mutates
  `self.model.ent_coef` in `_on_rollout_start`.
- **The schedule denominator must be `total_timesteps`, not `max_steps`.**
  This bug occurred twice. Assert inside the callback constructor, where it
  guards the value actually used.
- **`ent_coef=0.05` was 6× the policy gradient term.** Loss decomposition at
  run 2: pg −0.0301, entropy 0.05 × (−3.73) = −0.187, value 0.5 × 0.0017.
  The dominant instruction was "stay random".
- **`ent_coef=0.0005` is past the stability edge.** Use 0.005 → 0.001.
- **Reward scaling is mostly a no-op on the policy.** `normalize_advantage=True`
  standardizes advantages per minibatch, so a uniform ×10 is normalized away
  from the policy gradient — while inflating value MSE ~100×. Use
  `VecNormalize(norm_reward=True)` instead; it adapts as returns grow.
- **`target_kl=0.02` is the guard that catches slow KL drift.** Off by default.
- `gamma=0.99` gives a ~100-step horizon. Fine now; raise toward 0.999 once
  episodes lengthen past a few hundred steps.

### Reward shaping

Current (post-fix) structure in `_compute_reward`:

```
step:          -0.001  (-0.002 in shop)
blind beaten:  +0.15   (flat — ante scaling removed)
boss beaten:   +0.10
efficient:     +0.01 * hands_left
score progress:+0.02 * min(chip_delta / blind_target, 1.0)
terminal:      +0.5 win / -0.2 loss
```

The original `0.15 * (ante/8)` was backwards: ante 1 paid 0.019 against a −0.2
death penalty, so ten blinds were needed to offset one loss — and since the
agent lost every episode, that −0.2 was a near-constant offset carrying no
gradient. Flat scaling fixed this.

Consider potential-based shaping (`F = γΦ(s') − Φ(s)`) later; it's provably
policy-invariant, unlike the current form.

---

## 8. Experiment discipline

Four runs so far are only partly comparable because reward scale changed
mid-stream, making `ep_rew_mean` meaningless across runs.

Rules going forward:

1. **Never change reward shaping and hyperparameters in the same run.**
2. Freeze the reward definition; version it if it must change.
3. [x] Checkpoint every 50k steps —
   `CheckpointCallback(save_freq=50_000, save_path=...)`. Run 3 lost 364k steps
   of training to a crash with no checkpoint. **Implemented**:
   `scripts/train_ppo.py --checkpoint-freq` (default 50_000).
4. [x] Save `VecNormalize` statistics alongside the model, or evaluation is
   wrong. **Implemented**: `CheckpointCallback(save_vecnormalize=True)` in
   `train_ppo.py` (moot for `--extractor balatro` predictions specifically —
   `norm_obs=False`, so only reward normalization is saved, which doesn't
   affect action selection at eval — but kept for correctness/completeness).
5. [x] Compare on the **frozen eval seed set**, not on rollout statistics
   (noisy, confounded by exploration). **Implemented**:
   `jackdaw/env/eval_seeds.py::EVAL_SEEDS` (200 seeds) +
   `scripts/eval_agent.py` (heuristic/random) + `scripts/eval_ppo.py`
   (trained checkpoints) — see §9 for exact commands.
6. Pin `mean_ante_reached` to a `[1, 8]` axis. Auto-scaling makes a 1.00→1.03
   change look like progress.

### How to judge a run (quick reference)

**The only trustworthy comparison number**: `mean_ante` from
`uv run scripts/eval_ppo.py --model <path> --episodes 200`, not any
TensorBoard rollout stat. Compare against: random (1.00), heuristic (2.75,
the bar to beat), and the previous run. Also look at the ante
*distribution*, not just the mean — `python -c "import json,collections;
d=json.load(open('results/ppo_runN.json'));
print(collections.Counter(r['ante_reached'] for r in d['per_seed']))"` —
a mean pulled up by one or two lucky seeds is a very different result from
broad improvement across many.

**Training-time health** (TensorBoard, `tensorboard --logdir runs/balatro_ppo`
— see "Early warning signs" below for the exact crash-precursor pattern):
- `train/explained_variance` should trend up and stay non-decaying — this
  is literally what predicted the run 3/4 NaN crashes.
- `train/value_loss` climbing steeply/curving vertical, not gradually
  decreasing, is the other crash precursor.
- `rollout/ep_rew_mean`/`ep_len_mean` should trend upward, but are
  stochastic-rollout stats — directional sanity checks only, never the
  final number.
- `time/fps` cratering mid-run is itself a symptom (e.g. a stall loop the
  detector isn't catching, or a joker-generation bug inflating entity
  counts).
- Any NaN in logits (`ValueError: ... probs ... Simplex()`) — same failure
  mode as runs 3/4; checkpointing now means this no longer loses the whole
  run.

**Calibration for where this project is right now**: the heuristic-only
baseline (mean ante 2.75, win rate 0%) is the bar. Win rate staying 0% is
*not* itself a red flag yet — winning needs clearing ante 8, far beyond
current results either way — it only becomes a meaningful signal once mean
ante is much higher. A run scoring meaningfully above the previous run's
mean ante, with `explained_variance` healthy and no crashes, is progress;
a run at or below random (1.00) with otherwise-healthy training curves (as
Run 6 was) means the *architecture*, not the training dynamics, is wrong —
check what information a representation change might be destroying before
assuming it needs more steps or different hyperparameters.

### Reading the charts

- **Smooth:** `mean_ante_reached`, `ep_len_mean` (~0.6). Real trends, noisy,
  genuinely averageable.
- **Do NOT smooth:** `max_ante_reached` — it's a per-rollout maximum and maxima
  don't average meaningfully. Prefer a histogram or 90th percentile.
- **Do NOT smooth:** `win_rate` at ~0 — smoothing creates a nonzero-looking
  trace suggesting progress that isn't there.
- Read **raw** for before/after comparisons; EMA lags and shifts apparent onset.
- Variance itself is signal on `value_loss` and `policy_gradient_loss`.

### Early warning signs

Both crashes were visible ~200k steps ahead:
- `clip_fraction` sustained above ~0.3
- `approx_kl` rising monotonically
- `value_loss` curving vertical
- `explained_variance` declining while `value_loss` climbs

Use `VecCheckNan(env, raise_exception=True)` during development. It raises at
the moment a NaN appears and says whether it came from observations or the
network — far faster than reading a `Simplex()` traceback backwards. Has a
per-step cost; drop it for long runs.

---

## 9. Tooling

| Tool | Use | When |
|---|---|---|
| **py-spy** | Sampling profiler, flamegraphs, no code changes | Phase 0, first |
| **Weights & Biases** | Experiment tracking; a public report is a good portfolio artifact | Phase 0 |
| **CleanRL** | Single-file PPO you own. See `ppo_multidiscrete_mask.py` (Gym-µRTS) — closest existing reference to a factored, masked, entity-based action space | Phase 2 |
| **tyro** | Dataclass → CLI, lighter than Hydra | Phase 1 |
| **PufferLib** | Faster vectorization, complex obs spaces | If throughput-starved |
| **Sample Factory** | Async PPO for CPU-bound envs, built-in LSTM | If Phase 2 stalls on sample count |
| **Optuna** | Hyperparameter search | Phase 3 only |
| **pytest** golden tests | Observation snapshots | Phase 0, before refactoring |

**Move off SB3 at Phase 2, not before.** SB3 handles the custom feature
extractor fine. Once a factored pointer head with conditional cross-head masking
is needed — plus possibly recurrence, which sb3-contrib can't combine with
masking — fighting SB3's abstractions costs more than owning the loop.

Reading: Huang et al., *A Closer Look at Invalid Action Masking in Policy
Gradient Algorithms* (arXiv 2006.14171) — why masked log-probs matter.

---

## 10. Open questions

- [ ] Does `BridgeAdapter` expose remaining-deck composition? The live game lets
      players inspect it, so it should be legitimate — needs verification before
      relying on it in the observation.
- [x] What is the actual cost of `deepcopy(raw_state)`? **Answered**:
      dominates. `HeuristicAgent` (full deepcopy + `engine.step()` per
      candidate) runs at ~0.10 episodes/sec (~10s/episode,
      `uv run scripts/eval_agent.py --agent heuristic`) vs. the Gym env's
      ~417 steps/sec — roughly two orders of magnitude slower. Fine for
      offline baselining (this eval harness), *not* viable as an inline
      per-step training-loop component without the optimization below.
- [ ] Can `scoring.py` evaluate a candidate hand without a full state copy?
      Still open — worth resolving before using lookahead as an §5.3 Level 2
      observation feature *inside* training, where per-step cost matters.
      `_handle_play_hand` (`game.py`) populates ~a dozen synthetic snapshot
      fields before calling `score_hand`; a synthetic-snapshot path would
      need to reproduce those without duplicating engine bookkeeping.
- [x] Does `_enumerate_actions` dominate the profile, as suspected? **Yes** —
      see §6 Phase 0 for the measurement.
- [ ] What are real hand-size and joker-slot maxima in practice? Sets the new
      `max_count` values. Partially addressed: `balatro_spec.py` now uses
      this section's own "~14 and ~8" estimate (defect 5), not a value
      independently derived from real hand-size/joker-count data — an
      empirical pass (e.g. via Juggler/Turtle Bean/voucher stacking) could
      still tighten or raise these further.
