# RL Plan — Jackdaw Balatro

Working plan for training an agent to play Balatro via the Jackdaw simulator.

**This document is the plan: what to do next and why.** Individual training
runs — what each changed, scored, and taught — live in
[`RUNS.md`](RUNS.md), so this file can stay short and current as that one
grows. When a run resolves something here, update this file and put the
detail there.

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
validated against live Balatro through BalatroBot. Note this is currently
**aspirational rather than enforced** — see §5.3, finding 2.

### Success bar

| Agent | Mean ante | Max ante | Win rate |
|---|---|---|---|
| Random | 1.00 | — | 0% |
| **Heuristic** (exact one-step lookahead) | **2.75** | 7 | 0% |
| Best PPO so far (run 20, 1M steps + lookahead) | 1.87 | 6 | 0% |

**The RL agent is not interesting until it beats the heuristic.** That gap is
1.87 → 2.75 and is the number that matters — down from 1.42 → 2.75 at the
start of this work.

---

## 2. Hardware and compute budget

| Resource | Spec | Notes |
|---|---|---|
| CPU | Intel i5-12400F, 6C/12T | **The binding constraint.** |
| GPU | Radeon RX 6750 XT, 12 GB | gfx1031, not officially ROCm-supported |
| RAM | 32 GB | Not a limit |
| OS | Train on Linux; validate on Windows | LiveBackend/BalatroBot is Windows-only |

**Measured throughput: ~160 fps single-env; ~485 fps with
`--vec-env subproc --n-envs 8`** — a 500k-step run in ~17 min instead of ~52.
See `RUNS.md` "Throughput" and "Parallel rollout collection". The short
version:

- **~91% of env wall-clock is `_enumerate_actions`** — the top-K ranking that
  bought the 1.26 → 1.42 gain. Engine transition plus observation encoding is
  0.30 ms of it.
- **The network is nearly free per step, but it is the serial fraction.** It
  runs in the main process, which is why 8 workers give 3.0x end-to-end
  against 3.8x raw env scaling.
- **fps carries ±70% exogenous variance on this machine** (runs 7 and 8 are the
  same code and differ 1.7×). Never read an fps change as a code signal without
  `scripts/bench_step.py` confirming it.
- **`n_steps` is per-env**: use `--rollout-steps` rather than raising
  `--n-envs` alone, or the rollout multiplies by worker count.

**Do not rent cloud compute.** The arithmetic doesn't justify it. If throughput
later becomes the bottleneck, rent **CPU** (e.g. Hetzner CCX/AX, ~€40–60/mo for
16–32 cores), not GPU — a GPU instance would sit idle behind the same Python
simulator bottleneck.

**ROCm setup (optional):** Ubuntu 22.04/24.04, PyTorch ROCm wheel, and
`HSA_OVERRIDE_GFX_VERSION=10.3.0` to present gfx1031 as gfx1030. Free upside,
but the networks here are small MLPs and embeddings and account for <20% of
wall-clock — the GPU is close to irrelevant. Do not treat it as a blocker.

---

## 3. Current state

Twenty MaskablePPO runs. **Best: mean ante 1.865** (run 20 — 1M steps with
lookahead features), against the heuristic's 2.75. Full history in
[`RUNS.md`](RUNS.md).

The agent dies inside ante 1 about 58% of the time. Small blind at ante 1 needs
300 chips; a near-random 5-card play scores 20–60, so four hands reaches ~150.
Clearing the first blind requires actually selecting good hands, not sampling.

### What is established

- **Hand selection was the bottleneck, and fixing the action table was the only
  change that has produced a confirmed gain.** A deterministic top-K menu
  (defect #2) moved 1.26 → 1.42 across two seeds with ante-1 clear rates of
  73/70 per 200 versus a pre-fix 11-38. The groups don't overlap and the
  within-group spread collapsed to zero.
- **Hyperparameter tuning is exhausted.** Runs 3 and 4 had very different policy
  dynamics and near-identical task performance.
- **The seed-to-seed noise floor is ≈0.13 mean ante** at 500k steps (runs 7 vs
  8). A single run landing 0.1 above another means nothing.
- **The embedding path is not the lever, and not for the reason previously
  believed.** Four `balatro`-extractor runs average ~1.14 against the default
  extractor's 1.26. The tables sit at ~97% of their random init — but **not**
  because jokers go unseen: the agent touches 95% of the reachable joker
  catalog, and did so *before* the action-table fix too. Each exposure just
  contributes almost nothing. See `RUNS.md` "the joker embeddings barely move".
- **More survival will not unlock the embeddings.** That was the previous
  hypothesis and the exposure measurement contradicts it. The remaining gap is
  credit assignment, not data availability.
- **Training length moved the ceiling where nothing else had.** Every earlier
  change improved the ante-1 clear rate and left the ceiling at 3-4. At 1M
  steps run 20 reaches ante 6 with 48/200 episodes at ante 3+ (vs 14-25).
  Confounded with the LR schedule — see `RUNS.md`.
- **Lookahead features (§5.3 Level 2) help, inconsistently.** All four
  lookahead runs beat both baselines; ablation attributes run 15's gain
  causally to the feature content. Rescaling the block's field magnitudes to
  make that consistent was tried and failed (runs 16/17).

### What is unresolved

- **Why only one seed in two learns to use the lookahead features.** The
  features carry signal — run 15's gain is causally attributable to them by
  ablation — but run 13 largely ignored them. The leading candidate is the
  in-block scale imbalance (§5.3); untested.

### Recently resolved

- **The "NaN in logits" crash (runs 3, 4, 9, 14) — solved, and it was never a
  NaN.** A float32 softmax over `MAX_ACTIONS`=500 occasionally sums past
  `Simplex()`'s 1e-6 tolerance, and sb3-contrib's `apply_masking` validates a
  *stale* copy of it. Disabling distribution validation fixes it. Full
  diagnosis in `RUNS.md`. **Note what this costs retroactively**: run 9 was
  read as evidence about long-run stability and LR schedules, and run 4's
  "critic blowup" mechanism was treated as the standing explanation. Neither
  applies. Don't reason from those runs' crashes.
- **Live-bridge parity is aspirational.** `bot_state_to_game_state` returns
  neither Card objects nor `hand_levels`, so essentially no current observation
  feature is bridge-computable. See §5.3 finding 2.

---

## 4. Known defects

**Numbers are stable identifiers — code comments and tests reference them
(`known issue #2`, `#5`, `#15`, …). Never renumber; append instead.**

### Open

8. **Discard histogram is presence, not counts.** `observation.py:840` sets
   `= 1.0`, losing duplicates. Also covers the discard pile rather than the
   remaining draw pile, which is the more useful quantity (and is what the
   52-dim deck histogram in Phase 1 wants).
9. **Joker rarity is a cost proxy.** `observation.py:424` uses `base_cost / 20`
   despite the docstring claiming rarity ordinal. Read real rarity from
   `centers.json`.
10. **`ability_extra` is mush.** `observation.py:438-447` sums unrelated
    numeric fields into one float. Drop it or encode per-joker.
11. **Unseeded RNG on the default paths.** `gymnasium_wrapper.py:313`
    constructs `default_rng()` with OS entropy (only the seeded reset paths
    override it — see fixed #14), and `balatro_env.py:98-100` uses the global
    `random` module for deck/stake choice. Only bites when `back_keys`/`stakes`
    have more than one entry, which no current run does.
20. **The meta-jokers are registered but never actually exercised.**
    `jackdaw/cli/scenarios/jokers.py` registers `j_four_fingers`,
    `j_shortcut`, `j_smeared`, `j_splash` and `j_pareidolia` with
    `hand_preset=None`, which plays a generic hand `[0..4]`. Those five are
    exactly the jokers that change *which hands are detectable*
    (`_META_JOKER_FLAGS` in `hand_eval.py`), and a generic hand usually
    detects the same type with or without them — so the relaxed rules are
    probably never hit against the live game. No category targets hand
    detection at all.
    **Fix**: scenarios that inject a deliberate 4-card flush plus an off-suit
    card (Four Fingers), a gapped straight (Shortcut), and a mixed red/black
    flush (Smeared), asserting the detected hand type. This is the live-game
    counterpart to `tests/engine/test_hand_eval_refactor_golden.py`, which
    covers the same paths offline but can only prove "unchanged", never
    "matches real Balatro". Worth doing before the next `hand_eval` change
    (defect #18's follow-ups: `is_suit`, enum attribute access).
12. **Protocol drift.** `game_spec.py` types `GameEnvironment.step` as
    returning 6 values including a reward float; `BalatroEnvironment.step`
    returns 5. `runtime_checkable` only checks method existence, so this passes
    silently.

### Fixed

One-liners; the diagnosis that mattered is in `RUNS.md`.

1. **Joker identity destroyed** — `center_key` was one normalized float,
   telling the network joker #47 ≈ #48. Raw integer IDs now also emitted
   (`observation.py::encode_catalog_ids` → `obs["{name}_ids"]`) and consumed by
   `BalatroExtractor`. §5.1. *(Fixed, but did not help — see §3.)*
2. **Legal actions randomly vanished** — a global `_subsample` dropped the
   optimal play 8.3% of the time at hand size 8 and 68.6% at 10, and gave the
   same state a different menu on each visit. Now a deterministic top-K
   ranking. §5.3. *(This is the fix that worked.)*
3. **Parallel workers played identical games** — one `seed_prefix` across
   workers meant batch diversity 1 at N× cost, silently. Now
   `f"{prefix}_w{worker_id}"`.
4. **Shared features extractor** — a critic blowup could corrupt the policy
   through shared weights. Now `share_features_extractor=False`.
5. **Entity truncation** — `hand_card` max_count 8, `joker` 5, so the agent
   could act on cards it never observed. Now 20 and 10 (a tunable estimate,
   not a fixed contract — see §10).
6. **`Monitor` wrapper lost** when passing `VecNormalize`, killing
   `ep_rew_mean`/`ep_len_mean`. Now wrapped inside the env lambda.
7. **Metrics broken in sparse mode** — `_compute_reward` returned early before
   the tracker updates, so `ante_reached` reported 1 forever.
13. **Stale comment** in `balatro_spec.py` referencing a nonexistent
    `action_heads` module.
14. **`BalatroGymnasiumEnv._rng` unseeded on a string-only reset** — made
    "frozen" eval seeds non-reproducible whenever subsampling triggered
    (confirmed: same model, same seed, 9 steps on one process run and 2000 on
    another). Now reseeded from a stable `zlib.crc32` int, not Python's
    per-process-randomized `hash()`.
15. **Deterministic policy stalled on no-op action loops** — 25% of run 5's
    eval episodes hit `max_steps=2000` at ante 1 spamming `SwapHandLeft`, a
    heavy-tailed distribution hidden by rollout averages. Now a generic
    no-progress stall detector (`_progress_fingerprint` over
    `(chips, round, ante, hands_left, discards_left, dollars)`, 20-step limit,
    force-truncate + `-1.0`) rather than an exclusion list of "cosmetic"
    actions, so it survives a policy finding some other non-progressing loop.
    Validated on the run 5 checkpoint with no retraining: mean length 520.9 →
    30.0, throughput 16×, **mean ante unchanged** at 1.26.
17. **`SubprocVecEnv` never enabled** — `--vec-env subproc` now wires it,
    with `--rollout-steps` so raising `--n-envs` doesn't silently multiply the
    rollout. 3.0x end-to-end at 8 workers.
18. **`hand_eval.get_x_same` rebuilt the same rank grouping four times per
    hand**, with an O(n²) scan — the hottest function in training. Now
    `group_by_rank` computes it once (env 220 → 309 steps/sec). Engine change:
    guarded by a 4,011-hand golden fixture, **still needs `jackdaw validate`**.
19. **The benchmark suite was flaky and missed the hot path entirely** —
    unseeded global `random` (4 of 5 runs failed), and `test_env_steps_per_second`
    drives `DirectAdapter`, which never builds an action table. Seeded, plus
    two benchmarks covering the real path.
16. **`EntCoefSchedule` hardcoded** to `total_timesteps=500_000` regardless of
    the CLI flag; **`CheckpointCallback` wrote to a flat shared directory** so
    two runs reaching the same step count silently clobbered each other's
    recovery checkpoint. Both found while investigating run 9.

### Not a defect

- `_log_scale` (`observation.py:295-299`) is sign-symmetric and safe for
  negative dollars. Ruled out as a NaN source.
- `max_steps` — 10,000 never binds (episodes are ~20-40 steps). 2,000 is a loop
  guard; it caps how long a stall runs but doesn't prevent one, which is what
  fixed #14 is for.

---

## 5. Architecture decisions

### 5.1 Catalog embeddings — implemented, did not help

`center_key` is a **token**, not a quantity, so it gets
`nn.Embedding(NUM_CENTER_KEYS, 32)` rather than a normalized float. The env
emits raw integer IDs as separate channels
(`observation.py::encode_catalog_ids`, `Observation.joker_ids` /
`consumable_ids` / `shop_ids`, flowed through `GameObservation.entity_ids`,
exposed as `obs["{name}_ids"]`). Column 0 of the existing float arrays is left
as-is; `BalatroExtractor` just doesn't read it, so nothing downstream changed
dimension.

Do *not* recover IDs via `round(v[0] * 299)` — lossy and fragile.

**Gotcha worth keeping**: the channels are `spaces.Box`, not
`spaces.MultiDiscrete`. SB3's `preprocess_obs` **one-hot encodes every
`Discrete`/`MultiDiscrete` space before a features extractor sees it**, so
`MultiDiscrete([300]*10)` arrives at `forward()` as a 3000-dim one-hot rather
than 10 integers. `Box` passes through as a float tensor holding the integer,
and `.long()` recovers the index. This surfaced as a crash the first time a
real `model.learn()` exercised the pipeline — **extractor-only unit tests feed
`forward()` hand-built tensors and never call `preprocess_obs`, so they passed
against the broken version.** Validate any new features-extractor input with a
real (even tiny) `model.learn()` call.

**Outcome**: implemented and measured; see §3 and `RUNS.md`. Not the lever.

### 5.2 Set pooling — implemented, applies to catalog types only

Shared per-entity MLP, then masked mean pooling using `entity_counts`:
order-invariant, count-invariant, one lesson per joker instead of one per slot.

**Pooling applies only to `joker`/`consumable`/`shop_item`.**
`hand_card`/`pack_card` are flattened. Pooling them collapsed N cards in
specific slots into one averaged vector and destroyed exactly the per-slot
identity `PlayHand`/`Discard`'s `card_target` needs — run 6 regressed to
random-baseline. Pooling only makes sense once something downstream consumes a
per-entity stream via attention/pointer lookup (§5.4) instead of today's flat
`Discrete(500)` linear layer. Guarded by
`test_hand_card_order_is_not_pooled_away`.

**Embedding init scale**: `embed_init_std` defaults to 1.0. Shrinking it to 0.1
was tried (run 10) and scored *worse* — an untrained N(0,1) row is not only
noise, it's a random identity code the downstream Linear can read for free.
The knob is kept for study; treat "small init is obviously better" as a
hypothesis already tested and not supported.

### 5.3 Exact lookahead ("search")

The engine is deterministic, seeded, and a pure state transition. This is the
project's main advantage.

**Level 1 — exact one-step evaluation. Implemented.** For each legal
`PlayHand`: copy the state, apply it, read resulting chips, discard the copy,
pick the max. Exact, not heuristic — the true chip value of every play, jokers
and hand levels included. `jackdaw/env/heuristic_agent.py::HeuristicAgent`,
plus a hand-written near-flush/near-straight discard heuristic (lookahead can't
solve discards — discarding scores nothing; value appears next draw). **Mean
ante 2.75.**

**Level 2 — lookahead as observation features. Implemented, opt-in via
`--lookahead`.** Three measured findings constrained the design:

1. **Exact lookahead is unaffordable as an observation feature.** An
   observation feature is recomputed every step, unlike the heuristic which
   pays once per decision. Measured on an 8-card hand, 218 candidates:
   `deepcopy`+`engine.step` = **377 ms per decision**, of which **342 ms (91%)
   is `deepcopy`** — a 500k-step run would go from ~50 minutes to ~25 hours.
   `evaluate_hand` (pure, RNG-free, no state copy) does all 218 in **3.5 ms**.
   Any Level 2 implementation must be built on the cheap path, or on
   cheap-rank-then-exactly-score-top-K.
2. **These features cannot be computed against the live bridge.**
   `bot_state_to_game_state` says so plainly: *"Does NOT create a fully
   functional game_state (no RNG, no Card objects)"* — it returns `hand_keys`
   (strings), no `hand_levels`, no `blind` object. This is not specific to
   lookahead: essentially every existing observation feature has the same
   problem, so §1's live-bridge constraint is aspirational today. It does mean
   a policy trained on lookahead features can't be validated against real
   Balatro until the adapter rebuilds Cards/`hand_levels`/`blind`.
3. **The action space used to be unable to express the play the feature
   describes** — fixed, and that fix turned out to be the win on its own.

**Measured: recall@K of a cheap joker-blind ranking.** The action table doesn't
need accurate scores, only for the true-best play to be *somewhere in the
offered set*. Ranking all 218 subsets by `evaluate_hand` → hand-type value at
current level + scoring-card chips (no joker effects, no RNG), against exact
`deepcopy`+`step` ground truth over 60 real `SELECTING_HAND` states:

| K | 1 | 5 | 10 | 20-50 | 100 |
|---|---|---|---|---|---|
| recall@K | 48.3% | 80.0% | **95.0%** | 95.0% | 98.3% |

Mean rank of the exact-best play: 8.2. **K=10 captures 95%**, versus the old
subsampling's 91.7% at hand size 8 and 31.4% at hand size 10, with a ~20×
smaller action space. The plateau from K=10 to K=50 is the informative part:
the residual ~5% is *systematic*, not a sampling problem — hands where joker
effects dominate and a joker-blind ranker misjudges. More K doesn't fix it;
joker-aware cheap scoring would, if that 5% turns out to matter.

**Implemented in `gymnasium_wrapper.py`**: `PLAY_COMBO_BUDGET=12`,
`CARD_COMBO_BUDGET=20`, slots reserved per hand type and for
small-cardinality (setup) plays, one scoring pass shared by `PlayHand` and
`Discard`. Action table 500 → ~33 entries. This is defect #2's fix and the
source of the 1.26 → 1.42 gain.

#### What Level 2 actually adds

The original sketch — "best achievable chips this hand, best chips after one
optimal discard, ratios to blind target" — needed narrowing, because the
global vector already carries most of the *hand-level* version of it
(`observation.py` [211:235]): the best hand type detectable across all 8
cards, its base chips/mult at level, a score-to-blind ratio, flush and
straight proximity, draw outs from the remaining deck, and an urgency ratio.
Re-emitting those would have been redundant.

What was genuinely missing is the *menu*: the value of the best **subset the
action table will actually offer**, which differs from the hand-level
features in two ways that matter — it includes the scoring cards' own chip
contributions (the base-score features ignore them entirely), and it is a max
over the specific combos on offer rather than a property of the whole hand.
It is also the half of the picture that only became coherent after defect #2:
describing "the best achievable score is X" while the play achieving X was
missing from the menu 8-69% of the time was incoherent.

`_lookahead_features` emits 8 bounded dims (`obs["lookahead"]`) from the same
scoring pass the action table already runs:

| # | Feature | Why |
|---|---|---|
| 0 | menu available | Separates "no menu here" from "best play scores 0" — ~18% of steps aren't card-select |
| 1 | `log2(best offered value)` | Magnitude, scale-free |
| 2 | `best / chips still needed` | **Does my best play clear this blind right now?** |
| 3 | `best × hands_left / needed` | The multi-hand version — an optimistic upper bound on a plan, not a prediction |
| 4 | `best of any other hand type / best` | Is a *structurally different* play competitive? |
| 5 | `mean offered value / best` | Is the best an outlier or is everything alike? |
| 6 | distinct hand types offered | Menu breadth |
| 7 | cardinality of the best play | Does the best play commit the hand or keep cards back? |

Measured over 2,455 real card-select states: **+1.4% step cost**, every dim
has real spread, nothing saturates except #7 (48% at 5 cards, expected), and
#4 correlates **−0.03** with #2 — it carries genuinely independent
information. An earlier version of #4 used the runner-up's value instead and
measured 1.0 in 75% of states: non-scoring kickers make exact ties the common
case (a pair of 7s scores identically whichever three cards ride along), so
that dim was nearly dead.

**Deliberately not built: "best chips after one optimal discard."** It needs
a draw to evaluate, so it is either RNG-dependent or a full state copy per
candidate — the 377 ms path. The cheap surrogate for it (how close the kept
cards are to a flush or straight) already exists in the global vector at
[227:231], including real draw-out probabilities from the remaining deck.

**Opt-in, and off by default.** Adding an observation key changes the
observation space, which would make every existing checkpoint (runs 5-12, the
entire baseline set) unloadable. `scripts/eval_ppo.py` reads which kind a
checkpoint needs from its own saved observation space, so there is no flag to
get wrong at eval time.

**Level 3 — MCTS / expert iteration.** Out of scope. Balatro is stochastic
(shop, draws, packs), so it needs determinization or Stochastic MuZero. Noted
as a door that exists.

### 5.4 Factored action head (later)

Flat `Discrete(500)` has unstable semantics: index 37 means different things at
different timesteps and the network can't infer the enumeration from the
observation. (Defect #2's fix made the mapping at least *stationary* across
identical states, which it previously wasn't.)

Target architecture, matching the native `FactoredAction`:
- action-type head over 21 types
- pointer head attending over entity embeddings for `entity_target`
- card selection either autoregressive, or delegated entirely to §5.3

`GameSpec` already provides the contract: `entity_type_for_action()`,
`needs_entity_set`, `catalog_size`. This is filling a well-shaped hole, not
designing from scratch. It is also the point at which pooled entity
representations (§5.2) would finally have a consumer.

### 5.5 On the LSTM

- **No `MaskableRecurrentPPO` exists.** sb3-contrib ships `MaskablePPO` and
  `RecurrentPPO` separately; combining them means writing it. The common
  workaround (hard-masking inside the env) throws away correct logit zeroing
  and log-probs.
- **Balatro is closer to fully observed than it looks.** Jokers, vouchers, hand
  levels, money, ante, blind are all observable. The genuinely hidden state is
  mainly the remaining draw pile — and a 52-dim remaining-deck count histogram
  solves that directly, rather than making an LSTM learn card counting through
  BPTT.
- Building a joker synergy across antes is a **credit assignment** problem, not
  a memory problem. The current joker set is already observable. (The embedding
  post-mortem in `RUNS.md` points the same way.)

**Decision: LSTM is a controlled ablation after the factored head, not a
foundation.** Since the factored head already requires a custom policy, the
marginal cost of adding recurrence drops a lot there. If cheap memory is wanted
sooner, add explicit history features (last N actions, cards played this round,
shop items passed on).

---

## 6. Phases

### Phase 0 — Instrument ✅ complete

- [x] Fix defects 3, 5, 6, 7 (seed collision, truncation, Monitor, metrics)
- [x] Profile — confirmed `_enumerate_actions` dominates (§2)
- [x] Heuristic agent with exact one-step evaluation — **mean ante 2.75**
- [x] Frozen eval harness — `eval_seeds.py`, `rollout.py`, `scripts/eval_agent.py`
- [x] Observation golden tests — `tests/env/test_observation_golden.py`

Note on the heuristic's 2.75: zero wins. "Beat Red Deck / White Stake" is not
solved by hand-selection lookahead alone — shop/economy/joker-synergy play
(deliberately trivial in `HeuristicAgent`) is the missing half, and winning
requires surviving ante 8.

### Phase 1 — Representation

- [x] Emit integer catalog IDs from `observation.py` — §5.1
- [x] `BalatroExtractor` with embeddings + masked set pooling — §5.2.
      **Implemented and measured: no benefit at this budget, and the cold-start
      explanation for why was wrong** (§3). Not worth further tuning without a
      consumer for pooled representations (§5.4) or a fix to the credit-
      assignment problem.
- [x] `share_features_extractor=False`
- [x] Deterministic top-K action table — §5.3. *Was* Phase 2's "canonical
      enumeration" item, promoted when it turned out to be a prerequisite for
      Level 2 lookahead features, and it delivered the only confirmed gain.
- [x] **Lookahead features (§5.3 Level 2)** — an 8-dim `obs["lookahead"]`
      channel behind `--lookahead`, on the cheap `evaluate_hand` path,
      reusing the action table's own scoring pass (+1.4% step cost). Two
      seeds: **1.46 and 1.565** vs 1.420/1.420 without, non-overlapping, and
      the first run to reach ante 5. Ablation attributes run 15's gain
      causally to the feature content (−0.19 ante when permuted). Checked
      off as *implemented and beneficial*, with the caveat that only one of
      two seeds learned to exploit it — see `RUNS.md`.
- [ ] 52-dim remaining-deck **count** histogram — blocked on §10's
      `BridgeAdapter` deck-composition question
- [ ] Real joker rarity; drop `ability_extra` (defects 9, 10)
- [x] `SubprocVecEnv` (defect 17) — `--vec-env subproc --n-envs 8`, **3.0x
      end-to-end** (160 → 485 fps; 500k run ~52 min → ~17 min). Worker game
      diversity verified. `--rollout-steps` keeps the rollout constant so
      worker count buys only wall-clock. See `RUNS.md`.

**Exit:** `explained_variance` stops decaying; no NaN crashes; PPO approaches
the heuristic. **Not met** — 1.42 vs 2.75.

### Phase 2 — Action space and scaling

- [ ] Factored action head with entity pointer (§5.4)
- [ ] Shop / economy play — the untouched half of the game, and the only path
      to the ceiling (max ante has never exceeded 4)
- [ ] Curriculum: success = "clear ante N", auto-advance on eval threshold

**Exit:** beats the heuristic.

### Phase 3 — Polish

- [ ] LSTM ablation
- [ ] Optuna sweep (only now — sweeping before the representation is fixed just
      burns compute)
- [ ] Live validation via `BridgeAdapter` on Windows — requires extending
      `bot_state_to_game_state` first (§5.3 finding 2)
- [ ] Writeup

---

## 7. Hyperparameters

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
  `learning_rate`, `clip_range`, `clip_range_vf` with `get_schedule_fn`;
  `ent_coef` is a plain float. Pass a callable and you get a `TypeError`. Use a
  callback that mutates `self.model.ent_coef` in `_on_rollout_start`.
- **LR and `clip_range` schedule on fraction of *declared* `total_timesteps`,
  not absolute steps.** A run with a different `--total-timesteps` is therefore
  *not* a clean extension of a shorter one — it also has a different effective
  LR trajectory (run 9's was >3× run 7's at the same step). Left as-is because
  it's SB3's own convention and changing it is a schedule redesign needing its
  own isolated validation.
- **The schedule denominator must be `total_timesteps`, not `max_steps`.** This
  bug occurred twice. Assert inside the callback constructor, where it guards
  the value actually used.
- **`ent_coef=0.05` was 6× the policy gradient term.** Loss decomposition at run
  2: pg −0.0301, entropy 0.05 × (−3.73) = −0.187, value 0.5 × 0.0017. The
  dominant instruction was "stay random". `0.0005` is past the stability edge.
  Use 0.005 → 0.001.
- **Reward scaling is mostly a no-op on the policy.** `normalize_advantage=True`
  standardizes advantages per minibatch, so a uniform ×10 is normalized away —
  while inflating value MSE ~100×. Use `VecNormalize(norm_reward=True)`.
- **`target_kl=0.02` is the guard that catches slow KL drift.** Off by default.
- `gamma=0.99` gives a ~100-step horizon. Fine now; raise toward 0.999 once
  episodes lengthen past a few hundred steps.

### Reward shaping

```
step:          -0.001  (-0.002 in shop)
blind beaten:  +0.15   (flat — ante scaling removed)
boss beaten:   +0.10
efficient:     +0.01 * hands_left
score progress:+0.02 * min(chip_delta / blind_target, 1.0)
terminal:      +0.5 win / -0.2 loss
stalled:       -1.0   (on top of the terminal loss — defect #15)
```

The original `0.15 * (ante/8)` was backwards: ante 1 paid 0.019 against a −0.2
death penalty, so ten blinds were needed to offset one loss — and since the
agent lost every episode, that −0.2 was a near-constant offset carrying no
gradient. Flat scaling fixed this.

Consider potential-based shaping (`F = γΦ(s') − Φ(s)`) later; it's provably
policy-invariant, unlike the current form.

---

## 8. Experiment discipline

1. **Never change reward shaping and hyperparameters in the same run.** This
   applies to architecture too — run 6 bundled embeddings *and* pooling, and a
   negative result couldn't say which was responsible.
2. Freeze the reward definition; version it if it must change.
3. [x] Checkpoint every 50k steps (`--checkpoint-freq`, default 50,000, into a
   timestamped per-invocation directory).
4. [x] Save `VecNormalize` statistics alongside the model
   (`save_vecnormalize=True`).
5. [x] Compare on the **frozen eval seed set**, never rollout statistics.
6. Pin `mean_ante_reached` to a `[1, 8]` axis. Auto-scaling makes 1.00→1.03
   look like progress.
7. **Confirm any positive result with a second seed** before believing it. The
   noise floor is ≈0.13 mean ante; runs 6 and 10 both looked plausible and
   didn't survive.
8. **Record the run in `RUNS.md`** — what changed, the eval number, the ante
   distribution, and what it ruled in or out.

### How to judge a run

**The only trustworthy number**: `mean_ante` from
`uv run scripts/eval_ppo.py --model <path> --episodes 200`. Compare against
random (1.00), heuristic (2.75), and the previous run. Also look at the ante
*distribution*, not just the mean — a mean pulled up by two lucky seeds is a
very different result from broad improvement:

```bash
python -c "import json,collections; d=json.load(open('results/ppo_runN.json')); print(collections.Counter(r['ante_reached'] for r in d['per_seed']))"
```

**Training-time health** (`tensorboard --logdir runs/balatro_ppo`):
- `train/explained_variance` should trend up and stay non-decaying — this is
  literally what predicted the run 3/4 crashes.
- `train/value_loss` climbing steeply/curving vertical is the other precursor.
- `rollout/ep_rew_mean`/`ep_len_mean` should trend upward, but are
  stochastic-rollout stats — directional sanity checks only.
- `diag/max_abs_param` trending up is a slow weight blowup.
- `time/fps` cratering *mid-run* is a symptom (a stall loop, or a joker-
  generation bug inflating entity counts). `time/fps` differing *between* runs
  is usually just machine load — see §2.

**Calibration.** Win rate staying 0% is not yet a red flag; winning needs ante
8. A run at or below random with otherwise-healthy training curves (as run 6
was) means the *architecture* is wrong, not the training dynamics — check what
information a representation change might be destroying before assuming it
needs more steps.

### Reading the charts

- **Smooth:** `mean_ante_reached`, `ep_len_mean` (~0.6). Real, noisy,
  genuinely averageable trends.
- **Do NOT smooth:** `max_ante_reached` — a per-rollout maximum; maxima don't
  average meaningfully. Prefer a histogram or 90th percentile.
- **Do NOT smooth:** `win_rate` at ~0 — smoothing creates a nonzero-looking
  trace suggesting progress that isn't there.
- Read **raw** for before/after comparisons; EMA lags and shifts apparent onset.
- Variance itself is signal on `value_loss` and `policy_gradient_loss`.

---

## 9. Tooling

| Tool | Use | When |
|---|---|---|
| `scripts/eval_ppo.py` | The comparison number | Every run |
| `scripts/bench_step.py` | Where wall-clock goes, env vs network | When fps moves |
| `scripts/embed_drift.py` | Did an embedding table actually learn | Embedding work |
| `scripts/lookahead_ablation.py` | Is a feature used, or merely wired | **Any** new observation feature |
| **py-spy** | Sampling profiler, flamegraphs | Couldn't attach on Windows in this sandbox — `cProfile` substituted |
| **Weights & Biases** | Experiment tracking; a public report is a good portfolio artifact | Optional |
| **CleanRL** | Single-file PPO you own. `ppo_multidiscrete_mask.py` (Gym-µRTS) is the closest reference to a factored, masked, entity-based action space | Phase 2 |
| **PufferLib** / **Sample Factory** | Faster vectorization / async PPO for CPU-bound envs | If throughput-starved after `SubprocVecEnv` |
| **Optuna** | Hyperparameter search | Phase 3 only |

**Move off SB3 at Phase 2, not before.** SB3 handles the custom feature
extractor fine. Once a factored pointer head with conditional cross-head
masking is needed — plus possibly recurrence, which sb3-contrib can't combine
with masking — fighting SB3's abstractions costs more than owning the loop.

Reading: Huang et al., *A Closer Look at Invalid Action Masking in Policy
Gradient Algorithms* (arXiv 2006.14171) — why masked log-probs matter.

---

## 10. Open questions

- [ ] Does `BridgeAdapter` expose remaining-deck composition? The live game
      lets players inspect it, so it should be legitimate — needs verification
      before relying on it in the observation.
- [ ] Can `scoring.py` evaluate a candidate hand with joker effects but without
      a full state copy? The cheap `evaluate_hand` path (§5.3) is joker-*blind*,
      which is what costs the residual 5% of recall@10. `_handle_play_hand`
      populates ~a dozen synthetic snapshot fields before calling `score_hand`;
      a synthetic-snapshot path would need to reproduce those without
      duplicating engine bookkeeping.
- [ ] What are real hand-size and joker-slot maxima in practice? `max_count` is
      currently 20/10, an estimate rather than a value derived from real
      hand-size/joker-count data. An empirical pass (Juggler / Turtle Bean /
      voucher stacking) could tighten or raise these.
- [ ] Why do the embedding tables move only ~3% of a row's length per 500k
      steps when 95% of rows receive gradient? Credit assignment is the
      hypothesis; an auxiliary prediction loss on joker identity would test it
      cheaply.

### Answered

- **Does `_enumerate_actions` dominate the profile?** Yes — 92% of env
  wall-clock (§2).
- **What is the actual cost of `deepcopy(raw_state)`?** It dominates exact
  lookahead: 342 ms of 377 ms per decision, 91% (§5.3).
- **Are the joker embeddings starved of exposure?** No — 95% of the reachable
  joker catalog receives gradient, both before and after the action-table fix.
  See `RUNS.md`.
