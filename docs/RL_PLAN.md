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

### Crash diagnosis

Both crashes: `ValueError: ... probs ... to satisfy the constraint Simplex()`.
NaN in the logits reaching `MaskableCategorical`.

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

### Not a defect

`_log_scale` (`observation.py:295-299`) is sign-symmetric and safe for negative
dollars. Ruled out as a NaN source.

`max_steps=10_000` never binds — episodes are ~20 steps and ~280 complete per
rollout, which is good for credit assignment. Set it to 2,000 as a loop guard,
not for throughput.

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
- [ ] Emit integer catalog IDs from `observation.py`
- [ ] `BalatroExtractor` with embeddings + masked set pooling
- [ ] `share_features_extractor=False`
- [ ] 52-dim remaining-deck **count** histogram
- [ ] Lookahead features (§5.3 Level 2)
- [ ] Real joker rarity; drop `ability_extra`
- [ ] `SubprocVecEnv` × 6 on Linux with distinct per-worker seed prefixes

**Exit:** `explained_variance` stops decaying; no NaN crashes; PPO approaches
the heuristic.

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
3. Checkpoint every 50k steps —
   `CheckpointCallback(save_freq=50_000, save_path=...)`. Run 3 lost 364k steps
   of training to a crash with no checkpoint.
4. Save `VecNormalize` statistics alongside the model, or evaluation is wrong.
5. Compare on the **frozen eval seed set**, not on rollout statistics (noisy,
   confounded by exploration).
6. Pin `mean_ante_reached` to a `[1, 8]` axis. Auto-scaling makes a 1.00→1.03
   change look like progress.

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
