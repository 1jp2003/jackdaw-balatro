"""Gymnasium wrapper for BalatroEnvironment compatible with SB3's MaskablePPO.

Flattens the factored action space into a ``Discrete(MAX_ACTIONS)`` space by
enumerating all legal :class:`FactoredAction` instances each step.  Exposes an
``action_masks()`` method so ``sb3_contrib.MaskablePPO`` can mask illegal slots.

Example::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    obs, info = env.reset()
    mask = env.action_masks()
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from itertools import combinations
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.engine.data.hands import HAND_BASE, HandType
from jackdaw.engine.hand_eval import evaluate_hand
from jackdaw.env.action_space import ActionType, get_consumable_target_info
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation
from jackdaw.env.observation import NUM_HAND_TYPES, log_scale

MAX_ACTIONS: int = 500
# Consumable card-targeting still uses plain enumeration; kept small so the
# global MAX_ACTIONS cap stays unreachable in normal play.
CARD_COMBO_BUDGET: int = 20
# PlayHand/Discard menu size. K=10 puts the exactly-best play in the menu
# 95% of the time (docs/RL_PLAN.md §5.3, recall@K table); 12 buys a little
# headroom for the diversity reservations below. Fixed size regardless of
# hand size, which matters because Balatro has no hard hand-size cap.
PLAY_COMBO_BUDGET: int = 12
# Slots reserved for the smallest-cardinality combos in every menu.
_MIN_CARDINALITY_SLOTS: int = 2
# Width of the optional lookahead observation channel (`_lookahead_features`).
LOOKAHEAD_DIM: int = 8
# Divisor on the block's one magnitude field (index 1). 1.0 = off.
# Setting it to 10.0 to match the seven ratio fields was tried and did NOT
# help — see `_lookahead_features`. Kept as a knob for further study, in the
# same spirit as `BalatroExtractor.embed_init_std`.
_LOOKAHEAD_VALUE_SCALE: float = 1.0


def _stable_int_seed(text: str) -> int:
    """Deterministic non-negative int derived from a string.

    NOT Python's builtin ``hash()`` — that's randomized per-process by
    default (``PYTHONHASHSEED``), which would silently make anything seeded
    from it non-reproducible across runs/processes. ``zlib.crc32`` is
    deterministic and stable across platforms and Python versions.
    """
    return zlib.crc32(text.encode())


# Pre-compute entity layout from spec
_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
# Catalog-ID entity types only (joker/consumable/shop_item) — the
# embedding-lookup channels from docs/RL_PLAN.md Sec 5.1, keyed as
# f"{name}_ids" in the observation Dict, shape (max_count,) int.
_CATALOG_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.catalog_size) for et in _SPEC.entity_types if et.has_catalog_id
]
# Action types that take no targets at all
_SIMPLE_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
# Entity-only action types (entity target, no card select) — excludes UseConsumable
_ENTITY_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
# Card-only action types (card select, no entity) — PlayHand, Discard
_CARD_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)


def _subsample(items: list[Any], budget: int, rng: np.random.Generator) -> list[Any]:
    """Return *items* if within budget, else a random subsample.

    Retained only as a last-resort guard on the global MAX_ACTIONS cap.
    PlayHand/Discard no longer use it — see `_select_card_combos`, and
    known issue #2 for why random subsampling was the wrong tool.
    """
    if len(items) <= budget:
        return items
    indices = rng.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]


def _cheap_hand_value(cards: list[Any], jokers: list[Any], hand_levels: Any) -> tuple[float, str]:
    """Cheap, RNG-free value of playing *cards*: ``(score, hand_type)``.

    Deliberately joker-blind: uses the engine's own `evaluate_hand` for hand
    detection plus that hand type's chips/mult at its *current level*, and
    the scoring cards' chip values. ~0.016 ms per candidate versus ~1.73 ms
    for a full `deepcopy`+`engine.step` — 107x cheaper, which is what makes
    it affordable to rank every candidate on every step.

    Accuracy is sufficient for *ranking*, which is all the action table
    needs: measured over 60 real states, the exactly-best play (by full
    engine evaluation) lands in this ranking's top 10 **95%** of the time.
    The residual ~5% are hands where joker effects dominate and a
    joker-blind ranker misjudges — adding more candidates does not fix
    those (recall is flat from K=10 to K=50); joker-aware cheap scoring
    would. See docs/RL_PLAN.md §5.3.
    """
    result = evaluate_hand(cards, jokers)
    try:
        hand_type = HandType(result.detected_hand)
    except ValueError:
        return 0.0, ""
    if hand_levels is not None:
        chips, mult = hand_levels.get(hand_type)
    else:
        base = HAND_BASE[hand_type]
        chips, mult = base.s_chips, base.s_mult
    for card in result.scoring_cards:
        chips += card.get_chip_bonus() + card.get_nominal()
    return float(chips) * float(mult), str(hand_type)


def _score_card_combos(
    combos: list[tuple[int, ...]],
    hand: list[Any],
    jokers: list[Any],
    hand_levels: Any,
) -> list[tuple[float, str, tuple[int, ...]]]:
    """Score every combo once as ``(value, hand_type, combo)``.

    Computed once per step and shared by both PlayHand and Discard, which
    rank the same candidate set in opposite directions — scoring it twice
    would double the only meaningful cost this adds to `step()`.
    """
    return [
        (*_cheap_hand_value([hand[i] for i in combo], jokers, hand_levels), combo)
        for combo in combos
    ]


def _select_card_combos(
    combos: list[tuple[int, ...]],
    scored: list[tuple[float, str, tuple[int, ...]]],
    budget: int,
    *,
    prefer_high: bool,
) -> list[tuple[int, ...]]:
    """Deterministically pick <= *budget* combos, best-first with diversity.

    Replaces random subsampling (known issue #2), which dropped the optimal
    play 8.3% of the time on an 8-card hand and 68.6% at 10 cards, and gave
    identical states different menus from one visit to the next.

    ``prefer_high`` ranks by descending value (PlayHand — play your best
    hand). ``False`` ranks ascending (Discard — throw away your least
    valuable cards). The discard ordering is the weaker of the two
    heuristics: it ignores that discarding is really about what you *keep*
    and what you might draw. It is a menu, not a decision — the policy still
    chooses among what's offered — but it is worth revisiting.

    Two diversity reservations, because a pure top-by-score menu would
    quietly bound what the policy can ever learn:

    - **one slot per distinct hand type**, so a flush option survives even
      in a hand where pairs currently score higher;
    - **the smallest-cardinality combos**, which are the setup / draw-
      preserving plays. These score terribly right now by construction, so
      a greedy menu prunes exactly the moves a multi-hand plan needs, and
      an action never offered is one the policy can never discover.
    """
    if len(combos) <= budget:
        return sorted(combos)

    # Sort by value, tie-broken on the combo itself so the order is total
    # and identical states always produce an identical menu.
    scored = sorted(scored, key=lambda item: (-item[0] if prefer_high else item[0], item[2]))

    chosen: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def take(combo: tuple[int, ...]) -> None:
        if combo not in seen and len(chosen) < budget:
            seen.add(combo)
            chosen.append(combo)

    # Reserve: best of each hand type, and the smallest-cardinality options.
    best_per_type: dict[str, tuple[int, ...]] = {}
    for _value, hand_type, combo in scored:
        if hand_type and hand_type not in best_per_type:
            best_per_type[hand_type] = combo
    smallest = min(len(c) for c in combos)
    reserved_small = sorted(c for c in combos if len(c) == smallest)[:_MIN_CARDINALITY_SLOTS]

    for combo in reserved_small:
        take(combo)
    for combo in best_per_type.values():
        take(combo)
    for _value, _hand_type, combo in scored:
        take(combo)

    return sorted(chosen)


def _lookahead_features(
    scored: list[tuple[float, str, tuple[int, ...]]],
    gs: dict[str, Any],
) -> np.ndarray:
    """Describe the *menu* the policy is about to choose from (Sec 5.3 Level 2).

    The global vector already describes the *hand*: the best hand type
    detectable across all 8 cards, its base chips/mult at level, flush and
    straight proximity, draw outs, and an urgency ratio (`observation.py`
    [211:235]). What it has never contained is anything about the choice
    actually on offer — the value of the best *subset* the action table will
    let the policy play, which differs from the hand-level features in two
    ways that matter: it includes the scoring cards' own chip contributions,
    and it is a max over the specific combos being offered rather than a
    property of the whole hand.

    That distinction only became coherent once the action table stopped
    randomly subsampling (known issue #2). Describing "the best achievable
    score is X" while the play achieving X is missing from the menu 8-69% of
    the time was incoherent; now the ranked top-K menu and these features are
    computed from the same scoring pass, so the feature always refers to
    something the policy can actually select.

    Cost is ~zero: `scored` is the same list `_select_card_combos` consumes.

    Every field is bounded, and index 0 flags whether the rest is meaningful
    at all — outside SELECTING_HAND there is no menu, and "no data" must not
    be confusable with "a menu whose best play scores 0".

    Field scale — tried balancing it, it did not help
    -------------------------------------------------
    `VecNormalize` runs with ``norm_obs=False``, so these values reach the
    network raw and the first Linear computes ``sum(weight * value)``. A
    field's units therefore act as an importance weight before any learning
    happens. Measured on run 13's trained checkpoint, all eight dims carried
    near-equal weights (~0.08-0.10), so raw magnitude decided everything:
    index 1 contributed ``0.082 * 7.16 = 0.59`` to the pre-activation while
    index 2 — "does my best play clear the blind", the most decision-relevant
    number in the block — contributed ``0.104 * 0.33 = 0.034``. A 17x
    imbalance in favour of the *least* useful field, purely from units.

    The obvious inference — divide index 1 so every field lands near [0,1]
    and let the network weight them by usefulness — was tested in runs 16/17
    (`_LOOKAHEAD_VALUE_SCALE = 10.0`, both seeds, nothing else changed) and
    **came out worse**: seed 1 fell 1.565 -> 1.450, seed 0 moved 1.460 ->
    1.470, group mean 1.513 -> 1.460.

    The ablation says why. Permuting the block cost unscaled seed 1 **0.19
    ante**, but costs either rescaled run ~nothing — and the zeroed condition
    went from 1.02 to ~1.27, meaning the block matters *less* overall after
    rescaling, not differently. Shrinking the loud field did not hand its
    influence to the quiet ones; it just made the whole block quieter, and
    the network did not grow compensating weights within 500k steps (the LR
    anneals to ~0).

    This is the same result, and the same wrong reasoning, as
    `BalatroExtractor.embed_init_std` (run 10): "this input is
    disproportionately loud, so quieting it will let the useful signal
    through" has now failed twice in this codebase. Both times, reducing an
    input's magnitude reduced its total contribution rather than rebalancing
    it. Treat that inference as suspect here; if field scale is worth
    attacking, `VecNormalize(norm_obs=True)` addresses it globally and is a
    different experiment.
    """
    out = np.zeros(LOOKAHEAD_DIM, dtype=np.float32)
    if not scored:
        return out

    values = [value for value, _hand_type, _combo in scored]
    best = max(values)
    _best_value, best_type, best_combo = max(scored, key=lambda item: (item[0], item[2]))

    blind = gs.get("blind")
    blind_chips = getattr(blind, "chips", 0) if blind is not None else 0
    remaining = max(int(blind_chips) - int(gs.get("chips", 0)), 0)
    hands_left = int(gs.get("current_round", {}).get("hands_left", 0))

    out[0] = 1.0
    out[1] = log_scale(best) / _LOOKAHEAD_VALUE_SCALE
    if remaining > 0:
        # "Does my single best offered play finish this blind right now?"
        out[2] = min(best / remaining, 2.0) / 2.0
        # The multi-hand version: could I still get there if every hand I
        # have left were as good as this one? Deliberately optimistic — it
        # is an upper bound on a plan, not a prediction. This is the same
        # quantity as the global vector's urgency [234] but computed from
        # the real menu rather than base hand value, and oriented so that
        # higher is better.
        out[3] = min(best * max(hands_left, 1) / remaining, 2.0) / 2.0
    elif blind_chips > 0:
        # Blind already met — both ratios saturate rather than divide by 0.
        out[2] = out[3] = 1.0

    if best > 0:
        # How much does the choice actually matter? Not "value of the
        # runner-up" — non-scoring kickers make exact ties the common case
        # (a pair of 7s scores the same whichever three cards ride along),
        # so that ratio measured 1.0 in 75% of real states and carried
        # almost no information. The live question is whether a
        # *structurally different* play is competitive, so this is the best
        # play of any other hand type, relative to the best overall.
        best_other = max(
            (value for value, hand_type, _c in scored if hand_type and hand_type != best_type),
            default=0.0,
        )
        out[4] = min(best_other / best, 1.0)
        out[5] = min((sum(values) / len(values)) / best, 1.0)

    distinct_types = {hand_type for _v, hand_type, _c in scored if hand_type}
    out[6] = min(len(distinct_types) / NUM_HAND_TYPES, 1.0)
    # Whether the best play commits the whole hand or keeps cards back.
    out[7] = min(len(best_combo) / 5.0, 1.0)
    return out


def _card_combos(
    legal_cards: np.ndarray,
    min_select: int,
    max_select: int,
) -> list[tuple[int, ...]]:
    """Enumerate all card-index combinations within the selection range."""
    upper = min(len(legal_cards), max_select)
    lower = min(min_select, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result


class BalatroGymnasiumEnv(gymnasium.Env):
    """Gymnasium wrapper exposing a flat Discrete action space with action masking.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter`.
    back_keys, stakes, max_steps, seed_prefix:
        Forwarded to :class:`BalatroEnvironment`.
    reward_shaping:
        If True, use dense multi-signal reward: blind-beaten bonuses scaled
        by ante, score-progress within blinds, efficient-clear bonuses, and
        reduced terminal rewards.  When False, use sparse ±1.0 at terminal.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # Known issue #15: a deterministic policy can get stuck spamming a
    # zero-cost action (observed: SwapHandLeft, 1992/2000 steps on one eval
    # seed) that never changes chips/round/ante/hands/discards/dollars.
    # Detected generically via _progress_fingerprint rather than by
    # excluding specific "cosmetic" action types, so it also catches a
    # policy alternating between two different non-progressing actions.
    _STALL_STEPS_LIMIT: int = 20
    _STALL_PENALTY: float = -1.0

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
        reward_shaping: bool = False,
        lookahead_features: bool = False,
    ) -> None:
        super().__init__()
        self._inner = BalatroEnvironment(
            adapter_factory=adapter_factory,
            back_keys=back_keys,
            stakes=stakes,
            max_steps=max_steps,
            seed_prefix=seed_prefix,
        )
        self._reward_shaping = reward_shaping
        self._lookahead_features = lookahead_features

        # Observation space
        obs_spaces: dict[str, spaces.Space] = {
            "global": spaces.Box(
                -np.inf, np.inf, shape=(_SPEC.global_feature_dim,), dtype=np.float32
            ),
        }
        max_counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            obs_spaces[name] = spaces.Box(
                -np.inf, np.inf, shape=(max_count, feat_dim), dtype=np.float32
            )
            max_counts.append(max_count)
        obs_spaces["entity_counts"] = spaces.Box(
            low=0,
            high=np.array(max_counts, dtype=np.float32),
            shape=(len(_ENTITY_INFO),),
            dtype=np.float32,
        )
        # Catalog-ID channels (docs/RL_PLAN.md Sec 5.1) — 0 is reserved for
        # unknown/padding, valid IDs run 1..catalog_size inclusive, so the
        # per-slot range is [0, catalog_size]. Box, not MultiDiscrete: SB3's
        # preprocess_obs one-hot-encodes MultiDiscrete/Discrete spaces
        # before a features extractor ever sees them (300-way one-hot per
        # slot here), which is exactly the dense representation embeddings
        # exist to avoid. Box passes through as a plain float tensor holding
        # the integer value, which BalatroExtractor casts back with .long()
        # for the nn.Embedding lookup.
        for name, max_count, catalog_size in _CATALOG_ENTITY_INFO:
            obs_spaces[f"{name}_ids"] = spaces.Box(
                low=0, high=catalog_size, shape=(max_count,), dtype=np.int64
            )
        # Lookahead channel (docs/RL_PLAN.md §5.3 Level 2) — opt-in, because
        # adding an observation key changes the observation space and every
        # checkpoint trained without it (runs 5-12, the entire baseline set)
        # would stop loading. Off by default so those stay evaluable;
        # scripts/eval_ppo.py detects which kind a checkpoint needs from its
        # own saved observation space rather than requiring a matching flag.
        if lookahead_features:
            obs_spaces["lookahead"] = spaces.Box(
                low=0.0, high=np.inf, shape=(LOOKAHEAD_DIM,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)

        # Action space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # Internal state
        self._action_table: list[FactoredAction] = []
        self._rng = np.random.default_rng()
        self._prev_ante: int = 1
        self._prev_round: int = 0
        self._prev_chips: int = 0
        self._episode_max_ante: int = 1
        self._episode_max_round: int = 0
        self._stall_fingerprint: tuple[Any, ...] | None = None
        self._stall_steps: int = 0
        # Written by _enumerate_actions, read by _build_obs. The two always
        # run in that order (both reset() and step() enumerate first), which
        # is what lets the features be free: they are a summary of the
        # scoring pass the action table already had to do.
        self._lookahead: np.ndarray = np.zeros(LOOKAHEAD_DIM, dtype=np.float32)

    @staticmethod
    def _progress_fingerprint(gs: dict[str, Any]) -> tuple[Any, ...]:
        """The subset of game state that constitutes real progress.

        Anything NOT in this tuple (hand/joker order, which shop items are
        highlighted, etc.) is free to change without resetting the stall
        counter — only actions that spend a resource or move the score
        count as progress.
        """
        cr = gs.get("current_round", {})
        return (
            gs.get("chips", 0),
            gs.get("round", 0),
            gs.get("round_resets", {}).get("ante", 1),
            cr.get("hands_left", 0),
            cr.get("discards_left", 0),
            gs.get("dollars", 0),
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Gymnasium's `seed` param is typed int-only, but the underlying
        # Balatro game seed is a string (frozen eval sets, oracle seeds,
        # etc. are named strings like "EVAL_003"). `options["game_seed"]`
        # is the Gymnasium-idiomatic way to pass that through without
        # abusing `seed`'s type — it takes priority over `seed` when both
        # are given.
        #
        # Whichever seed is used must ALSO reseed the action-table
        # subsampling RNG (self._rng, consumed by _enumerate_actions when
        # legal actions exceed MAX_ACTIONS) — a game_seed-only reset that
        # left self._rng on OS entropy would silently make evaluation on a
        # "frozen" seed non-reproducible across runs whenever subsampling
        # kicks in, defeating the entire point of a string game_seed for
        # eval. Derive a stable int from the string (not Python's hash(),
        # which is itself randomized per-process by default).
        game_seed = (options or {}).get("game_seed")
        kwargs: dict[str, Any] = {}
        if game_seed is not None:
            self._rng = np.random.default_rng(_stable_int_seed(str(game_seed)))
            kwargs["seed"] = str(game_seed)
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
            kwargs["seed"] = str(seed)

        game_obs, game_mask, info = self._inner.reset(**kwargs)
        self._prev_ante = self._inner.episode_ante
        self._prev_round = 0
        self._prev_chips = 0
        self._episode_max_ante = 1
        self._episode_max_round = 0
        self._stall_fingerprint = self._progress_fingerprint(info.get("raw_state", {}))
        self._stall_steps = 0
        self._action_table = self._enumerate_actions(game_mask, info)
        obs = self._build_obs(game_obs)
        return obs, {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        factored = self._action_table[action]
        game_obs, terminated, truncated, game_mask, info = self._inner.step(factored)

        # Known issue #15: force-truncate a policy that's spending steps
        # without changing any real game resource, before the reward is
        # computed — so the forced truncation still gets the normal
        # terminal-loss reward from _compute_reward (dense mode's -0.2/-0.5
        # bonus and sparse mode's -1.0), on top of which _STALL_PENALTY is
        # added below to distinguish "lost by stalling" from "lost normally".
        fingerprint = self._progress_fingerprint(info.get("raw_state", {}))
        if fingerprint == self._stall_fingerprint:
            self._stall_steps += 1
        else:
            self._stall_steps = 0
            self._stall_fingerprint = fingerprint
        stalled = self._stall_steps >= self._STALL_STEPS_LIMIT
        if stalled and not (terminated or truncated):
            truncated = True

        reward = self._compute_reward(info, terminated, truncated)
        if stalled:
            reward += self._STALL_PENALTY

        # Rebuild action table for next step
        if not (terminated or truncated):
            self._action_table = self._enumerate_actions(game_mask, info)
        else:
            self._action_table = []
            # No enumeration ran, so the lookahead block would otherwise
            # carry the previous step's menu into a terminal observation.
            self._lookahead = np.zeros(LOOKAHEAD_DIM, dtype=np.float32)

        obs = self._build_obs(game_obs)
        step_info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            step_info["balatro/ante_reached"] = self._episode_max_ante
            step_info["balatro/rounds_beaten"] = self._episode_max_round
            step_info["balatro/won"] = self._inner.episode_won
        return obs, reward, terminated, truncated, step_info

    def action_masks(self) -> np.ndarray:
        """Return bool mask of shape ``(MAX_ACTIONS,)`` for MaskablePPO."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: len(self._action_table)] = True
        return mask

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(self, info: dict[str, Any], terminated: bool, truncated: bool) -> float:
        """Compute step reward from game state deltas."""
        gs: dict[str, Any] = info.get("raw_state", {})
        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)

        # Episode-max trackers must update every step regardless of reward
        # mode — they feed step_info["balatro/ante_reached"]/"rounds_beaten"
        # at terminal. Previously this lived only in the dense-shaping path
        # below, so sparse mode (the default) always reported ante_reached=1
        # and rounds_beaten=0 (known issue #7).
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

        if not self._reward_shaping:
            if terminated or truncated:
                return 1.0 if self._inner.episode_won else -1.0
            return 0.0

        phase = gs.get("phase")

        # Step cost — discourages stalling; doubled in shop phase
        reward = -0.002 if phase == "shop" else -0.001

        # 1. Blind beaten: round increased → +0.15 * ante_scale
        if round_num > self._prev_round:
            reward += 0.15
            # 2. Boss blind beaten (ante increased) → extra +0.1 * ante_scale
            if ante > self._prev_ante:
                reward += 0.1
            # 3. Efficient clear: hands remaining bonus
            hands_left = gs.get("current_round", {}).get("hands_left", 0)
            reward += 0.01 * hands_left

        # 4. Score progress within a blind: chips gained toward target
        blind = gs.get("blind")
        blind_target = getattr(blind, "chips", 0) if blind is not None else 0
        if blind_target > 0 and chips > self._prev_chips:
            chip_delta = chips - self._prev_chips
            reward += 0.02 * min(chip_delta / blind_target, 1.0)

        # 5. Terminal
        if terminated or truncated:
            reward += 0.5 if self._inner.episode_won else -0.2

        # Update trackers
        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips

        return reward

    def _build_obs(self, game_obs: GameObservation) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {
            "global": game_obs.global_context.astype(np.float32),
        }
        counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            arr = game_obs.entities.get(name)
            padded = np.zeros((max_count, feat_dim), dtype=np.float32)
            if arr is not None and arr.shape[0] > 0:
                n = min(arr.shape[0], max_count)
                padded[:n] = arr[:n]
                counts.append(n)
            else:
                counts.append(0)
            obs[name] = padded
        obs["entity_counts"] = np.array(counts, dtype=np.float32)

        # Catalog-ID channels — 0 (unknown/padding) fills slots beyond the
        # real entity count, same convention center_key_id() already uses.
        for name, max_count, _catalog_size in _CATALOG_ENTITY_INFO:
            ids = game_obs.entity_ids.get(name)
            padded_ids = np.zeros((max_count,), dtype=np.int64)
            if ids is not None and ids.shape[0] > 0:
                n = min(ids.shape[0], max_count)
                padded_ids[:n] = ids[:n]
            obs[f"{name}_ids"] = padded_ids

        if self._lookahead_features:
            obs["lookahead"] = self._lookahead

        return obs

    def _enumerate_actions(
        self, mask: GameActionMask, info: dict[str, Any]
    ) -> list[FactoredAction]:
        actions: list[FactoredAction] = []
        type_mask = mask.type_mask
        raw_state = info.get("raw_state", {})
        hand = raw_state.get("hand", [])
        jokers = raw_state.get("jokers", [])
        hand_levels = raw_state.get("hand_levels")

        # PlayHand and Discard rank the same candidate set in opposite
        # directions, so build and score it once for both.
        card_combos: list[tuple[int, ...]] = []
        card_combo_scores: list[tuple[float, str, tuple[int, ...]]] = []
        if any(type_mask[t] for t in _CARD_ONLY_TYPES):
            card_combos = _card_combos(
                np.nonzero(mask.card_mask)[0], mask.min_card_select, mask.max_card_select
            )
            # Ranking is only needed when the menu must be trimmed, but the
            # lookahead features need the scores in every card-select state —
            # including the small ones, which are also the cheap ones.
            if hand and (len(card_combos) > PLAY_COMBO_BUDGET or self._lookahead_features):
                card_combo_scores = _score_card_combos(card_combos, hand, jokers, hand_levels)

        self._lookahead = (
            _lookahead_features(card_combo_scores, raw_state)
            if self._lookahead_features
            else np.zeros(LOOKAHEAD_DIM, dtype=np.float32)
        )

        for t in range(len(type_mask)):
            if not type_mask[t]:
                continue

            # --- Simple (no targets) ---
            if t in _SIMPLE_TYPES:
                actions.append(FactoredAction(action_type=t))

            # --- Entity-only ---
            elif t in _ENTITY_ONLY_TYPES:
                if t in mask.entity_masks:
                    legal = np.nonzero(mask.entity_masks[t])[0]
                    for idx in legal:
                        actions.append(FactoredAction(action_type=t, entity_target=int(idx)))

            # --- Card-only (PlayHand / Discard) ---
            elif t in _CARD_ONLY_TYPES:
                if card_combo_scores:
                    combos = _select_card_combos(
                        card_combos,
                        card_combo_scores,
                        PLAY_COMBO_BUDGET,
                        # Play your best hand; discard your least valuable cards.
                        prefer_high=(t == ActionType.PlayHand),
                    )
                else:
                    # Few enough candidates to offer them all, or no hand to
                    # rank against (defensive).
                    combos = sorted(card_combos)[:PLAY_COMBO_BUDGET]
                for combo in combos:
                    actions.append(FactoredAction(action_type=t, card_target=combo))

            # --- UseConsumable (entity + optional cards) ---
            elif t == ActionType.UseConsumable:
                if t in mask.entity_masks:
                    legal_entities = np.nonzero(mask.entity_masks[t])[0]
                    raw_consumables = info.get("raw_state", {}).get("consumables", [])
                    for c_idx in legal_entities:
                        c_idx_int = int(c_idx)
                        if c_idx_int < len(raw_consumables):
                            card = raw_consumables[c_idx_int]
                            min_cards, max_cards, needs = get_consumable_target_info(card)
                            if needs:
                                legal_cards = np.nonzero(mask.card_mask)[0]
                                combos = _card_combos(legal_cards, min_cards, max_cards)
                                # Deterministic slice in canonical order: a
                                # consumable's best targets depend on its own
                                # effect, which _cheap_hand_value doesn't
                                # model, so there's nothing meaningful to
                                # rank by here — but it still must not vary
                                # between visits to the same state.
                                combos = sorted(combos)[:CARD_COMBO_BUDGET]
                                for combo in combos:
                                    actions.append(
                                        FactoredAction(
                                            action_type=t,
                                            entity_target=c_idx_int,
                                            card_target=combo,
                                        )
                                    )
                            else:
                                actions.append(
                                    FactoredAction(action_type=t, entity_target=c_idx_int)
                                )

        # Global cap. With per-type budgets above this should now be
        # unreachable in normal play (PlayHand + Discard = 2 x
        # PLAY_COMBO_BUDGET, consumable targeting <= CARD_COMBO_BUDGET each),
        # so this is a guard, not a routine path. Truncate rather than
        # random-sample: a deterministic menu is the entire point of the
        # change, and silently varying the menu between visits to the same
        # state is what known issue #2 was.
        if len(actions) > MAX_ACTIONS:
            actions = actions[:MAX_ACTIONS]

        return actions
