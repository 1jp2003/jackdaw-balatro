"""Scripted heuristic baseline agent.

Per ``docs/RL_PLAN.md`` Sec 5.3 (Level 1): the engine is deterministic and a
pure-per-copy state transition, so the true chip value of every legal
``PlayHand`` can be found exactly — deep-copy the state, run the real engine
on each candidate, and keep the highest score. This is *exact*, not
approximate: candidates are scored by running ``jackdaw.engine.game.step``
itself (not a re-implementation of scoring), so lookahead results are
bit-identical to actually playing that hand.

Deliberately not done here (documented follow-ups, not built speculatively):

- Discards are not solvable by one-step lookahead (a discard scores nothing;
  its value only appears on the next draw). ``docs/RL_PLAN.md`` proposes
  reusing an external ``flush_bot.py`` discard strategy that does not exist
  in this repo, so :meth:`HeuristicAgent._choose_discard` is a small
  hand-written near-flush/near-straight heuristic instead.
- If profiling ``scripts/eval_agent.py`` shows ``copy.deepcopy`` dominates
  per-decision cost, an optimization exists but is not implemented: score
  candidates via ``scoring.score_hand``/``score_hand_base`` with a small
  synthetic snapshot dict instead of a full state deep-copy (see
  ``docs/RL_PLAN.md`` Sec 10). Not built now because it would duplicate the
  snapshot bookkeeping ``game.py::_handle_play_hand`` performs before
  calling ``score_hand``, risking drift between the two paths.
- Shop/pack/joker-ordering decisions are intentionally trivial (see
  ``_act_shop``) — this baseline is scoped to hand-selection quality, not
  economy or joker-synergy play.
"""

from __future__ import annotations

import copy
from itertools import combinations
from typing import Any

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.game import IllegalActionError
from jackdaw.engine.game import step as engine_step
from jackdaw.engine.hand_eval import is_suit
from jackdaw.env.action_space import (
    ActionMask,
    ActionType,
    FactoredAction,
    factored_to_engine_action,
)

_SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")

# Discard the best-found hand instead of playing it when it's this weak
# (and it's not the last hand of the round — see _act_selecting_hand).
_DISCARD_WORTHY = {HandType.HIGH_CARD, HandType.PAIR}


class HeuristicAgent:
    """Baseline agent: exact one-step ``PlayHand`` lookahead + simple defaults.

    Satisfies the :class:`~jackdaw.env.agents.Agent` protocol. Reference
    baseline per ``docs/RL_PLAN.md`` Sec 5.3 — "the RL agent is not
    interesting until it beats this."
    """

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        gs = info["raw_state"]
        phase = gs.get("phase")
        if isinstance(phase, str):
            phase = GamePhase(phase)

        if phase == GamePhase.SELECTING_HAND:
            return self._act_selecting_hand(gs, action_mask)

        if phase == GamePhase.BLIND_SELECT:
            if action_mask.type_mask[ActionType.SelectBlind]:
                return FactoredAction(action_type=ActionType.SelectBlind)
            return self._fallback(action_mask)

        if phase == GamePhase.ROUND_EVAL:
            if action_mask.type_mask[ActionType.CashOut]:
                return FactoredAction(action_type=ActionType.CashOut)
            return self._fallback(action_mask)

        if phase == GamePhase.SHOP:
            return self._act_shop(gs, action_mask)

        if phase == GamePhase.PACK_OPENING:
            if action_mask.type_mask[ActionType.SkipPack]:
                return FactoredAction(action_type=ActionType.SkipPack)
            return self._fallback(action_mask)

        # GAME_OVER or an unrecognized phase — episode is ending or in an
        # unanticipated state; never crash.
        return self._fallback(action_mask)

    # ------------------------------------------------------------------
    # SELECTING_HAND: exact PlayHand lookahead + discard decision
    # ------------------------------------------------------------------

    def _act_selecting_hand(self, gs: dict[str, Any], action_mask: ActionMask) -> FactoredAction:
        can_play = bool(action_mask.type_mask[ActionType.PlayHand])
        cr = gs.get("current_round", {})
        can_discard = (
            bool(action_mask.type_mask[ActionType.Discard]) and cr.get("discards_left", 0) > 0
        )
        hands_left = cr.get("hands_left", 0)

        best_subset: tuple[int, ...] | None = None
        best_score = -1
        best_type: str | None = None
        for subset in self._candidate_subsets(action_mask):
            scored = self._score_subset(gs, subset)
            if scored is None:
                continue
            score, hand_type = scored
            if score > best_score:
                best_subset, best_score, best_type = subset, score, hand_type

        if best_subset is None:
            # Every candidate was illegal to actually step (e.g. Cerulean
            # Bell's forced-selection card excluded from all subsets we
            # tried, or hands_left hit 0 between mask-build and here).
            return self._fallback(action_mask)

        # Discarding scores nothing — never discard on the last hand, that
        # would waste the only remaining scoring chance this round.
        if can_discard and hands_left > 1 and (not can_play or best_type in _DISCARD_WORTHY):
            discard_action = self._choose_discard(gs, best_subset, action_mask)
            if discard_action is not None:
                return discard_action

        if can_play:
            return FactoredAction(action_type=ActionType.PlayHand, card_target=best_subset)

        return self._fallback(action_mask)

    def _candidate_subsets(self, action_mask: ActionMask) -> list[tuple[int, ...]]:
        legal = np.nonzero(action_mask.card_mask)[0].tolist()
        lo = action_mask.min_card_select
        hi = min(action_mask.max_card_select, len(legal))
        subsets: list[tuple[int, ...]] = []
        for k in range(lo, hi + 1):
            subsets.extend(combinations(legal, k))
        return subsets

    def _score_subset(self, gs: dict[str, Any], subset: tuple[int, ...]) -> tuple[int, str] | None:
        """Deep-copy *gs*, actually play *subset* on the copy, read the result.

        Returns ``None`` if the copy rejects the play (e.g. a forced-card
        selection rule) rather than raising — the caller just skips it.
        """
        gs_copy = copy.deepcopy(gs)
        fa = FactoredAction(action_type=ActionType.PlayHand, card_target=subset)
        try:
            engine_action = factored_to_engine_action(fa, gs_copy)
            gs_copy = engine_step(gs_copy, engine_action)
        except IllegalActionError:
            return None
        result = gs_copy.get("last_score_result")
        if result is None:
            return None
        return result.total, result.hand_type

    def _choose_discard(
        self,
        gs: dict[str, Any],
        best_subset: tuple[int, ...],
        action_mask: ActionMask,
    ) -> FactoredAction | None:
        """Discard everything except the best play and any live draw.

        Simple, cheap, and not exact — one-step lookahead can't value a
        discard (see module docstring). Keeps cards that are part of the
        best-scoring subset just found, or part of a near-flush (a suit
        with >=4 cards present) or near-straight (a 5-rank window with >=4
        distinct ranks present, ace-high only). Everything else is
        discarded.
        """
        hand = gs.get("hand", [])
        keep: set[int] = set(best_subset)

        suit_counts = {s: sum(1 for c in hand if is_suit(c, s, flush_calc=True)) for s in _SUITS}
        best_suit, best_suit_count = max(suit_counts.items(), key=lambda kv: kv[1])
        if best_suit_count >= 4:
            keep |= {i for i, c in enumerate(hand) if is_suit(c, best_suit, flush_calc=True)}

        ids_present: dict[int, int] = {}
        for i, c in enumerate(hand):
            cid = c.get_id()
            if 1 < cid < 15:
                ids_present[cid] = i
        best_window: set[int] = set()
        for lo in range(2, 11):  # windows [lo, lo+4] cover 2-6 .. 10-14 (ace-high)
            present = set(range(lo, lo + 5)) & ids_present.keys()
            if len(present) >= 4:
                candidate = {ids_present[r] for r in present}
                if len(candidate) > len(best_window):
                    best_window = candidate
        keep |= best_window

        legal = set(np.nonzero(action_mask.card_mask)[0].tolist())
        discard = sorted((set(range(len(hand))) - keep) & legal)
        if not discard:
            return None

        n = min(len(discard), action_mask.max_card_select)
        n = max(n, action_mask.min_card_select)
        return FactoredAction(action_type=ActionType.Discard, card_target=tuple(discard[:n]))

    # ------------------------------------------------------------------
    # SHOP: trivial economy default
    # ------------------------------------------------------------------

    def _act_shop(self, gs: dict[str, Any], action_mask: ActionMask) -> FactoredAction:
        """Buy the single cheapest affordable Joker if one is legal, else advance.

        No rerolling, vouchers, consumable purchases, or planet/tarot use —
        this baseline is scoped to hand-selection quality (see module
        docstring).
        """
        if action_mask.type_mask[ActionType.BuyCard]:
            shop_cards = gs.get("shop_cards", [])
            mask = action_mask.entity_masks.get(ActionType.BuyCard)
            if mask is not None:
                joker_idxs = [
                    int(i)
                    for i in np.nonzero(mask)[0]
                    if isinstance(getattr(shop_cards[i], "ability", None), dict)
                    and shop_cards[i].ability.get("set") == "Joker"
                ]
                if joker_idxs:
                    cheapest = min(joker_idxs, key=lambda i: shop_cards[i].cost)
                    return FactoredAction(action_type=ActionType.BuyCard, entity_target=cheapest)
        return FactoredAction(action_type=ActionType.NextRound)

    # ------------------------------------------------------------------
    # Defensive fallback — mirrors RandomAgent's target-filling, but
    # deterministic (lowest legal type/entity/cards) rather than random.
    # ------------------------------------------------------------------

    def _fallback(self, action_mask: ActionMask) -> FactoredAction:
        legal_types = np.nonzero(action_mask.type_mask)[0]
        if len(legal_types) == 0:
            return FactoredAction(action_type=ActionType.SelectBlind)
        at = int(legal_types[0])

        entity_target: int | None = None
        card_target: tuple[int, ...] | None = None

        if at in action_mask.entity_masks:
            legal_entities = np.nonzero(action_mask.entity_masks[at])[0]
            if len(legal_entities) > 0:
                entity_target = int(legal_entities[0])

        if at in (ActionType.PlayHand, ActionType.Discard):
            legal_cards = np.nonzero(action_mask.card_mask)[0]
            if len(legal_cards) > 0:
                n = max(1, min(len(legal_cards), action_mask.min_card_select))
                card_target = tuple(int(i) for i in legal_cards[:n])

        return FactoredAction(action_type=at, card_target=card_target, entity_target=entity_target)


__all__ = ["HeuristicAgent"]
