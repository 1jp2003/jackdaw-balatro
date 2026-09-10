#!/usr/bin/env python3
"""Generate tests/fixtures/hand_eval_refactor_golden.json.

This fixture locks in the *current* behaviour of `evaluate_hand` across a
broad sweep of randomized hands, so that a performance refactor of
`jackdaw/engine/hand_eval.py` can be proven behaviour-preserving offline.

It is deliberately NOT a correctness oracle. Correctness against the Lua
source is `tests/engine/test_hand_eval_oracle.py`, which compares against
real Lua output; this file only says "the refactor changed nothing". Both
are needed: the oracle has a handful of curated hands, this has thousands of
randomized ones including the degenerate shapes (empty hands, five of a
kind, wheel straights) that a rewrite is most likely to get subtly wrong.

Regenerate ONLY when a behaviour change to hand evaluation is intended and
has been validated against the Lua oracle — never to make a failing test
pass.

Usage::

    uv run scripts/gen_hand_eval_golden.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_eval import evaluate_hand

FIXTURE = Path("tests/fixtures/hand_eval_refactor_golden.json")

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
ABBREV = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "Jack": "J",
    "Queen": "Q",
    "King": "K",
    "Ace": "A",
}
# Enhancements that change how a card counts for flushes/scoring.
ENHANCEMENTS = ["c_base", "m_wild", "m_stone", "m_glass", "m_steel", "m_gold"]


def build_card(suit: str, rank: str, enhancement: str) -> Card:
    card = Card()
    card.set_base(f"{suit[0]}_{ABBREV[rank]}", suit, rank)
    card.set_ability(enhancement)
    return card


def sample_specs(rng: random.Random, n: int) -> list[list[tuple[str, str, str]]]:
    """Hand specs biased toward the shapes a rank-grouping rewrite can break."""
    specs: list[list[tuple[str, str, str]]] = []
    for _ in range(n):
        size = rng.choice([0, 1, 2, 3, 4, 5, 5, 5])
        # Narrow rank/suit pools make pairs, trips, quads, five-of-a-kind and
        # flushes common instead of astronomically rare.
        rank_pool = RANKS[: rng.choice([1, 2, 3, 5, 13])]
        suit_pool = SUITS[: rng.choice([1, 2, 4])]
        spec = [
            (
                suit_pool[rng.randrange(len(suit_pool))],
                rank_pool[rng.randrange(len(rank_pool))],
                ENHANCEMENTS[0] if rng.random() < 0.75 else rng.choice(ENHANCEMENTS),
            )
            for _ in range(size)
        ]
        specs.append(spec)
    if n == 0:
        specs = []
    # Straights and wheel straights, which random sampling rarely produces.
    for low in range(0, 9):
        specs.append([(SUITS[i % 4], RANKS[low + i], "c_base") for i in range(5)])
    specs.append([(SUITS[i % 4], r, "c_base") for i, r in enumerate(["Ace", "2", "3", "4", "5"])])
    specs.append([(SUITS[0], r, "c_base") for r in ["10", "Jack", "Queen", "King", "Ace"]])
    return specs


def result_signature(spec: list[tuple[str, str, str]], joker_keys: list[str]) -> dict:
    hand = [build_card(s, r, e) for s, r, e in spec]
    jokers = [create_joker(key) for key in joker_keys] or None
    result = evaluate_hand(hand, jokers=jokers)
    index = {id(c): i for i, c in enumerate(hand)}
    return {
        "hand": [list(entry) for entry in spec],
        "jokers": joker_keys,
        "detected_hand": result.detected_hand,
        # Positions, not repr: this is what a grouping rewrite can permute.
        "scoring_cards": [index[id(c)] for c in result.scoring_cards],
        "poker_hands": {
            name: [[index[id(c)] for c in group] for group in groups]
            for name, groups in sorted(result.poker_hands.items())
            if groups
        },
    }


def main() -> None:
    rng = random.Random(20260909)
    specs = sample_specs(rng, 4000)

    # Production never calls evaluate_hand bare: scoring.py passes jokers
    # (line 465) and flag_overrides (line 199), and those meta jokers change
    # which hands are detectable at all. A fixture that only covers the
    # no-joker path would miss any interaction between the rank grouping and
    # flush/straight detection under relaxed rules.
    joker_sets: list[list[str]] = [
        [],
        [],
        [],
        ["j_four_fingers"],
        ["j_shortcut"],
        ["j_smeared"],
        ["j_four_fingers", "j_smeared"],
        ["j_shortcut", "j_four_fingers"],
        ["j_splash"],
        ["j_pareidolia"],
    ]

    cases = [result_signature(spec, rng.choice(joker_sets)) for spec in specs]
    # And every joker set against every straight/flush-shaped hand, since the
    # random sample rarely produces those and they are where relaxed
    # detection rules actually bite.
    structured = sample_specs(random.Random(7), 0)
    for spec in structured:
        for joker_keys in joker_sets:
            cases.append(result_signature(spec, joker_keys))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"cases": cases}, indent=0))
    detected = {c["detected_hand"] for c in cases}
    with_jokers = sum(1 for c in cases if c["jokers"])
    print(f"wrote {FIXTURE} with {len(cases)} cases ({with_jokers} with meta jokers)")
    print(f"hand types covered ({len(detected)}): {', '.join(sorted(detected))}")


if __name__ == "__main__":
    main()
