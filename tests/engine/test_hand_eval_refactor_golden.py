"""Behaviour-lock for `evaluate_hand` across performance refactors.

`hand_eval.py` is the hottest code under the action-table ranking (~84% of
enumeration time), so it attracts optimization — but it is also engine code,
where `CLAUDE.md` requires bit-exactness and `jackdaw validate` needs a live
BalatroBot server that CI and most dev machines do not have.

This fixture closes that gap offline: 4,011 randomized hands, biased toward
the shapes a rank-grouping or flush rewrite is most likely to break (empty
hands, five of a kind, flush houses, wheel straights, wild/stone
enhancements), with their exact detected hand, scoring-card *positions* and
full `poker_hands` decomposition recorded.

It is NOT a correctness oracle — it locks in whatever the code did when the
fixture was generated. `test_hand_eval_oracle.py` is the correctness check,
comparing against real Lua output. The pair is what makes a refactor safe to
reason about: the oracle says "matches the source", this says "unchanged".

If this fails after an intended behaviour change, regenerate with
`uv run scripts/gen_hand_eval_golden.py` — but only after the Lua oracle
test passes, never to silence a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_eval import evaluate_hand

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hand_eval_refactor_golden.json"

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


def _build(suit: str, rank: str, enhancement: str) -> Card:
    card = Card()
    card.set_base(f"{suit[0]}_{ABBREV[rank]}", suit, rank)
    card.set_ability(enhancement)
    return card


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    assert FIXTURE.exists(), f"missing fixture {FIXTURE}; run scripts/gen_hand_eval_golden.py"
    return json.loads(FIXTURE.read_text())["cases"]


def test_fixture_covers_the_exotic_hand_types(cases: list[dict]) -> None:
    """A golden file that only ever sees High Card would guard nothing."""
    detected = {case["detected_hand"] for case in cases}
    for required in (
        "Five of a Kind",
        "Flush Five",
        "Flush House",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Straight",
        "Flush",
        "Two Pair",
        "NULL",
    ):
        assert required in detected, f"fixture never produces {required}"
    assert len(cases) > 3000


def test_fixture_exercises_the_meta_jokers(cases: list[dict]) -> None:
    """Production never calls `evaluate_hand` bare — `scoring.py` passes
    jokers and flag_overrides, and Four Fingers / Shortcut / Smeared change
    which hands are detectable at all. A fixture covering only the no-joker
    path would miss any interaction between rank grouping and relaxed
    flush/straight rules."""
    seen = {key for case in cases for key in case.get("jokers", [])}
    for required in ("j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia"):
        assert required in seen, f"fixture never exercises {required}"
    assert sum(1 for c in cases if c.get("jokers")) > 1000


def test_evaluate_hand_matches_golden(cases: list[dict]) -> None:
    mismatches: list[str] = []
    for i, case in enumerate(cases):
        hand = [_build(*entry) for entry in case["hand"]]
        jokers = [create_joker(key) for key in case.get("jokers", [])] or None
        result = evaluate_hand(hand, jokers=jokers)
        index = {id(c): pos for pos, c in enumerate(hand)}

        if result.detected_hand != case["detected_hand"]:
            mismatches.append(
                f"case {i}: detected {result.detected_hand!r} != {case['detected_hand']!r}"
            )
            continue

        scoring = [index[id(c)] for c in result.scoring_cards]
        if scoring != case["scoring_cards"]:
            mismatches.append(f"case {i}: scoring_cards {scoring} != {case['scoring_cards']}")
            continue

        # Card *order* inside each group matters: downstream scoring walks
        # these lists in order, so a permutation is a real behaviour change.
        actual = {
            name: [[index[id(c)] for c in group] for group in groups]
            for name, groups in sorted(result.poker_hands.items())
            if groups
        }
        if actual != case["poker_hands"]:
            mismatches.append(f"case {i}: poker_hands {actual} != {case['poker_hands']}")

    assert not mismatches, f"{len(mismatches)} mismatch(es):\n" + "\n".join(mismatches[:10])
