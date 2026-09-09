"""Observation golden tests.

Snapshots ``encode_observation``'s output for a couple of fixed, cheaply
reproducible game states and asserts it hasn't drifted. Per CLAUDE.md's
Conventions ("add observation golden tests before refactoring
observation.py") and docs/RL_PLAN.md Phase 0 — this is the safety net a
future observation.py refactor (e.g. the center_key-as-embedding fix, known
issue #1) needs so a representation change is a deliberate, visible diff
against tests/fixtures/observation_golden.json rather than a silent change
in what the policy network sees. Also protects live-bridge parity: any
accidental change here would equally invalidate trained-policy replay
against real Balatro.

To intentionally update the fixture after a real observation.py change,
regenerate it with the same snapshot logic used here (reset -> DirectAdapter,
optionally SelectBlind, encode_observation, round to 6dp) and review the
diff like any other code change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jackdaw.engine.actions import SelectBlind
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.observation import Observation, encode_observation

SEED = "OBS_GOLDEN_1"
BACK = "b_red"
STAKE = 1

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "observation_golden.json"


def _load_golden() -> dict[str, dict[str, list]]:
    with open(_FIXTURE_PATH) as f:
        return json.load(f)


_GOLDEN = _load_golden()


def _assert_matches(obs: Observation, expected: dict[str, list], scenario: str) -> None:
    fields = {
        "global_context": obs.global_context,
        "hand_cards": obs.hand_cards,
        "jokers": obs.jokers,
        "consumables": obs.consumables,
        "shop_cards": obs.shop_cards,
        "pack_cards": obs.pack_cards,
    }
    for name, actual in fields.items():
        expected_arr = np.array(expected[name], dtype=np.float32)
        # An empty golden array is stored as [] (no known column count);
        # reshape to match actual's column count so shape comparison and
        # allclose both work for the (0, D) case.
        if expected_arr.size == 0 and actual.size == 0:
            assert actual.shape[0] == expected_arr.reshape(-1).shape[0] == 0
            continue
        assert actual.shape == expected_arr.shape, (
            f"{scenario}.{name}: shape {actual.shape} != golden {expected_arr.shape}"
        )
        assert np.allclose(actual, expected_arr, atol=1e-5), (
            f"{scenario}.{name}: values drifted from tests/fixtures/observation_golden.json"
        )


class TestObservationGolden:
    def test_at_blind_select(self) -> None:
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        gs = adapter.raw_state
        assert gs.get("phase") == "blind_select"

        obs = encode_observation(gs)
        _assert_matches(obs, _GOLDEN["at_blind_select"], "at_blind_select")

    def test_at_selecting_hand(self) -> None:
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        gs = adapter.raw_state
        assert gs.get("phase") == "selecting_hand"

        obs = encode_observation(gs)
        _assert_matches(obs, _GOLDEN["at_selecting_hand"], "at_selecting_hand")

    @pytest.mark.parametrize("scenario", ["at_blind_select", "at_selecting_hand"])
    def test_global_context_dim_matches_spec(self, scenario: str) -> None:
        from jackdaw.env.observation import D_GLOBAL

        assert len(_GOLDEN[scenario]["global_context"]) == D_GLOBAL
