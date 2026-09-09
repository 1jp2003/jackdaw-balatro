"""Tests for BalatroExtractor (docs/RL_PLAN.md Sec 5.2).

Requires the ``train`` optional dependency group (torch, stable-baselines3)
— skipped entirely when it's not installed (e.g. CI's ``uv sync --dev``),
consistent with how scripts/train_ppo.py and scripts/eval_ppo.py are
train-extras-gated.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jackdaw.env.game_interface import DirectAdapter  # noqa: E402
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv  # noqa: E402


@pytest.fixture()
def env() -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)


def _obs_batch_to_tensors(obs: dict[str, np.ndarray], batch_size: int = 3) -> dict:
    """Stack a single obs dict into a fake batch of tensors."""
    return {key: torch.as_tensor(np.stack([arr] * batch_size)) for key, arr in obs.items()}


class TestBalatroExtractor:
    def test_forward_output_shape(self, env: BalatroGymnasiumEnv) -> None:
        from jackdaw.env.feature_extractor import BalatroExtractor

        obs, _ = env.reset(seed=0)
        extractor = BalatroExtractor(env.observation_space, features_dim=256)

        batch = _obs_batch_to_tensors(obs, batch_size=4)
        out = extractor(batch)

        assert out.shape == (4, 256)
        assert torch.isfinite(out).all()

    def test_forward_with_real_jokers_present(self, env: BalatroGymnasiumEnv) -> None:
        """Exercise the embedding-lookup path (padding_idx=0 branch is
        already covered by the all-empty-entities case above)."""
        from jackdaw.engine.card import Card
        from jackdaw.env.feature_extractor import BalatroExtractor

        obs, _ = env.reset(seed=0)
        joker = Card()
        joker.set_ability("j_joker")
        env._inner._adapter.raw_state["jokers"] = [joker]

        from jackdaw.env.observation import encode_observation

        game_obs = encode_observation(env._inner._adapter.raw_state).to_game_observation()
        obs = env._build_obs(game_obs)
        assert obs["joker_ids"][0] > 0  # sanity: a real, non-padding ID is present

        extractor = BalatroExtractor(env.observation_space, features_dim=128)
        batch = _obs_batch_to_tensors(obs, batch_size=2)
        out = extractor(batch)

        assert out.shape == (2, 128)
        assert torch.isfinite(out).all()

    def test_different_batches_give_different_output(self) -> None:
        """Sanity check the extractor isn't accidentally ignoring its input
        (e.g. a masking bug that zeros everything out). Compared after
        SelectBlind, not at the raw reset() state — at BLIND_SELECT no hand
        has been dealt yet, so different seeds legitimately produce
        byte-identical initial observations (nothing seed-dependent has
        been revealed), which isn't a useful discriminator here."""
        from jackdaw.env.feature_extractor import BalatroExtractor

        env1 = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        env2 = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        _, info1 = env1.reset(seed=1)
        _, info2 = env2.reset(seed=2)
        legal1 = int(np.nonzero(info1["action_mask"])[0][0])
        legal2 = int(np.nonzero(info2["action_mask"])[0][0])
        obs1, *_ = env1.step(legal1)
        obs2, *_ = env2.step(legal2)

        extractor = BalatroExtractor(env1.observation_space, features_dim=64)
        out1 = extractor(_obs_batch_to_tensors(obs1, batch_size=1))
        out2 = extractor(_obs_batch_to_tensors(obs2, batch_size=1))

        assert not torch.allclose(out1, out2)

    def test_hand_card_order_is_not_pooled_away(self, env: BalatroGymnasiumEnv) -> None:
        """Regression test for the Run 6 finding: an earlier version pooled
        every entity type uniformly, including hand_card, which collapsed
        the 8 individual cards into one averaged vector and destroyed the
        per-slot identity PlayHand/Discard's card_target needs. Swapping
        two hand card slots must change the extractor's output — if it
        doesn't, hand_card is being pooled again."""
        from jackdaw.env.feature_extractor import BalatroExtractor

        obs, info = env.reset(seed=5)
        legal = int(np.nonzero(info["action_mask"])[0][0])
        obs, *_ = env.step(legal)  # advance into SELECTING_HAND, hand dealt

        assert int(obs["entity_counts"][0]) >= 2, "need >=2 real hand cards for this test"

        swapped = {k: v.copy() for k, v in obs.items()}
        swapped["hand_card"][[0, 1]] = swapped["hand_card"][[1, 0]]
        # hand_card has no catalog ID (has_catalog_id=False), so no *_ids
        # array to swap in lockstep.

        extractor = BalatroExtractor(env.observation_space, features_dim=32)
        out_original = extractor(_obs_batch_to_tensors(obs, batch_size=1))
        out_swapped = extractor(_obs_batch_to_tensors(swapped, batch_size=1))

        assert not torch.allclose(out_original, out_swapped)

    def test_gradients_flow_to_embeddings(self, env: BalatroGymnasiumEnv) -> None:
        """The whole point of Sec 5.1/5.2 is that joker identity becomes
        learnable — confirm the embedding table actually receives
        gradients on a backward pass through a state with a real joker."""
        from jackdaw.engine.card import Card
        from jackdaw.env.feature_extractor import BalatroExtractor
        from jackdaw.env.observation import encode_observation

        env.reset(seed=0)
        joker = Card()
        joker.set_ability("j_joker")
        env._inner._adapter.raw_state["jokers"] = [joker]
        game_obs = encode_observation(env._inner._adapter.raw_state).to_game_observation()
        obs = env._build_obs(game_obs)

        extractor = BalatroExtractor(env.observation_space, features_dim=32)
        batch = _obs_batch_to_tensors(obs, batch_size=2)
        out = extractor(batch)
        out.sum().backward()

        joker_embedding_grad = extractor.embeddings["joker"].weight.grad
        assert joker_embedding_grad is not None
        assert torch.any(joker_embedding_grad != 0)
