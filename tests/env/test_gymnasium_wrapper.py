"""Tests for the Gymnasium wrapper around BalatroEnvironment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from jackdaw.env.action_space import ActionType
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv


@pytest.fixture()
def env() -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)


# ------------------------------------------------------------------
# Space definitions
# ------------------------------------------------------------------


class TestSpaces:
    def test_action_space_is_discrete(self, env: BalatroGymnasiumEnv) -> None:
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == MAX_ACTIONS

    def test_observation_space_keys(self, env: BalatroGymnasiumEnv) -> None:
        assert isinstance(env.observation_space, spaces.Dict)
        expected = {
            "global",
            "hand_card",
            "joker",
            "consumable",
            "shop_item",
            "pack_card",
            "entity_counts",
            # Catalog-ID channels (docs/RL_PLAN.md Sec 5.1), one per
            # has_catalog_id=True entity type.
            "joker_ids",
            "consumable_ids",
            "shop_item_ids",
        }
        assert set(env.observation_space.spaces.keys()) == expected

    def test_catalog_id_spaces_are_int_boxes(self, env: BalatroGymnasiumEnv) -> None:
        """Box, not MultiDiscrete — SB3 one-hot-encodes MultiDiscrete/Discrete
        spaces before a features extractor ever sees them, which defeats the
        point of an embedding lookup. See gymnasium_wrapper.py's __init__."""
        for key in ("joker_ids", "consumable_ids", "shop_item_ids"):
            space = env.observation_space.spaces[key]
            assert isinstance(space, spaces.Box)
            assert space.dtype == np.int64


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env: BalatroGymnasiumEnv) -> None:
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_matches_observation_space(self, env: BalatroGymnasiumEnv) -> None:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_action_mask_in_info(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        assert "action_mask" in info
        mask = info["action_mask"]
        assert mask.shape == (MAX_ACTIONS,)
        assert mask.dtype == bool

    def test_at_least_one_legal_action(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        assert info["action_mask"].any()

    def test_game_seed_option_selects_the_string_seed(self) -> None:
        """options["game_seed"] lets a caller pass a string Balatro seed
        (e.g. from EVAL_SEEDS) through the int-typed `seed` param, without
        touching the action-table subsampling RNG."""
        env_a = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        env_b = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)

        obs_a, _ = env_a.reset(options={"game_seed": "GAME_SEED_OPTION_TEST"})
        obs_b, _ = env_b.reset(options={"game_seed": "GAME_SEED_OPTION_TEST"})

        assert np.array_equal(obs_a["global"], obs_b["global"])
        for name in ("hand_card", "joker", "consumable", "shop_item", "pack_card"):
            assert np.array_equal(obs_a[name], obs_b[name])

    def test_game_seed_option_takes_priority_over_seed(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        obs_game_seed, _ = env.reset(seed=1, options={"game_seed": "PRIORITY_TEST"})

        env2 = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        obs_direct, _ = env2.reset(options={"game_seed": "PRIORITY_TEST"})

        assert np.array_equal(obs_game_seed["global"], obs_direct["global"])

    def test_game_seed_alone_also_reseeds_subsampling_rng(self) -> None:
        """Regression test: a game_seed-only reset (no int `seed`) must
        deterministically reseed self._rng too, not just the underlying
        Balatro game. Before this fix, self._rng stayed on OS entropy
        whenever only `options["game_seed"]` was passed, so two resets on
        the identical game_seed could enumerate DIFFERENT action tables
        whenever _enumerate_actions's random subsampling kicked in
        (len(actions) > MAX_ACTIONS) — silently breaking eval
        reproducibility for a "frozen" seed set. Verified directly against
        self._rng's own output rather than hunting for a real game state
        that happens to exceed MAX_ACTIONS actions."""
        env_a = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        env_a.reset(options={"game_seed": "SUBSAMPLE_RNG_TEST"})
        draw_a = env_a._rng.integers(0, 1_000_000, size=10)

        env_b = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        env_b.reset(options={"game_seed": "SUBSAMPLE_RNG_TEST"})
        draw_b = env_b._rng.integers(0, 1_000_000, size=10)

        assert np.array_equal(draw_a, draw_b)


# ------------------------------------------------------------------
# Step
# ------------------------------------------------------------------


class TestStep:
    def test_step_returns_five_tuple(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_matches_space(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        obs, reward, terminated, truncated, step_info = env.step(action)
        if not (terminated or truncated):
            assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        _, reward, *_ = env.step(action)
        assert isinstance(reward, float)

    def test_mid_episode_reward_is_zero(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        _, reward, terminated, truncated, _ = env.step(action)
        if not (terminated or truncated):
            assert reward == 0.0


# ------------------------------------------------------------------
# Action masks
# ------------------------------------------------------------------


class TestActionMasks:
    def test_action_masks_method_exists(self, env: BalatroGymnasiumEnv) -> None:
        assert callable(getattr(env, "action_masks", None))

    def test_action_masks_shape_and_dtype(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (MAX_ACTIONS,)
        assert mask.dtype == bool

    def test_padding_slots_are_false(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        mask = env.action_masks()
        n_legal = len(env._action_table)
        assert mask[:n_legal].all()
        assert not mask[n_legal:].any()

    def test_action_table_within_budget(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        assert len(env._action_table) <= MAX_ACTIONS


# ------------------------------------------------------------------
# Integration: random episodes
# ------------------------------------------------------------------


class TestRandomEpisodes:
    @pytest.mark.slow
    def test_ten_random_episodes(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=500)
        rng = np.random.default_rng(123)

        for ep in range(10):
            obs, info = env.reset(seed=ep)
            assert env.observation_space.contains(obs)
            mask = info["action_mask"]

            for _ in range(500):
                legal = np.nonzero(mask)[0]
                assert len(legal) > 0, "No legal actions but episode not done"
                action = int(rng.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                mask = info["action_mask"]
                if terminated or truncated:
                    break
            assert terminated or truncated, f"Episode {ep} did not finish in 500 steps"

    def test_single_episode_completes(self, env: BalatroGymnasiumEnv) -> None:
        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=0)
        mask = info["action_mask"]

        for _ in range(200):
            legal = np.nonzero(mask)[0]
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                break


# ------------------------------------------------------------------
# Metrics (known issue #7: sparse mode never updated episode trackers)
# ------------------------------------------------------------------


class TestCatalogIdChannels:
    """docs/RL_PLAN.md Sec 5.1: obs["{name}_ids"] must carry the raw
    catalog ID for real entities and 0 (unknown/padding) beyond
    entity_counts, matching card.py-jokers's position 1:1."""

    def test_joker_ids_match_real_entities_and_pad_with_zero(self) -> None:
        from jackdaw.engine.card import Card
        from jackdaw.env.observation import center_key_id

        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)
        env.reset(seed=0)

        joker = Card()
        joker.set_ability("j_joker")
        env._inner._adapter.raw_state["jokers"] = [joker]

        from jackdaw.env.observation import encode_observation

        game_obs = encode_observation(env._inner._adapter.raw_state).to_game_observation()
        obs = env._build_obs(game_obs)

        n_jokers = int(obs["entity_counts"][1])  # joker is entity index 1
        assert n_jokers == 1
        assert obs["joker_ids"][0] == center_key_id(joker.center_key)
        assert obs["joker_ids"][0] > 0
        # Everything past the real entity count is padding.
        assert np.all(obs["joker_ids"][n_jokers:] == 0)


class TestStallDetection:
    """Known issue #15: a policy repeating an action that never changes
    chips/round/ante/hands_left/discards_left/dollars must be forced to
    truncate rather than allowed to spend the full max_steps budget."""

    def _first_legal_index(self, action_table, action_type: ActionType) -> int | None:
        for i, fa in enumerate(action_table):
            if fa.action_type == action_type:
                return i
        return None

    def test_repeated_no_progress_action_forces_truncation(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=5000)
        _, info = env.reset(seed=0)
        # Advance past blind select into SELECTING_HAND.
        legal = np.nonzero(info["action_mask"])[0]
        obs, _reward, terminated, truncated, info = env.step(int(legal[0]))
        assert not terminated and not truncated

        swap_idx = self._first_legal_index(env._action_table, ActionType.SwapHandLeft)
        assert swap_idx is not None, "expected SwapHandLeft legal with the starting 8-card hand"

        reward = 0.0
        for _ in range(env._STALL_STEPS_LIMIT + 2):
            obs, reward, terminated, truncated, info = env.step(swap_idx)
            if terminated or truncated:
                break
            swap_idx = self._first_legal_index(env._action_table, ActionType.SwapHandLeft)
            assert swap_idx is not None

        assert truncated
        assert not terminated
        assert reward < 0, "stall penalty must make this worse than a normal 0.0 mid-episode reward"

    def test_normal_play_does_not_trigger_stall_truncation(self, env: BalatroGymnasiumEnv) -> None:
        """A real episode (random legal actions, which always include
        progress-making choices some of the time) shouldn't be truncated by
        the stall detector before it naturally terminates or truncates on
        max_steps — this just guards against a threshold set so low it
        false-positives on ordinary play."""
        rng = np.random.default_rng(3)
        _, info = env.reset(seed=3)
        mask = info["action_mask"]
        stall_flagged = False

        for _ in range(200):
            legal = np.nonzero(mask)[0]
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if truncated and env._stall_steps >= env._STALL_STEPS_LIMIT:
                stall_flagged = True
            if terminated or truncated:
                break

        assert not stall_flagged


class TestMetricsInSparseMode:
    def test_trackers_update_without_reward_shaping(self, env: BalatroGymnasiumEnv) -> None:
        """env fixture defaults reward_shaping=False (sparse). Previously
        _compute_reward's early return in sparse mode skipped the tracker
        updates entirely, so _episode_max_ante/_episode_max_round (and thus
        the terminal balatro/ante_reached, rounds_beaten info) stayed
        hardcoded at their reset values (1, 0) forever, regardless of actual
        progress. Call _compute_reward directly with a fabricated raw_state
        so this is deterministic, not dependent on a random episode reaching
        ante 2+ within a bounded step budget."""
        assert env._reward_shaping is False
        env.reset(seed=0)

        fake_info = {"raw_state": {"round_resets": {"ante": 3}, "round": 2, "chips": 0}}
        reward = env._compute_reward(fake_info, terminated=False, truncated=False)

        assert reward == 0.0  # sparse mode: 0 mid-episode regardless of progress
        assert env._episode_max_ante == 3
        assert env._episode_max_round == 2
