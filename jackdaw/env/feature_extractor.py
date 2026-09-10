"""Embedding + masked-pooling feature extractor for MaskablePPO.

Implements docs/RL_PLAN.md Sec 5.1 (catalog embeddings) for all three
catalog entity types (joker, consumable, shop_item — ``has_catalog_id=True``
in ``balatro_game_spec()``), and Sec 5.2 (masked mean pooling) for those
same three types only.

Pooling is deliberately NOT applied to ``hand_card``/``pack_card``
(``has_catalog_id=False``) — an earlier version pooled all five entity
types uniformly and a real 500k-step run regressed hard (mean ante 1.26 ->
1.00, i.e. down to random-baseline level) versus the default flatten+concat
extractor. Root cause: pooling collapses "N individual cards in specific
slots" into one averaged vector, destroying exactly the per-slot identity
`PlayHand`/`Discard`'s ``card_target`` (and every other entity-targeting
action's ``entity_target`` pointer) needs to select *which* card/entity —
information the policy head still needs today because the action space is
still a flat ``Discrete(500)`` linear layer, not the pointer/attention head
sketched in Sec 5.4 ("later") that would know how to consume a pooled-away
representation. Pooling only makes sense once something downstream
consumes a per-entity stream via attention/pointer lookup instead of a
flat linear layer over the whole feature vector. `hand_card`/`pack_card`
are flattened instead (same information a default extractor gets, just
still routed through a per-entity-type MLP for capacity/consistency).
Catalog entity types get pooled because they don't carry this project's
current dominant decision (hand selection) and because SellJoker/
SwapJokers*/UseConsumable/BuyCard are comparatively rare, low-stakes,
already-legal-action-masked choices — pooling them is a real bet, not a
proven-safe one, and should be re-examined the same way if a future run
regresses again.

Requires the ``train`` optional dependency group (torch, stable-baselines3).
Not imported by ``jackdaw.env.__init__`` — importing this module pulls in
torch, which the base package must stay usable without (CI's ``uv sync
--dev`` doesn't install ``train`` extras, and most of the test suite
imports ``jackdaw.env`` for reasons unrelated to training).
"""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from jackdaw.env.balatro_spec import balatro_game_spec


class BalatroExtractor(BaseFeaturesExtractor):
    """Per-entity-type embedding + MLP + masked-mean-pool feature extractor.

    Parameters
    ----------
    observation_space:
        The Dict space `BalatroGymnasiumEnv` builds — must contain
        ``global``, one ``(max_count, feat_dim)`` Box per entity type,
        ``entity_counts``, and one ``{name}_ids`` MultiDiscrete per
        catalog entity type.
    features_dim:
        Output dimension of this extractor (SB3's ``features_dim``).
    embed_dim:
        Dimension of each catalog-ID embedding vector.
    entity_hidden_dim:
        Hidden width of each per-entity-type MLP (also its pooled output
        width).
    global_hidden_dim:
        Hidden width of the global-context MLP.
    embed_init_std:
        Std of the catalog embedding initialization. Defaults to 1.0
        (``nn.Embedding``'s own default) — shrinking it was tried and did
        not help, see the note below before changing it.

    Embedding init scale — tried shrinking, it did not help
    ------------------------------------------------------
    Measured with ``scripts/embed_drift.py`` (docs/RUNS.md, "the joker
    embeddings barely move"): after 500k steps the joker table has drifted
    **2.2%** from its random init, consumable 0.8%, shop_item 3.2%. The
    tables are effectively frozen at whatever they were initialized to — so
    init scale is not a minor detail, it *is* most of what the network sees.

    This is **not** an exposure problem, despite an earlier analysis here
    saying so. 142 of the 150 reachable joker keys (95%) receive gradient;
    the earlier "161/300 rows never seen" figure counted against the whole
    shared catalog, but only 150 of those 299 keys are jokers at all. What
    is small is the update per exposure: a seen row moves ~3% of its own
    length over a full run.

    The obvious-looking inference — "an untrained N(0,1) row is a norm-5.66
    random vector drowning 15 normalized feature dims, so shrink it so
    unseen rows are ≈0 and the extractor degrades gracefully to
    features-only" — was tested in run 10 (`embed_init_std=0.1`, otherwise
    identical to run 7) and came out at **mean ante 1.07 vs run 7's 1.26**,
    the lowest of any non-buggy run.

    The likely reason it backfired: a fixed random row is not only noise, it
    is also a *random identity code*. A unique separable 32-dim signature
    per joker is something the downstream Linear can read to tell jokers
    apart with zero training (the random-features effect). Shrinking the
    init 10× removed that distinguishability and made every joker look
    nearly alike, which evidently cost more than the noise it removed.
    Default reverted to 1.0; the parameter is kept for further study, but
    treat "small init is obviously better" as a hypothesis already tried
    and not supported.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        embed_dim: int = 32,
        entity_hidden_dim: int = 64,
        global_hidden_dim: int = 128,
        embed_init_std: float = 1.0,
    ) -> None:
        super().__init__(observation_space, features_dim)

        spec = balatro_game_spec()
        self._entity_info: list[tuple[str, int, int, bool, int]] = [
            (et.name, et.max_count, et.feature_dim, et.has_catalog_id, et.catalog_size)
            for et in spec.entity_types
        ]

        self.embeddings = nn.ModuleDict()
        self.entity_mlps = nn.ModuleDict()
        for name, max_count, feat_dim, has_catalog_id, catalog_size in self._entity_info:
            if has_catalog_id:
                # Pooled path (joker/consumable/shop_item): per-slot MLP
                # input is that slot's own feature vector, pooled after.
                # padding_idx=0 matches the ID convention (0 = unknown/pad)
                # used by center_key_id() and the padding fill in
                # BalatroGymnasiumEnv._build_obs — its embedding row is
                # excluded from gradient updates and stays at its initial
                # (effectively arbitrary) value, which is fine since
                # padding slots are also zeroed out of the pooled mean.
                embedding = nn.Embedding(catalog_size + 1, embed_dim, padding_idx=0)
                # Re-init small (see "Embedding init scale" above). normal_
                # overwrites every row including padding_idx, which
                # nn.Embedding had zeroed at construction — restore that
                # zero explicitly or padding slots stop being neutral.
                with torch.no_grad():
                    nn.init.normal_(embedding.weight, mean=0.0, std=embed_init_std)
                    embedding.weight[0].fill_(0.0)
                self.embeddings[name] = embedding
                in_dim = feat_dim + embed_dim
            else:
                # Flattened path (hand_card/pack_card): input is every
                # slot's feature vector concatenated, preserving per-slot
                # identity for PlayHand/Discard's card_target.
                in_dim = max_count * feat_dim
            self.entity_mlps[name] = nn.Sequential(nn.Linear(in_dim, entity_hidden_dim), nn.ReLU())

        # The optional lookahead block (docs/RL_PLAN.md §5.3 Level 2) is
        # concatenated onto the global context rather than given its own
        # branch: it *is* global context — a summary of the whole menu, not
        # a per-entity quantity — and it is far too small (8 dims) to earn a
        # separate MLP. Read the width from the space so this module stays
        # agnostic about how many features the env chose to emit.
        lookahead_space = observation_space.spaces.get("lookahead")
        self._lookahead_dim = 0 if lookahead_space is None else int(lookahead_space.shape[0])
        self.global_mlp = nn.Sequential(
            nn.Linear(spec.global_feature_dim + self._lookahead_dim, global_hidden_dim), nn.ReLU()
        )

        summary_total = entity_hidden_dim * len(self._entity_info)
        self.head = nn.Sequential(
            nn.Linear(global_hidden_dim + summary_total, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict[str, Any]) -> torch.Tensor:
        entity_parts: list[torch.Tensor] = []
        entity_counts = observations["entity_counts"]  # (B, num_entity_types)

        for idx, (name, max_count, feat_dim, has_catalog_id, _catalog_size) in enumerate(
            self._entity_info
        ):
            feats = observations[name]  # (B, max_count, feat_dim)

            if not has_catalog_id:
                # Flatten — keeps per-slot identity, matches the default
                # extractor's behavior for these two entity types.
                entity_parts.append(self.entity_mlps[name](feats.reshape(feats.shape[0], -1)))
                continue

            ids = observations[f"{name}_ids"].long()  # (B, max_count)
            emb = self.embeddings[name](ids)  # (B, max_count, embed_dim)
            x = torch.cat([emb, feats], dim=-1)
            h = self.entity_mlps[name](x)  # (B, max_count, entity_hidden_dim)

            n = entity_counts[:, idx : idx + 1]  # (B, 1) — real entity count
            arange = torch.arange(max_count, device=h.device).unsqueeze(0)  # (1, max_count)
            valid = (arange < n).unsqueeze(-1).to(h.dtype)  # (B, max_count, 1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B, hidden)
            entity_parts.append(pooled)

        global_x = observations["global"]
        if self._lookahead_dim:
            global_x = torch.cat([global_x, observations["lookahead"]], dim=-1)
        global_h = self.global_mlp(global_x)
        combined = torch.cat([global_h, *entity_parts], dim=-1)
        return self.head(combined)


__all__ = ["BalatroExtractor"]
