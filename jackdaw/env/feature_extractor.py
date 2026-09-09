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
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        embed_dim: int = 32,
        entity_hidden_dim: int = 64,
        global_hidden_dim: int = 128,
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
                self.embeddings[name] = nn.Embedding(catalog_size + 1, embed_dim, padding_idx=0)
                in_dim = feat_dim + embed_dim
            else:
                # Flattened path (hand_card/pack_card): input is every
                # slot's feature vector concatenated, preserving per-slot
                # identity for PlayHand/Discard's card_target.
                in_dim = max_count * feat_dim
            self.entity_mlps[name] = nn.Sequential(nn.Linear(in_dim, entity_hidden_dim), nn.ReLU())

        self.global_mlp = nn.Sequential(
            nn.Linear(spec.global_feature_dim, global_hidden_dim), nn.ReLU()
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

        global_h = self.global_mlp(observations["global"])
        combined = torch.cat([global_h, *entity_parts], dim=-1)
        return self.head(combined)


__all__ = ["BalatroExtractor"]
