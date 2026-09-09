"""Frozen evaluation seed set for baseline/RL agent comparison.

Committed and deterministic — never regenerated at random. Per
``docs/RL_PLAN.md`` Phase 0 ("Frozen eval harness: 200 held-out seeds") and
``CLAUDE.md`` Conventions ("Compare results on the frozen eval seed set, not
rollout statistics"). Do not change this list once results referencing it by
seed have been recorded anywhere (docs, tensorboard runs) — that would
silently invalidate every existing comparison.
"""

from __future__ import annotations

EVAL_SEEDS: list[str] = [f"EVAL_{i:03d}" for i in range(200)]

__all__ = ["EVAL_SEEDS"]
