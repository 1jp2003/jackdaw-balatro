"""Regression tests for the crash that killed runs 3, 4, 9 and 14.

Diagnosed as a float32 tolerance issue, not a NaN and not a training
pathology — see docs/RUNS.md, "Run 14's crash". PyTorch's ``Simplex()``
constraint uses a fixed 1e-6 absolute tolerance, and a float32 softmax over
``MAX_ACTIONS`` = 500 categories sits close enough to it that an unlucky row
occasionally lands outside. When that happens, constructing the action
distribution raises and the whole run dies with every training metric
healthy.

Requires the ``train`` optional dependency group.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS  # noqa: E402


def _load_train_ppo():  # type: ignore[no-untyped-def]
    """Import scripts/train_ppo.py, which is a script rather than a module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("train_ppo", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEVIATION_AT_CRASH = 1.073e-6
"""The exact excess captured at run 14's crash: one row of 256 summed to
1.0000010729 against Simplex()'s 1e-6 tolerance."""


def _drifted_masking_distribution():  # type: ignore[no-untyped-def]
    """A MaskableCategorical whose cached `probs` sits just off the simplex.

    Reproduces the upstream ordering bug rather than a synthetic tensor.
    `MaskableCategorical.apply_masking` re-runs `Categorical.__init__` with
    the masked logits *before* refreshing `self.probs`:

        super().__init__(logits=logits)            # validates stale probs
        self.probs = logits_to_probs(self.logits)  # refresh — too late

    so PyTorch validates the `probs` left over from the previous, *unmasked*
    parameterization. That stale tensor is a float32 softmax over all
    MAX_ACTIONS categories, which is why the tensor in the crash traceback
    had 500 non-zero entries even though only ~40 actions were legal.
    """
    from sb3_contrib.common.maskable.distributions import MaskableCategorical

    logits = torch.randn(1, MAX_ACTIONS, generator=torch.Generator().manual_seed(0))
    dist = MaskableCategorical(logits=logits.float())
    stale = dist.probs.clone()
    stale[0, 0] += DEVIATION_AT_CRASH
    dist.probs = stale

    masks = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool)
    masks[0, :40] = True  # a realistic table size: measured median 40, max 52
    return dist, masks


class TestSimplexToleranceCrash:
    def test_float32_softmax_sits_at_the_tolerance_boundary(self) -> None:
        """Pin the numerical fact the crash rests on.

        Over 200k random draws the worst observed drift was 9.5e-7 — just
        *under* the 1e-6 limit. The margin is that thin, and a real run
        evaluates on the order of 10^7 rows, so the tail gets reached. This
        cannot be trained away, and no amount of healthy `explained_variance`
        prevents it.
        """
        generator = torch.Generator().manual_seed(0)
        worst = 0.0
        for _ in range(3000):
            logits = torch.randn(MAX_ACTIONS, generator=generator, dtype=torch.float32) * 4.0
            worst = max(worst, abs(float(torch.softmax(logits, dim=-1).sum()) - 1.0))
        assert worst > 1e-8, f"expected drift within an order of the tolerance, got {worst:.3g}"

    def test_apply_masking_validates_stale_probs_and_raises(self) -> None:
        """With validation on, this is the crash that killed 4 runs."""
        dist, masks = _drifted_masking_distribution()
        previous = torch.distributions.Distribution._validate_args
        try:
            torch.distributions.Distribution.set_default_validate_args(True)
            with pytest.raises(ValueError, match="Simplex"):
                dist.apply_masking(masks)
        finally:
            torch.distributions.Distribution.set_default_validate_args(previous)

    def test_disable_distribution_validation_prevents_it(self) -> None:
        """The fix: the same sequence must complete, and stay correct."""
        train_ppo = _load_train_ppo()
        dist, masks = _drifted_masking_distribution()

        previous = torch.distributions.Distribution._validate_args
        try:
            torch.distributions.Distribution.set_default_validate_args(True)
            train_ppo.disable_distribution_validation()
            dist.apply_masking(masks)

            # Masking still works and the refreshed probs are sane — the
            # point is that a 1e-6 excess was never a correctness problem.
            assert torch.isfinite(dist.probs).all()
            assert float(dist.probs[0, 40:].max()) < 1e-12, "illegal actions must stay ~0"
            assert float(dist.probs.sum()) == pytest.approx(1.0, abs=1e-5)
            assert torch.isfinite(dist.entropy()).all()
        finally:
            torch.distributions.Distribution.set_default_validate_args(previous)

    def test_training_entrypoint_disables_validation(self) -> None:
        """Guard against the call being dropped from main() in a refactor —
        the failure mode is a run dying hours in, which is expensive to
        rediscover (it has been rediscovered four times).
        """
        source = (Path(__file__).resolve().parents[2] / "scripts" / "train_ppo.py").read_text()
        main_body = source.split("def main()", 1)[1]
        assert "disable_distribution_validation()" in main_body
