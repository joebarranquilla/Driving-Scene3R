"""Hermetic CPU tests for the photometric losses."""

from __future__ import annotations

import pytest
import torch

from semantic_gs.train.losses import (
    masked_l1_loss,
    photometric_loss,
    psnr,
    ssim,
)


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------

def test_l1_identity_is_zero() -> None:
    img = torch.rand(8, 12, 3)
    assert masked_l1_loss(img, img).item() < 1e-7


def test_l1_unmasked_matches_torch_mean() -> None:
    rng = torch.Generator().manual_seed(0)
    p = torch.rand(6, 4, 3, generator=rng)
    g = torch.rand(6, 4, 3, generator=rng)
    expected = (p - g).abs().mean().item()
    assert abs(masked_l1_loss(p, g).item() - expected) < 1e-7


def test_l1_with_mask_uses_only_selected_pixels() -> None:
    pred = torch.zeros(4, 4, 3)
    gt   = torch.ones(4, 4, 3)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 0] = True
    # Only the (0, 0) pixel contributes, per-channel diff = 1.
    assert abs(masked_l1_loss(pred, gt, mask).item() - 1.0) < 1e-6


def test_l1_mask_shape_mismatch_raises() -> None:
    p = torch.zeros(4, 4, 3)
    g = torch.zeros(4, 4, 3)
    bad_mask = torch.zeros(5, 5, dtype=torch.bool)
    with pytest.raises(ValueError):
        masked_l1_loss(p, g, bad_mask)


def test_l1_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        masked_l1_loss(torch.zeros(4, 4, 3), torch.zeros(4, 5, 3))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identity_is_one() -> None:
    img = torch.rand(16, 16, 3)
    assert abs(ssim(img, img).item() - 1.0) < 1e-4


def test_ssim_bounded_zero_to_one() -> None:
    p = torch.rand(16, 16, 3)
    g = torch.rand(16, 16, 3)
    s = float(ssim(p, g).item())
    assert -0.1 <= s <= 1.0   # could go slightly negative for random images


def test_ssim_rejects_non_image_shape() -> None:
    with pytest.raises(ValueError):
        ssim(torch.zeros(16, 16, 4), torch.zeros(16, 16, 4))
    with pytest.raises(ValueError):
        ssim(torch.zeros(16, 3), torch.zeros(16, 3))


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_identity_is_very_high() -> None:
    img = torch.rand(8, 8, 3)
    assert psnr(img, img) >= 80.0


def test_psnr_known_value() -> None:
    # MSE = 0.25 -> PSNR = -10*log10(0.25) = 6.02 dB
    pred = torch.full((4, 4, 3), 0.5)
    gt   = torch.zeros(4, 4, 3)
    assert abs(psnr(pred, gt) - 6.0206) < 1e-3


# ---------------------------------------------------------------------------
# Combined photometric loss
# ---------------------------------------------------------------------------

def test_photometric_loss_returns_scalar_and_components() -> None:
    p = torch.rand(8, 8, 3)
    g = torch.rand(8, 8, 3)
    loss, comp = photometric_loss(p, g)
    assert loss.ndim == 0
    assert set(comp) >= {"l1", "ssim", "loss"}
    assert all(isinstance(v, float) for v in comp.values())


def test_photometric_loss_backprops() -> None:
    p = torch.rand(8, 8, 3, requires_grad=True)
    g = torch.rand(8, 8, 3)
    loss, _ = photometric_loss(p, g)
    loss.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()


def test_photometric_loss_respects_lambda_ssim_bounds() -> None:
    p = torch.rand(16, 16, 3)
    g = torch.rand(16, 16, 3)
    loss_l1_only, _   = photometric_loss(p, g, lambda_ssim=0.0)
    loss_ssim_only, _ = photometric_loss(p, g, lambda_ssim=1.0)
    # All-L1 loss should equal the bare L1 value; all-SSIM should equal (1 - ssim).
    assert abs(float(loss_l1_only.item()) - masked_l1_loss(p, g).item()) < 1e-6
    assert abs(float(loss_ssim_only.item()) - (1.0 - ssim(p, g).item())) < 1e-6

