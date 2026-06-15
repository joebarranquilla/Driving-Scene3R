"""Photometric losses for Phase 3 RGB training.

Image tensor convention used throughout this module:
``(H, W, 3)`` ``float32`` in ``[0, 1]`` (channels-last). The trainer
moves frames onto the GPU in this layout before calling these functions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------

def masked_l1_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean per-pixel L1 between two ``(H, W, 3)`` tensors.

    When ``mask`` (``bool (H, W)``) is given, only pixels where ``mask=True``
    contribute. The reduction normalises by the number of contributing
    *pixel-channels* so the magnitude is comparable to an unmasked L1.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"masked_l1_loss: pred {pred.shape} != gt {gt.shape}")
    diff = (pred - gt).abs()                                # (..., H, W, 3)
    if mask is None:
        return diff.mean()
    if mask.shape != pred.shape[:-1]:
        raise ValueError(
            f"masked_l1_loss: mask shape {tuple(mask.shape)} must match "
            f"image spatial dims {tuple(pred.shape[:-1])}"
        )
    m = mask.to(pred.dtype).unsqueeze(-1)                   # (..., H, W, 1)
    num = (diff * m).sum()
    den = m.sum() * pred.shape[-1] + 1e-8
    return num / den


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_window(
    window_size: int, sigma: float, device, dtype,
) -> torch.Tensor:
    """Separable Gaussian kernel of shape (window_size, window_size)."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    *,
    window_size: int   = 11,
    sigma:       float = 1.5,
) -> torch.Tensor:
    """Structural similarity between two ``(H, W, 3)`` images in ``[0, 1]``.

    Returns a scalar in ``[0, 1]`` (1 = identical). Implementation is the
    standard Wang et al. 2004 SSIM with a Gaussian window, applied
    independently per channel and then averaged.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"ssim: pred {pred.shape} != gt {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"ssim: expected (H, W, 3), got {tuple(pred.shape)}")

    # (H, W, C) -> (1, C, H, W) for conv2d
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)

    window = _gaussian_window(window_size, sigma, p.device, p.dtype)
    window = window.expand(3, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_p  = F.conv2d(p,     window, padding=pad, groups=3)
    mu_g  = F.conv2d(g,     window, padding=pad, groups=3)
    mu_p2 = mu_p * mu_p
    mu_g2 = mu_g * mu_g
    mu_pg = mu_p * mu_g
    sigma_p2 = F.conv2d(p * p, window, padding=pad, groups=3) - mu_p2
    sigma_g2 = F.conv2d(g * g, window, padding=pad, groups=3) - mu_g2
    sigma_pg = F.conv2d(p * g, window, padding=pad, groups=3) - mu_pg

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_pg + c1) * (2 * sigma_pg + c2)
    den = (mu_p2 + mu_g2 + c1) * (sigma_p2 + sigma_g2 + c2)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# Combined photometric loss
# ---------------------------------------------------------------------------

def photometric_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    *,
    mask:         Optional[torch.Tensor] = None,
    lambda_ssim:  float                  = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard 3DGS photometric loss: ``(1-λ)·L1 + λ·(1-SSIM)``.

    SSIM is intentionally **not** masked: masking a sliding-window metric
    introduces edge artefacts. The L1 part absorbs the dynamic-region
    filtering instead, which is sufficient in practice.

    Returns
    -------
    loss
        Scalar loss tensor (requires grad if inputs do).
    components
        ``{"l1", "ssim", "loss"}`` as Python floats — for logging.
    """
    l1  = masked_l1_loss(pred, gt, mask)
    s   = ssim(pred, gt)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - s)
    return loss, {
        "l1":   float(l1.detach().item()),
        "ssim": float(s.detach().item()),
        "loss": float(loss.detach().item()),
    }


# ---------------------------------------------------------------------------
# Eval metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak-signal-to-noise ratio in dB for ``(H, W, 3)`` tensors in ``[0, 1]``."""
    mse = ((pred - gt) ** 2).mean()
    if float(mse.item()) <= 0.0:
        return 100.0
    return float(-10.0 * torch.log10(mse).item())

