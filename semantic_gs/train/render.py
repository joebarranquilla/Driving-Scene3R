"""Thin wrapper around ``gsplat.rasterization`` for a single :class:`Frame`.

Camera conventions
------------------
* :class:`~semantic_gs.data.frame.Frame` stores **camera-to-world** (``c2w``).
* ``gsplat.rasterization`` expects **world-to-camera** (``w2c``) view
  matrices (a.k.a. ``viewmats``). We invert ``c2w`` here.
* Both ``Frame`` and ``gsplat`` use the OpenCV camera convention
  (``+X`` right, ``+Y`` down, ``+Z`` forward), so no axis re-mapping is
  needed beyond the c2w→w2c inversion.

This module **lazily imports** :mod:`gsplat`, which has CUDA-only
kernels. Importing :mod:`semantic_gs.train.render` on a CPU box is
therefore safe; the import only fails the first time
:func:`render_frame` is actually called.
"""

from __future__ import annotations

import numpy as np
import torch

from semantic_gs.data.frame import Frame
from semantic_gs.model.gaussians import GaussianModel


# ---------------------------------------------------------------------------
# Frame → (viewmat, K)
# ---------------------------------------------------------------------------

def frame_viewmat_K(
    frame: Frame, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(viewmat_w2c (4,4), K (3,3))`` as ``float32`` on ``device``."""
    T_c2w = frame.T_cam_to_world.astype(np.float64)
    T_w2c = np.linalg.inv(T_c2w)
    viewmat = torch.from_numpy(T_w2c.astype(np.float32)).to(device)
    K       = torch.from_numpy(frame.camera.K.astype(np.float32)).to(device)
    return viewmat, K


# ---------------------------------------------------------------------------
# Single-frame render
# ---------------------------------------------------------------------------

def render_frame(
    model: GaussianModel,
    frame: Frame,
    *,
    near_plane: float                              = 0.1,
    far_plane:  float                              = 200.0,
    bg_color:   tuple[float, float, float]         = (0.0, 0.0, 0.0),
) -> dict:
    """Render the 3DGS model from ``frame``'s camera viewpoint.

    Returns a dict with:
      * ``rgb``   – ``(H, W, 3)`` ``float32`` predicted image in ``[0, 1+]``
        (caller should ``clamp`` for display / loss)
      * ``alpha`` – ``(H, W, 1)`` ``float32`` per-pixel accumulated alpha
      * ``meta``  – the gsplat ``meta`` dict (contains ``means2d``,
                    ``depths``, ``radii``, ... for densification / debug).
    """
    from gsplat import rasterization  # noqa: PLC0415  CUDA-only, lazy

    device = model.means.device
    g       = model.activated()
    viewmat, K = frame_viewmat_K(frame, device)
    H, W    = frame.camera.height, frame.camera.width

    renders, alphas, meta = rasterization(
        means       = g.means,
        quats       = g.quats,
        scales      = g.scales,
        opacities   = g.opacities,
        colors      = g.colors,
        viewmats    = viewmat.unsqueeze(0),     # (1, 4, 4)
        Ks          = K.unsqueeze(0),           # (1, 3, 3)
        width       = W,
        height      = H,
        sh_degree   = None,                     # use colors directly as RGB
        near_plane  = near_plane,
        far_plane   = far_plane,
        packed      = False,
        backgrounds = torch.tensor([bg_color], device=device, dtype=torch.float32),
    )
    # renders: (1, H, W, 3), alphas: (1, H, W, 1)
    return {"rgb": renders[0], "alpha": alphas[0], "meta": meta}


