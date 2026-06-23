"""Trainable 3D Gaussian Splatting model (vanilla 3DGS, RGB only).

Phase 3 contract: takes a :class:`~semantic_gs.data.pointcloud.SemanticPointCloud`
as initialisation and exposes the per-Gaussian parameters in a form ready
to feed into ``gsplat.rasterization``. **No semantics yet** — Phase 4 will
subclass / extend this with a per-Gaussian semantic-logit head.

Parameter parameterisation (standard 3DGS):

* ``means``          stored in metres (world coords)
* ``quats_raw``      unnormalised quaternion (wxyz) — normalised on use
* ``log_scales``     log-space scales — exp on use
* ``opacity_logits`` inverse-sigmoid of opacity — sigmoid on use
* ``colors``         RGB in ``[0, 1]``

Keeping everything in its unconstrained form lets the optimizer move
freely without violating positivity / unit-norm constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from semantic_gs.data.pointcloud import SemanticPointCloud


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: float) -> float:
    """Logit (inverse of sigmoid). ``x`` must be in ``(0, 1)``."""
    if not (0.0 < x < 1.0):
        raise ValueError(f"_inverse_sigmoid: x must be in (0, 1), got {x}")
    return float(np.log(x / (1.0 - x)))


def _estimate_initial_scales(
    xyz: np.ndarray,
    *,
    k: int = 3,
    min_scale_m: float = 1e-4,
    max_scale_m: float = 0.5,
) -> np.ndarray:
    """Per-point isotropic scale = mean distance to ``k`` nearest neighbours.

    This is the standard 3DGS init: each Gaussian is sized to overlap with
    its local neighbourhood. Clipped to ``[min_scale_m, max_scale_m]`` to
    avoid pathological blobs from a few isolated outlier points.

    Returns ``float32 (N,)``.
    """
    from scipy.spatial import cKDTree  # local import: scipy is a heavy dep

    n = xyz.shape[0]
    if n <= k + 1:
        return np.full((n,), max_scale_m * 0.2, dtype=np.float32)

    tree = cKDTree(xyz)
    # +1 because the first neighbour is the point itself (distance 0).
    dists, _ = tree.query(xyz, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    return np.clip(mean_d, min_scale_m, max_scale_m).astype(np.float32)


# ---------------------------------------------------------------------------
# activated-tensor bundle (what the rasterizer consumes)
# ---------------------------------------------------------------------------

@dataclass
class GaussianTensors:
    """Constraint-satisfied tensors ready to pass to ``gsplat.rasterization``.

    All fields share ``device`` and ``dtype=float32``.
    """

    means:     torch.Tensor   # (N, 3)        world-space, metres
    quats:     torch.Tensor   # (N, 4) wxyz   unit-norm
    scales:    torch.Tensor   # (N, 3)        positive, metres
    opacities: torch.Tensor   # (N,)          in [0, 1]
    colors:    torch.Tensor   # (N, 3)        RGB in [0, 1]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class GaussianModel(nn.Module):
    """A bag of trainable 3D Gaussians with per-Gaussian RGB.

    See module docstring for the parameterisation conventions.
    """

    def __init__(
        self,
        means_init:    torch.Tensor,    # (N, 3) float32
        colors_init:   torch.Tensor,    # (N, 3) float32 in [0, 1]
        scales_init:   torch.Tensor,    # (N, 3) float32 metres (positive)
        opacity_init:  float = 0.1,
    ) -> None:
        super().__init__()

        if means_init.ndim != 2 or means_init.shape[1] != 3:
            raise ValueError(
                f"means_init must be (N, 3), got {tuple(means_init.shape)}"
            )
        n = int(means_init.shape[0])
        if n == 0:
            raise ValueError("GaussianModel: refusing to init with 0 Gaussians")
        if colors_init.shape != (n, 3):
            raise ValueError(
                f"colors_init must be ({n}, 3), got {tuple(colors_init.shape)}"
            )
        if scales_init.shape != (n, 3):
            raise ValueError(
                f"scales_init must be ({n}, 3), got {tuple(scales_init.shape)}"
            )

        # identity quaternion (wxyz)
        quats = torch.zeros(n, 4, dtype=torch.float32)
        quats[:, 0] = 1.0

        log_scales    = torch.log(torch.clamp(scales_init.float(), min=1e-6))
        opacity_logit = torch.full(
            (n,), _inverse_sigmoid(opacity_init), dtype=torch.float32
        )

        self.means          = nn.Parameter(means_init.float())
        self.quats_raw      = nn.Parameter(quats)
        self.log_scales     = nn.Parameter(log_scales)
        self.opacity_logits = nn.Parameter(opacity_logit)
        self.colors         = nn.Parameter(colors_init.float().clamp(0.0, 1.0))

    # ------------------------------------------------------------------
    @classmethod
    def from_semantic_pointcloud(
        cls,
        pc: SemanticPointCloud,
        *,
        opacity_init:     float = 0.1,
        scale_k:          int   = 3,
        max_init_scale_m: float = 0.5,
    ) -> "GaussianModel":
        """Build a model from a teammate-produced semantic point cloud.

        Initial Gaussian scale is the mean distance to ``scale_k`` nearest
        neighbours (capped at ``max_init_scale_m``). RGB is copied
        verbatim from the cloud; opacity is constant ``opacity_init``;
        rotation is identity.
        """
        if pc.n_points == 0:
            raise ValueError(
                "Cannot initialise GaussianModel from an empty point cloud"
            )
        scales_1d = _estimate_initial_scales(
            pc.xyz, k=scale_k, max_scale_m=max_init_scale_m,
        )
        scales_3d = np.repeat(scales_1d[:, None], 3, axis=1)    # (N, 3) isotropic
        return cls(
            means_init   = torch.from_numpy(pc.xyz.astype(np.float32)),
            colors_init  = torch.from_numpy(pc.rgb.astype(np.float32)),
            scales_init  = torch.from_numpy(scales_3d),
            opacity_init = opacity_init,
        )

    # ------------------------------------------------------------------
    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def activated(self) -> GaussianTensors:
        """Apply activations to get tensors the rasterizer wants."""
        return GaussianTensors(
            means     = self.means,
            quats     = torch.nn.functional.normalize(self.quats_raw, dim=-1),
            scales    = torch.exp(self.log_scales),
            opacities = torch.sigmoid(self.opacity_logits),
            colors    = self.colors.clamp(0.0, 1.0),
        )

    # ------------------------------------------------------------------
    def param_groups(self, lrs: dict[str, float] | None = None) -> list[dict]:
        """Standard 3DGS Adam param groups with per-tensor learning rates.

        Default LRs are the values from the Inria 3DGS paper. The ``means``
        LR is scene-scale-dependent in the original paper; we use the
        constant value and rely on a sensible point-cloud bounding box.
        """
        lrs = lrs or {}
        return [
            {"params": [self.means],          "lr": lrs.get("means",   1.6e-4), "name": "means"},
            {"params": [self.colors],         "lr": lrs.get("colors",  2.5e-3), "name": "colors"},
            {"params": [self.opacity_logits], "lr": lrs.get("opacity", 5.0e-2), "name": "opacity"},
            {"params": [self.quats_raw],      "lr": lrs.get("quats",   1.0e-3), "name": "quats"},
            {"params": [self.log_scales],     "lr": lrs.get("scales",  5.0e-3), "name": "scales"},
        ]

