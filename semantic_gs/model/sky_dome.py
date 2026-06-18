"""Synthetic sky-dome (and optional ground) Gaussians for free-view rendering.

Why this exists
---------------
The static driving-scene cloud has **no sky** — sky pixels have unreliable
stereo depth and are excluded upstream (see ``data/static_mask.py`` and the
teammate lift script). With an empty background, free-view orbits render the
sky region as flat black (or the viewer's default), which looks broken.

Rather than re-colouring garbage-depth sky points (which produces blue
*floaters* at wrong distances), we add a **parallax-free dome** of Gaussians at
a large fixed radius around the scene:

* Sky = upper hemisphere (world "up"), vertical gradient horizon→zenith.
* Optional ground = lower hemisphere, neutral colour, so orbiting *below* the
  horizon does not reveal a hard blue void.

The dome is **synthetic backdrop geometry**, not reconstructed scene content.
It is large enough never to intersect real geometry and has no parallax, so it
behaves like a skybox while remaining a normal part of the exported 3DGS PLY
(viewer-independent — shows up identically in SuperSplat / SIBR / your viewer).

World-frame convention
----------------------
This project's world frame inherits the KITTI camera convention where **+Y is
down** (gravity), so the default "up" direction is ``(0, -1, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkyDomeConfig:
    """Parameters controlling the synthetic sky-dome backdrop.

    Attributes
    ----------
    enabled
        Master switch. When ``False`` no dome is built.
    n_points
        Number of points sampled on the *full* sphere before hemisphere
        masking. The kept count is ~half this for a sky-only dome.
    radius_scale
        Dome radius = ``radius_scale * scene_radius`` (clamped). ``scene_radius``
        is the half-diagonal of the init-cloud bounding box.
    radius_min_m, radius_max_m
        Hard clamps on the dome radius (metres). ``radius_max_m`` should stay
        below the renderer ``far_plane`` so the dome is not clipped.
    up
        World-space "up" unit direction. Default ``(0, -1, 0)`` because +Y is
        down in this project's world frame.
    horizon_skirt
        How far *below* the horizon (in ``elevation`` units, 0..1) to keep sky
        points, giving a small overlap so there is no hard seam at the horizon.
    zenith_color, horizon_color
        RGB in ``[0, 1]`` for straight-up and at-the-horizon; linearly blended
        by elevation.
    include_ground
        If ``True``, also keep the lower hemisphere as a neutral ground shell.
    ground_color
        RGB in ``[0, 1]`` used for the ground shell (and the below-horizon
        blend target).
    opacity
        Per-Gaussian opacity in ``(0, 1)``. High so the dome reads as an opaque
        backdrop rather than a translucent haze.
    scale_mult
        Multiplier on the inter-point spacing used as the isotropic Gaussian
        scale. ``>1`` makes neighbouring dome splats overlap (no pin-holes).
    """

    enabled:       bool                      = False
    n_points:      int                       = 40_000
    radius_scale:  float                     = 3.0
    radius_min_m:  float                     = 30.0
    radius_max_m:  float                     = 180.0
    up:            tuple[float, float, float] = (0.0, -1.0, 0.0)
    horizon_skirt: float                     = 0.08
    zenith_color:  tuple[float, float, float] = (0.25, 0.45, 0.85)
    horizon_color: tuple[float, float, float] = (0.70, 0.82, 0.92)
    include_ground: bool                     = False
    ground_color:  tuple[float, float, float] = (0.45, 0.45, 0.45)
    opacity:       float                     = 0.95
    scale_mult:    float                     = 1.6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``(n, 3)`` near-uniformly distributed unit vectors on the sphere."""
    if n < 1:
        raise ValueError(f"_fibonacci_sphere: n must be >= 1, got {n}")
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)              # polar angle, uniform in area
    golden = np.pi * (1.0 + 5.0 ** 0.5)             # golden-angle increment
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _resolve_radius(scene_radius: float, cfg: SkyDomeConfig, far_plane: float) -> float:
    """Clamp the dome radius into ``[radius_min, min(radius_max, 0.9*far_plane)]``."""
    upper = min(cfg.radius_max_m, 0.9 * float(far_plane))
    upper = max(upper, cfg.radius_min_m)            # never invert the interval
    r = cfg.radius_scale * float(max(scene_radius, 0.0))
    return float(np.clip(r, cfg.radius_min_m, upper))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def make_sky_dome_gaussians(
    center: np.ndarray,
    scene_radius: float,
    cfg: SkyDomeConfig,
    far_plane: float,
) -> dict[str, np.ndarray]:
    """Build dome Gaussian parameters as plain numpy arrays.

    Parameters
    ----------
    center
        ``(3,)`` world-space centre the dome is built around (scene bbox centre).
    scene_radius
        Scene half-diagonal (metres); drives the dome radius via
        ``cfg.radius_scale``.
    cfg
        :class:`SkyDomeConfig`.
    far_plane
        Renderer far-plane (metres); the dome radius is kept below it.

    Returns
    -------
    dict with float32 arrays:
        ``means``     ``(M, 3)`` world-space positions
        ``colors``    ``(M, 3)`` RGB in ``[0, 1]``
        ``scales``    ``(M, 3)`` positive isotropic scales (metres)
        ``opacities`` ``(M,)``   in ``(0, 1)``
        ``radius``    ``(1,)``   the resolved dome radius (diagnostics)
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = _resolve_radius(scene_radius, cfg, far_plane)

    up = np.asarray(cfg.up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    dirs = _fibonacci_sphere(int(cfg.n_points))             # (n, 3) unit
    elevation = dirs @ up                                   # +1 zenith, -1 nadir

    # ---- select sky (upper hemisphere + skirt) and optional ground ----------
    if cfg.include_ground:
        keep = np.ones(dirs.shape[0], dtype=bool)
    else:
        keep = elevation >= -float(cfg.horizon_skirt)
    dirs = dirs[keep]
    elevation = elevation[keep]
    m = dirs.shape[0]
    if m == 0:
        raise ValueError("make_sky_dome_gaussians: no points survived hemisphere mask")

    means = center[None, :] + radius * dirs                # (m, 3)

    # ---- vertical colour gradient -------------------------------------------
    zenith  = np.asarray(cfg.zenith_color,  dtype=np.float64)
    horizon = np.asarray(cfg.horizon_color, dtype=np.float64)
    ground  = np.asarray(cfg.ground_color,  dtype=np.float64)

    t_sky = np.clip(elevation, 0.0, 1.0)[:, None]          # 0 horizon → 1 zenith
    sky_col = horizon[None, :] + t_sky * (zenith - horizon)[None, :]

    t_gnd = np.clip(-elevation, 0.0, 1.0)[:, None]         # 0 horizon → 1 nadir
    gnd_col = horizon[None, :] + t_gnd * (ground - horizon)[None, :]

    colors = np.where(elevation[:, None] >= 0.0, sky_col, gnd_col)
    colors = np.clip(colors, 0.0, 1.0)

    # ---- scales: tile the sphere so neighbouring splats overlap -------------
    # Mean nearest-neighbour spacing on a Fibonacci sphere of N points is
    # ~3.0862 * R / sqrt(N). Scale each Gaussian by ``scale_mult`` of that so
    # the dome has no pin-holes.
    spacing = radius * 3.0862 / float(np.sqrt(cfg.n_points))
    scale = max(spacing * cfg.scale_mult, 1e-3)
    scales = np.full((m, 3), scale, dtype=np.float32)

    opacities = np.full((m,), float(np.clip(cfg.opacity, 1e-4, 1.0 - 1e-4)),
                        dtype=np.float32)

    return {
        "means":     means.astype(np.float32),
        "colors":    colors.astype(np.float32),
        "scales":    scales,
        "opacities": opacities,
        "radius":    np.asarray([radius], dtype=np.float32),
    }


def scene_center_radius(bbox_min: np.ndarray, bbox_max: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(center (3,), half_diagonal)`` from an axis-aligned bbox."""
    mn = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    mx = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    center = 0.5 * (mn + mx)
    half_diag = 0.5 * float(np.linalg.norm(mx - mn))
    return center.astype(np.float32), half_diag

