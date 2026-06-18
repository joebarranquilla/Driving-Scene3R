"""Hermetic CPU tests for the synthetic sky dome (no gsplat / CUDA needed)."""

from __future__ import annotations

import numpy as np
import torch

from semantic_gs.data.pointcloud import SemanticPointCloud
from semantic_gs.model.gaussians import GaussianModel
from semantic_gs.model.sky_dome import (
    SkyDomeConfig,
    make_sky_dome_gaussians,
    scene_center_radius,
)


# ---------------------------------------------------------------------------
# dome builder
# ---------------------------------------------------------------------------

def test_dome_points_lie_on_sphere_of_resolved_radius():
    cfg = SkyDomeConfig(enabled=True, n_points=4000)
    center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dome = make_sky_dome_gaussians(center, scene_radius=10.0, cfg=cfg, far_plane=200.0)

    radius = float(dome["radius"][0])
    r = np.linalg.norm(dome["means"] - center[None, :], axis=1)
    np.testing.assert_allclose(r, radius, rtol=1e-4, atol=1e-3)
    # radius_scale(3) * scene_radius(10) = 30, within [30, 180] clamp.
    assert abs(radius - 30.0) < 1e-3


def test_sky_only_dome_is_upper_hemisphere():
    # up = (0, -1, 0): "up" means negative world-Y, so sky points have y < center.
    cfg = SkyDomeConfig(enabled=True, n_points=4000, horizon_skirt=0.08)
    center = np.zeros(3, dtype=np.float32)
    dome = make_sky_dome_gaussians(center, scene_radius=10.0, cfg=cfg, far_plane=200.0)
    radius = float(dome["radius"][0])
    # Allow the small horizon skirt below y=0.
    assert (dome["means"][:, 1] <= radius * cfg.horizon_skirt + 1e-4).all()


def test_ground_shell_adds_lower_hemisphere_points():
    common = dict(n_points=4000)
    sky = make_sky_dome_gaussians(
        np.zeros(3, np.float32), 10.0,
        SkyDomeConfig(enabled=True, include_ground=False, **common), 200.0,
    )
    full = make_sky_dome_gaussians(
        np.zeros(3, np.float32), 10.0,
        SkyDomeConfig(enabled=True, include_ground=True, **common), 200.0,
    )
    assert full["means"].shape[0] > sky["means"].shape[0]
    # Ground shell must contain clearly-below-horizon points (world +Y is down).
    assert (full["means"][:, 1] > 1.0).any()


def test_dome_outputs_are_valid_ranges_and_dtypes():
    cfg = SkyDomeConfig(enabled=True, n_points=2000)
    dome = make_sky_dome_gaussians(np.zeros(3, np.float32), 10.0, cfg, 200.0)
    assert dome["means"].dtype == np.float32
    assert dome["colors"].dtype == np.float32
    assert (dome["colors"] >= 0.0).all() and (dome["colors"] <= 1.0).all()
    assert (dome["scales"] > 0.0).all()
    assert (dome["opacities"] > 0.0).all() and (dome["opacities"] < 1.0).all()


def test_radius_is_clamped_below_far_plane():
    cfg = SkyDomeConfig(enabled=True, n_points=1000, radius_scale=100.0)
    dome = make_sky_dome_gaussians(np.zeros(3, np.float32), 50.0, cfg, far_plane=50.0)
    # radius_scale*scene_radius = 5000, but clamped to min(radius_max, 0.9*far).
    assert float(dome["radius"][0]) <= 0.9 * 50.0 + 1e-4


def test_scene_center_radius_from_bbox():
    mn = np.array([-2.0, -4.0, 0.0], dtype=np.float32)
    mx = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    center, half_diag = scene_center_radius(mn, mx)
    np.testing.assert_allclose(center, [0.0, 0.0, 3.0], atol=1e-5)
    assert abs(half_diag - 0.5 * np.linalg.norm(mx - mn)) < 1e-5


# ---------------------------------------------------------------------------
# model integration
# ---------------------------------------------------------------------------

def _tiny_pc(n: int = 64, seed: int = 0) -> SemanticPointCloud:
    rng = np.random.default_rng(seed)
    return SemanticPointCloud(
        xyz    = rng.normal(size=(n, 3)).astype(np.float32),
        rgb    = rng.uniform(size=(n, 3)).astype(np.float32),
        labels = np.zeros(n, dtype=np.int32),
    )


def test_append_gaussians_grows_model_and_keeps_grads():
    model = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=64))
    n0 = model.num_gaussians

    cfg = SkyDomeConfig(enabled=True, n_points=2000)
    dome = make_sky_dome_gaussians(np.zeros(3, np.float32), 5.0, cfg, 200.0)
    m = dome["means"].shape[0]

    model.append_gaussians(
        means=dome["means"], colors=dome["colors"],
        scales=dome["scales"], opacities=dome["opacities"],
    )
    assert model.num_gaussians == n0 + m

    # All parameter tensors grew consistently and remain trainable leaves.
    for p in (model.means, model.colors, model.log_scales,
              model.opacity_logits, model.quats_raw):
        assert p.shape[0] == n0 + m
        assert p.requires_grad and p.is_leaf

    # Activations round-trip: appended scales/opacities recover their inputs.
    g = model.activated()
    np.testing.assert_allclose(
        g.scales[n0:].detach().numpy(), dome["scales"], rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        g.opacities[n0:].detach().numpy(), dome["opacities"], rtol=1e-3, atol=1e-3,
    )
    # Identity quaternion (wxyz) for the appended dome Gaussians.
    np.testing.assert_allclose(
        g.quats[n0:].detach().numpy(),
        np.tile([1.0, 0.0, 0.0, 0.0], (m, 1)), atol=1e-6,
    )


def test_append_gaussians_validates_shapes():
    model = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=16))
    import pytest
    with pytest.raises(ValueError):
        model.append_gaussians(
            means=torch.zeros(4, 3), colors=torch.zeros(4, 3),
            scales=torch.ones(4, 3), opacities=torch.full((3,), 0.5),  # wrong M
        )

