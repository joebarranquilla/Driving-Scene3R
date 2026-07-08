"""CPU-hermetic tests for :mod:`semantic_gs.scripts.render_replay`.

The gsplat rasterization itself is CUDA-only and covered by the live smoke
run; these tests pin the pure-numpy parts: the gaussian-PLY reader (as the
exact inverse of ``save_gaussians_ply``) and the object posing math.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_gs.export.ply import save_gaussians_ply
from semantic_gs.model.gaussians import GaussianModel
from semantic_gs.scripts.render_replay import (
    build_schedule,
    canonicalize_object,
    load_gaussian_ply,
    pose_object,
)


def _random_model(n: int = 32, seed: int = 0) -> GaussianModel:
    g = torch.Generator().manual_seed(seed)
    model = GaussianModel(
        means_init=torch.randn(n, 3, generator=g),
        colors_init=torch.rand(n, 3, generator=g),
        scales_init=torch.rand(n, 3, generator=g) * 0.5 + 0.01,
        opacity_init=0.37,
    )
    # Exercise non-default per-gaussian quats/opacities too.
    with torch.no_grad():
        model.quats_raw.copy_(torch.randn(n, 4, generator=g))
        model.opacity_logits.copy_(torch.randn(n, generator=g))
    return model


def test_load_gaussian_ply_inverts_exporter(tmp_path):
    model = _random_model()
    ply_path = tmp_path / "g.ply"
    save_gaussians_ply(model, ply_path)

    loaded = load_gaussian_ply(ply_path)
    act = model.activated()

    np.testing.assert_allclose(loaded["means"], act.means.detach().numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(loaded["scales"], act.scales.detach().numpy(),
                               rtol=1e-5)
    np.testing.assert_allclose(loaded["opacities"],
                               act.opacities.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(loaded["colors"],
                               act.colors.detach().numpy(), atol=1e-6)
    # Normalized quats may differ by sign per-row; compare up to sign.
    q_l, q_m = loaded["quats"], act.quats.detach().numpy()
    sign = np.sign(np.sum(q_l * q_m, axis=1, keepdims=True))
    np.testing.assert_allclose(q_l, q_m * sign, atol=1e-5)


def test_load_gaussian_ply_rejects_non_gaussian(tmp_path):
    from plyfile import PlyData, PlyElement
    arr = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    p = tmp_path / "plain.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(p)
    with pytest.raises(ValueError, match="missing"):
        load_gaussian_ply(p)


def _box_asset() -> dict[str, np.ndarray]:
    """Asset: a 1 x 2 x 4 box of gaussians offset from the origin."""
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 1, 3), np.linspace(0, 2, 5), np.linspace(0, 4, 9),
        indexing="ij",
    )
    means = np.stack([xs.ravel() + 10, ys.ravel() - 5, zs.ravel() + 7], -1)
    n = len(means)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return {
        "means": means.astype(np.float32),
        "colors": np.full((n, 3), 0.5, dtype=np.float32),
        "opacities": np.full(n, 0.9, dtype=np.float32),
        "scales": np.full((n, 3), 0.05, dtype=np.float32),
        "quats": quats,
    }


def test_canonicalize_object_centers_scales_aligns():
    car = canonicalize_object(_box_asset(), target_size=4.5)
    ext = car["means"].max(0) - car["means"].min(0)
    assert ext.max() == pytest.approx(4.5, rel=1e-5)      # longest side (z)
    assert ext[0] == pytest.approx(4.5 / 4, rel=1e-5)     # aspect preserved
    np.testing.assert_allclose(car["means"].mean(0), 0.0, atol=1e-5)
    assert car["scales"][0, 0] == pytest.approx(0.05 * 4.5 / 4, rel=1e-5)
    # M_ALIGN = 180-deg about +Z: identity quats become (0, 0, 0, 1).
    np.testing.assert_allclose(car["quats"][0], [0, 0, 0, 1], atol=1e-6)


def test_pose_object_translation_only():
    car = canonicalize_object(_box_asset(), target_size=4.5)
    means, quats = pose_object(car, 3.0, -1.0, 50.0, theta=0.0)
    np.testing.assert_allclose(means.mean(0), [3.0, -1.0, 50.0], atol=1e-5)
    np.testing.assert_allclose(quats, car["quats"], atol=1e-6)


def test_pose_object_quarter_turn_swaps_extents():
    """theta = pi/2 (heading +X) must turn the long (+Z) axis into +X."""
    car = canonicalize_object(_box_asset(), target_size=4.5)
    means, _ = pose_object(car, 0.0, 0.0, 0.0, theta=np.pi / 2)
    ext = means.max(0) - means.min(0)
    assert ext[0] == pytest.approx(4.5, rel=1e-4)          # was z, now x
    assert ext[2] == pytest.approx(4.5 / 4, rel=1e-4)      # was x, now z
    assert ext[1] == pytest.approx(4.5 / 2, rel=1e-4)      # y untouched


def test_canonicalize_yaw_offset_180_flips_x_and_z():
    """yaw_offset=180 must equal R_y(180) on top of the base alignment:
    x and z coordinates negate, y and all extents stay the same."""
    base = canonicalize_object(_box_asset(), target_size=4.5)
    flip = canonicalize_object(_box_asset(), target_size=4.5,
                               yaw_offset_deg=180.0)
    np.testing.assert_allclose(flip["means"][:, 0], -base["means"][:, 0],
                               atol=1e-5)
    np.testing.assert_allclose(flip["means"][:, 2], -base["means"][:, 2],
                               atol=1e-5)
    np.testing.assert_allclose(flip["means"][:, 1], base["means"][:, 1],
                               atol=1e-5)
    ext_b = base["means"].max(0) - base["means"].min(0)
    ext_f = flip["means"].max(0) - flip["means"].min(0)
    np.testing.assert_allclose(ext_f, ext_b, atol=1e-5)
    # Quats must stay unit-length after the composition.
    np.testing.assert_allclose(np.linalg.norm(flip["quats"], axis=1), 1.0,
                               atol=1e-5)


def test_build_schedule_overlapping_tracks():
    trajs = [
        {"frame_indices": [10, 11, 12, 13]},          # traj 0
        {"frame_indices": [12, 13, 14]},              # traj 1
        {"frame_indices": [50]},                      # traj 2 (disjoint)
    ]
    schedule, frame_range, max_active = build_schedule(trajs)
    assert frame_range == [10, 11, 12, 13, 14, 50]
    assert max_active == 2
    assert schedule[10] == [(0, 0)]
    assert schedule[12] == [(0, 2), (1, 0)]           # both active, own steps
    assert schedule[13] == [(0, 3), (1, 1)]
    assert schedule[14] == [(1, 2)]
    assert schedule[50] == [(2, 0)]
