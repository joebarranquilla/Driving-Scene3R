"""Hermetic CPU tests for :class:`semantic_gs.model.gaussians.GaussianModel`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semantic_gs.data.pointcloud import SemanticPointCloud
from semantic_gs.model.gaussians import GaussianModel, _inverse_sigmoid


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _tiny_pc(n: int = 128, seed: int = 0) -> SemanticPointCloud:
    rng = np.random.default_rng(seed)
    return SemanticPointCloud(
        xyz    = rng.normal(size=(n, 3)).astype(np.float32),
        rgb    = rng.uniform(size=(n, 3)).astype(np.float32),
        labels = np.zeros(n, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_from_pc_has_expected_shapes_and_grads() -> None:
    pc = _tiny_pc(n=64)
    m  = GaussianModel.from_semantic_pointcloud(pc)
    assert m.num_gaussians == 64
    assert m.means.shape          == (64, 3)
    assert m.quats_raw.shape      == (64, 4)
    assert m.log_scales.shape     == (64, 3)
    assert m.opacity_logits.shape == (64,)
    assert m.colors.shape         == (64, 3)
    for p in (m.means, m.quats_raw, m.log_scales, m.opacity_logits, m.colors):
        assert p.dtype == torch.float32
        assert p.requires_grad


def test_activated_satisfies_constraints() -> None:
    m = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=32))
    g = m.activated()
    assert (g.opacities >= 0).all() and (g.opacities <= 1).all()
    assert (g.scales > 0).all()
    qn = g.quats.norm(dim=-1)
    assert torch.allclose(qn, torch.ones_like(qn), atol=1e-5)
    assert (g.colors >= 0).all() and (g.colors <= 1).all()


def test_initial_opacity_matches_request() -> None:
    m = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=16), opacity_init=0.1)
    g = m.activated()
    assert torch.allclose(g.opacities, torch.full_like(g.opacities, 0.1), atol=1e-5)


def test_initial_quaternion_is_identity() -> None:
    m = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=8))
    g = m.activated()
    expected = torch.zeros_like(g.quats); expected[:, 0] = 1.0
    assert torch.allclose(g.quats, expected, atol=1e-6)


def test_param_groups_exposes_all_named_groups() -> None:
    m  = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=8))
    pg = m.param_groups({})
    names = {g["name"] for g in pg}
    assert names == {"means", "colors", "opacity", "quats", "scales"}
    # Every parameter must be wired into exactly one group.
    n_params_in_groups = sum(len(g["params"]) for g in pg)
    assert n_params_in_groups == sum(1 for _ in m.parameters())


def test_empty_point_cloud_raises() -> None:
    pc = SemanticPointCloud(
        xyz=np.zeros((0, 3), np.float32),
        rgb=np.zeros((0, 3), np.float32),
        labels=np.zeros((0,), np.int32),
    )
    with pytest.raises(ValueError):
        GaussianModel.from_semantic_pointcloud(pc)


def test_loss_backward_runs_on_cpu() -> None:
    """Gradient flow through the activated tensors works without CUDA."""
    m = GaussianModel.from_semantic_pointcloud(_tiny_pc(n=24))
    g = m.activated()
    fake_loss = g.means.sum() + g.scales.sum() + g.opacities.sum() \
                + g.quats.sum() + g.colors.sum()
    fake_loss.backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_inverse_sigmoid_round_trip() -> None:
    for x in (0.05, 0.1, 0.5, 0.9):
        logit = _inverse_sigmoid(x)
        recovered = float(torch.sigmoid(torch.tensor(logit)))
        assert abs(recovered - x) < 1e-6
    with pytest.raises(ValueError):
        _inverse_sigmoid(0.0)
    with pytest.raises(ValueError):
        _inverse_sigmoid(1.0)


def test_shape_validation_rejects_bad_inputs() -> None:
    n = 10
    bad_means = torch.zeros(n, 2)      # wrong last dim
    good      = torch.zeros(n, 3)
    with pytest.raises(ValueError):
        GaussianModel(bad_means, good, good)
    with pytest.raises(ValueError):
        GaussianModel(good, torch.zeros(n + 1, 3), good)
    with pytest.raises(ValueError):
        GaussianModel(good, good, torch.zeros(n, 2))

