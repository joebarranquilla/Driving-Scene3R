"""Trainable Gaussian-Splatting model (Phase 3+)."""

from semantic_gs.model.gaussians import GaussianModel, GaussianTensors
from semantic_gs.model.sky_dome import (
    SkyDomeConfig,
    make_sky_dome_gaussians,
    scene_center_radius,
)

__all__ = [
    "GaussianModel",
    "GaussianTensors",
    "SkyDomeConfig",
    "make_sky_dome_gaussians",
    "scene_center_radius",
]

