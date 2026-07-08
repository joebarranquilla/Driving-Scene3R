"""Tests for the shared seg-source loader factory + unproject helper."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.adapters import build_kitti_loader
from semantic_gs.geometry.cameras import PinholeCamera, unproject_pixels


def test_factory_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="seg-source"):
        build_kitti_loader("m2f", sequence_dir=tmp_path, depth_dir=tmp_path,
                           pose_path=tmp_path / "p.txt")


def test_factory_sam3_names_missing_flags(tmp_path):
    with pytest.raises(ValueError, match=r"--sam3-dir"):
        build_kitti_loader("sam3", sequence_dir=tmp_path, depth_dir=tmp_path,
                           pose_path=tmp_path / "p.txt")


def test_factory_mask2former_names_missing_flags(tmp_path):
    with pytest.raises(ValueError, match=r"--pano-dir"):
        build_kitti_loader("mask2former", sequence_dir=tmp_path,
                           depth_dir=None, pose_path=None)


def test_unproject_pixels_pinhole_inverse():
    cam = PinholeCamera(width=640, height=480, fx=500.0, fy=400.0,
                        cx=320.0, cy=240.0)
    # Principal point at any depth -> on the optical axis.
    pts = unproject_pixels(cam, np.array([320.0]), np.array([240.0]),
                           np.array([7.0]))
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 7.0], atol=1e-12)
    # One pixel right of cx at depth Z -> X = Z / fx.
    pts = unproject_pixels(cam, np.array([321.0]), np.array([240.0]),
                           np.array([10.0]))
    np.testing.assert_allclose(pts[0], [10.0 / 500.0, 0.0, 10.0], atol=1e-12)
    # Round-trip: project back through K reproduces the pixel.
    u, v, z = np.array([100.5]), np.array([400.25]), np.array([23.0])
    p = unproject_pixels(cam, u, v, z)[0]
    assert p[0] / p[2] * cam.fx + cam.cx == pytest.approx(u[0])
    assert p[1] / p[2] * cam.fy + cam.cy == pytest.approx(v[0])
