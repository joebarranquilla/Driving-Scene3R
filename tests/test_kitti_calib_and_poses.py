"""Tests for ``semantic_gs.geometry.poses`` — calib + poses parsing and
the cam-0 → cam-i baseline shift.
"""

from __future__ import annotations

import numpy as np

from semantic_gs.geometry.poses import (
    cam_i_to_world_from_cam0,
    parse_kitti_calib,
    parse_kitti_poses,
)


# A trimmed-down version of the real calib.txt at the repo root.
_REAL_CALIB_TXT = """\
P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03
P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.372877000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 2.369057000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.915215000000e-03
"""


def test_parse_kitti_calib_real_values(tmp_path):
    calib_path = tmp_path / "calib.txt"
    calib_path.write_text(_REAL_CALIB_TXT)
    calib = parse_kitti_calib(calib_path)

    assert set(calib) == {"P0", "P1", "P2", "P3"}
    P2 = calib["P2"]
    assert P2.shape == (3, 4) and P2.dtype == np.float64
    # spot-check a few values
    assert np.isclose(P2[0, 0], 718.856)
    assert np.isclose(P2[0, 2], 607.1928)
    assert np.isclose(P2[0, 3], 45.38225)


def test_parse_kitti_poses_roundtrip(tmp_path):
    # Three deterministic 3x4 transforms: identity, +X translation, +Z translation.
    raw = np.array([
        [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0],     # identity
        [1, 0, 0, 0.5, 0, 1, 0, 0,  0, 0, 1, 0],    # x += 0.5
        [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 1.2],   # z += 1.2
    ], dtype=np.float64)
    path = tmp_path / "00.txt"
    np.savetxt(path, raw, fmt="%.12e")

    poses = parse_kitti_poses(path)
    assert poses.shape == (3, 4, 4) and poses.dtype == np.float64
    np.testing.assert_array_equal(poses[:, 3, :], np.tile([0, 0, 0, 1], (3, 1)))
    np.testing.assert_allclose(poses[0, :3, :], raw[0].reshape(3, 4))
    np.testing.assert_allclose(poses[1, :3, 3], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(poses[2, :3, 3], [0.0, 0.0, 1.2])


def test_cam_i_to_world_identity_pose_gives_negative_baseline():
    """With cam-0 at world origin, cam-2 must sit at the negative of the
    translation that ``P2`` encodes for cam-0 in cam-2 frame."""
    fx, fy, cx, cy = 700.0, 700.0, 600.0, 200.0
    baseline = 0.06  # 6 cm
    P2 = np.array([
        [fx, 0.0, cx, fx * baseline],
        [0.0, fy, cy, 0.0          ],
        [0.0, 0.0, 1.0, 0.0        ],
    ], dtype=np.float64)
    T_cam0_world = np.eye(4, dtype=np.float64)

    T_cam2_world = cam_i_to_world_from_cam0(T_cam0_world, P2)

    # cam0_in_cam2 = (+baseline, 0, 0)  ⇒  cam2 lives at (-baseline, 0, 0)
    # in cam-0's frame (== world frame when T_cam0_world = I).
    np.testing.assert_allclose(T_cam2_world[:3, 3], [-baseline, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T_cam2_world[:3, :3], np.eye(3))
    np.testing.assert_allclose(T_cam2_world[3, :],  [0, 0, 0, 1])


def test_cam_i_to_world_composes_with_nonidentity_pose():
    """The baseline shift must be applied in cam-0's frame, not in world."""
    fx, fy, cx, cy = 700.0, 700.0, 600.0, 200.0
    baseline = 0.06
    P2 = np.array([
        [fx, 0.0, cx, fx * baseline],
        [0.0, fy, cy, 0.0          ],
        [0.0, 0.0, 1.0, 0.0        ],
    ], dtype=np.float64)

    # cam-0 rotated 90° around world-Y and translated to (10, 0, 0).
    Ry = np.array([[ 0, 0, 1, 0],
                   [ 0, 1, 0, 0],
                   [-1, 0, 0, 0],
                   [ 0, 0, 0, 1]], dtype=np.float64)
    T_cam0_world = Ry.copy()
    T_cam0_world[:3, 3] = [10.0, 0.0, 0.0]

    T_cam2_world = cam_i_to_world_from_cam0(T_cam0_world, P2)

    # cam-2's origin in world = T_cam0_world @ (-baseline, 0, 0, 1)
    expected = T_cam0_world @ np.array([-baseline, 0, 0, 1.0])
    np.testing.assert_allclose(T_cam2_world[:3, 3], expected[:3], atol=1e-12)
    # Rotation unchanged because for rectified KITTI cams T_cami_to_cam0 is identity rot.
    np.testing.assert_allclose(T_cam2_world[:3, :3], T_cam0_world[:3, :3])

