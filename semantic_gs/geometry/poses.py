"""KITTI calibration / pose parsing and cam-0 -> cam-i baseline shifting.

Pose convention (see ``semantic_gs/README.md``):
- A KITTI odometry pose file row is a 3x4 matrix ``T_cam0_to_world``: it
  maps a point in cam-0 frame to world coords. The world is defined as
  cam-0 at frame 0, so the first pose is always the identity (up to
  floating-point noise).
- For ``image_2`` (left color), we need ``T_cam2_to_world``. The KITTI
  rectified projection matrix ``P_i = K_i @ [I | t]`` encodes the
  position of cam-0 in cam-i frame via ``t = K_i^{-1} @ P_i[:, 3]``.
  The cam-i-to-cam-0 transform is therefore identity rotation with
  translation ``-t`` (negative because we flip the direction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# calib.txt
# ---------------------------------------------------------------------------

def parse_kitti_calib(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a KITTI odometry ``calib.txt`` file.

    Returns a dict ``{key: (3, 4) float64}`` for every row found. Common
    keys are ``P0, P1, P2, P3`` (projection matrices) and ``Tr`` (LiDAR
    -> cam-0). ``Tr`` is optional and only present in some sequences.
    """
    out: dict[str, np.ndarray] = {}
    path = Path(path)
    with path.open("r") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = np.fromstring(value, sep=" ", dtype=np.float64)
            if vals.size != 12:
                raise ValueError(
                    f"{path}:{line_no}: expected 12 floats for key {key!r}, "
                    f"got {vals.size}"
                )
            out[key.strip()] = vals.reshape(3, 4)
    if not out:
        raise ValueError(f"{path}: no projection rows parsed")
    return out


# ---------------------------------------------------------------------------
# poses/<seq>.txt
# ---------------------------------------------------------------------------

def parse_kitti_poses(path: str | Path) -> np.ndarray:
    """Parse a KITTI odometry ``poses/XX.txt`` file.

    Each non-empty line is a 12-number row-major 3x4 matrix that maps a
    point in cam-0 frame at time ``i`` into the **world** frame (cam-0
    at time 0).

    Returns
    -------
    poses : ``(N, 4, 4) float64`` — the bottom row ``[0, 0, 0, 1]`` is
        appended to every matrix.
    """
    path = Path(path)
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        # Single-line file: reshape to (1, 12) so the rest is uniform.
        raw = raw[None, :]
    if raw.shape[1] != 12:
        raise ValueError(
            f"{path}: expected 12 floats per row, got {raw.shape[1]}"
        )
    n = raw.shape[0]
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :] = raw.reshape(n, 3, 4)
    poses[:, 3, 3] = 1.0
    return poses


# ---------------------------------------------------------------------------
# cam-0 -> cam-i baseline shift
# ---------------------------------------------------------------------------

def cam_i_to_world_from_cam0(
    T_cam0_to_world: np.ndarray,
    P_i: np.ndarray,
) -> np.ndarray:
    """Convert a cam-0-to-world transform into a cam-i-to-world transform.

    Parameters
    ----------
    T_cam0_to_world
        ``(4, 4) float64`` — pose of cam-0 in world coordinates.
    P_i
        ``(3, 4) float64`` — KITTI rectified projection matrix
        ``P_i = K_i @ [I | t_cam0_in_cami]``.

    Returns
    -------
    T_cami_to_world : ``(4, 4) float64``
    """
    if T_cam0_to_world.shape != (4, 4):
        raise ValueError(f"T_cam0_to_world must be (4, 4), got {T_cam0_to_world.shape}")
    if P_i.shape != (3, 4):
        raise ValueError(f"P_i must be (3, 4), got {P_i.shape}")

    K_i = P_i[:3, :3]
    # t = K_i^{-1} @ P_i[:, 3] is the position of cam-0's origin expressed in cam-i frame.
    t_cam0_in_cami = np.linalg.solve(K_i, P_i[:, 3])

    # cam-i -> cam-0 is identity rotation with translation -t (since for
    # rectified KITTI cameras the relative rotation is identity).
    T_cami_to_cam0 = np.eye(4, dtype=np.float64)
    T_cami_to_cam0[:3, 3] = -t_cam0_in_cami

    return (T_cam0_to_world @ T_cami_to_cam0).astype(np.float64, copy=False)

