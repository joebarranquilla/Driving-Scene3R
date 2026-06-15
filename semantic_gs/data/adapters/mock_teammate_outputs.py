"""Write synthetic data in the **exact** on-disk layout produced by the
teammates' inference scripts. Used by tests and demos so this module
can be exercised end-to-end without any real dataset or GPU run.

Layout produced by :func:`materialize_dummy_as_kitti_layout`::

    <root>/
        sequences/<seq_id>/
            calib.txt                 # P0..P3
            image_2/<stem>.png        # RGB frames
        poses/<seq_id>.txt            # one cam0->world 3x4 per frame
        depth/<seq_id>/<stem>.npz     # teammate depth-NPZ contract
        panoptic/<seq_id>/<stem>.npz  # teammate panoptic-NPZ contract
        panoptic/id2label.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_gs.data.adapters.depth_npz import DEPTH_KEY
from semantic_gs.data.adapters.panoptic_npz import PANOPTIC_KEYS
from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.pointcloud import SemanticPointCloud


# ---------------------------------------------------------------------------
# Low-level writers (one file each)
# ---------------------------------------------------------------------------

def write_depth_npz(path: str | Path, depth: np.ndarray) -> None:
    """Write a depth array under the teammate's ``"depth"`` key."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{DEPTH_KEY: depth.astype(np.float32, copy=False)})


def write_panoptic_npz(
    path: str | Path,
    panoptic_seg: np.ndarray,
    segment_ids:  np.ndarray,
    label_ids:    np.ndarray,
    scores:       np.ndarray,
) -> None:
    """Write the four-key panoptic NPZ exactly as the teammate script does."""
    arrays = {
        PANOPTIC_KEYS[0]: panoptic_seg.astype(np.int32,   copy=False),
        PANOPTIC_KEYS[1]: segment_ids .astype(np.int32,   copy=False),
        PANOPTIC_KEYS[2]: label_ids   .astype(np.int32,   copy=False),
        PANOPTIC_KEYS[3]: scores      .astype(np.float32, copy=False),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_id2label_json(path: str | Path, id2label: dict[int, str]) -> None:
    """Write ``id2label.json`` with stringified integer keys (teammate format)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump({str(k): str(v) for k, v in id2label.items()}, fh, indent=2)


def write_kitti_calib_txt(path: str | Path, matrices: dict[str, np.ndarray]) -> None:
    """Write a KITTI-style ``calib.txt`` containing the given 3x4 matrices."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        for key, M in matrices.items():
            M = np.asarray(M, dtype=np.float64)
            if M.shape != (3, 4):
                raise ValueError(f"{key}: expected (3, 4), got {M.shape}")
            row = " ".join(f"{v:.12e}" for v in M.reshape(-1))
            fh.write(f"{key}: {row}\n")


def write_kitti_poses_txt(path: str | Path, T_cam0_to_world: np.ndarray) -> None:
    """Write a KITTI-style ``poses/XX.txt``.

    ``T_cam0_to_world`` is ``(N, 4, 4) float64``; each row written is the
    first three rows flattened (12 floats), matching the official format.
    """
    if T_cam0_to_world.ndim != 3 or T_cam0_to_world.shape[1:] != (4, 4):
        raise ValueError(
            f"T_cam0_to_world must be (N, 4, 4); got {T_cam0_to_world.shape}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = T_cam0_to_world[:, :3, :].reshape(T_cam0_to_world.shape[0], 12)
    np.savetxt(path, flat, fmt="%.12e")


def write_image_2_png(path: str | Path, rgb: np.ndarray) -> None:
    """Save an RGB ``uint8 (H, W, 3)`` array as PNG (PIL, RGB order)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be uint8 (H, W, 3); got {rgb.shape} {rgb.dtype}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Semantic-point-cloud writers (mirror scripts/utils.py::save_ply +
# scripts/lift_to_semantic_pointcloud.py --save_npz layout)
# ---------------------------------------------------------------------------

def write_semantic_pointcloud_ply(
    path: str | Path,
    xyz: np.ndarray,                   # (N, 3) float32 metres
    rgb: np.ndarray,                   # (N, 3) float32 [0, 1]
    labels: np.ndarray | None = None,  # (N,)   int32 — class IDs (optional)
) -> None:
    """Write a semantic point cloud as ASCII PLY in the teammate's exact format.

    Property order: ``x y z red green blue [label]`` where colour channels
    are stored as ``uchar`` (0-255) and the optional ``label`` is ``int``.
    """
    n = int(xyz.shape[0])
    if xyz.shape != (n, 3) or rgb.shape != (n, 3):
        raise ValueError(f"xyz/rgb must be (N, 3); got {xyz.shape}, {rgb.shape}")
    if labels is not None and labels.shape != (n,):
        raise ValueError(f"labels must be (N,); got {labels.shape}")

    cols_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    with_labels = labels is not None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        fh.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        if with_labels:
            fh.write("property int label\n")
        fh.write("end_header\n")
        if with_labels:
            for (x, y, z), (r, g, b), lbl in zip(xyz, cols_u8, labels):
                fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {int(lbl)}\n")
        else:
            for (x, y, z), (r, g, b) in zip(xyz, cols_u8):
                fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def write_semantic_pointcloud_npz(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Write the teammate's optional NPZ: keys ``xyz``, ``colors``, ``labels``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz   = xyz.astype(np.float32, copy=False),
        colors= rgb.astype(np.float32, copy=False),
        labels= labels.astype(np.int32, copy=False),
    )


# ---------------------------------------------------------------------------
# High-level fixture builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KITTILayoutPaths:
    """Resolved paths returned by :func:`materialize_dummy_as_kitti_layout`."""

    sequence_dir: Path     # contains image_2/, calib.txt
    depth_dir:    Path     # per-frame .npz
    panoptic_dir: Path     # per-frame .npz
    pose_path:    Path     # the single .txt file for this sequence
    id2label_path: Path    # panoptic_dir.parent / "id2label.json"


# A KITTI-shaped P2 with a 6 cm baseline along +x.
# Used to fabricate a calib.txt that exercises the cam0->cam2 baseline math.
def make_synthetic_kitti_calib(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline_x_m: float = 0.06,
) -> dict[str, np.ndarray]:
    """Return P0..P3 as if all four cameras shared (fx, fy, cx, cy).

    P2 is offset along +x by ``+fx * baseline_x_m``; P3 is offset by
    ``-fx * baseline_x_m``. P0 and P1 have zero offset. This is enough
    to exercise the rectified-baseline code path without depending on a
    real dataset.
    """
    def P(t_x: float) -> np.ndarray:
        return np.array([
            [fx, 0.0, cx, fx * t_x],
            [0.0, fy, cy, 0.0     ],
            [0.0, 0.0, 1.0, 0.0   ],
        ], dtype=np.float64)
    return {"P0": P(0.0), "P1": P(0.0), "P2": P(+baseline_x_m), "P3": P(-baseline_x_m)}


def materialize_dummy_as_kitti_layout(
    loader: DummySequenceLoader,
    root: str | Path,
    seq_id: str = "00",
) -> KITTILayoutPaths:
    """Persist every :class:`Frame` of ``loader`` as a KITTI-layout fixture.

    The dummy loader's intrinsics ``(fx, fy, cx, cy)`` are baked into the
    ``calib.txt``'s ``P2`` (the camera the rest of the pipeline uses).
    """
    root = Path(root)

    sequence_dir = root / "sequences" / seq_id
    image_dir    = sequence_dir / "image_2"
    calib_path   = sequence_dir / "calib.txt"
    pose_path    = root / "poses" / f"{seq_id}.txt"
    depth_dir    = root / "depth" / seq_id
    pano_dir     = root / "panoptic" / seq_id
    id2label_path = root / "panoptic" / "id2label.json"

    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pano_dir.mkdir(parents=True, exist_ok=True)

    # --- calib.txt (P0..P3 sharing the dummy intrinsics) ----------------
    cam = loader[0].camera
    matrices = make_synthetic_kitti_calib(cam.fx, cam.fy, cam.cx, cam.cy)
    write_kitti_calib_txt(calib_path, matrices)

    # --- per-frame artefacts + accumulated poses (as cam0->world) -------
    # The dummy's Frame.T_cam_to_world is the cam-2 pose (since the dummy
    # camera is conceptually image_2 with the dummy intrinsics). To write
    # a teammate-shaped poses.txt we need the cam-0 pose; we invert the
    # rectified baseline shift derived from P2.
    P2 = matrices["P2"]
    K2 = P2[:3, :3]
    t_cam0_in_cam2 = np.linalg.solve(K2, P2[:, 3])
    T_cam0_to_cam2 = np.eye(4, dtype=np.float64)
    T_cam0_to_cam2[:3, 3] = t_cam0_in_cam2  # cam0 origin's position in cam2 frame

    poses_cam0 = np.zeros((len(loader), 4, 4), dtype=np.float64)
    id2label_for_file: dict[int, str] = dict(loader.id2label)

    for i in range(len(loader)):
        f = loader[i]
        stem = f.frame_id

        write_image_2_png(image_dir / f"{stem}.png", f.rgb)
        write_depth_npz   (depth_dir / f"{stem}.npz", f.depth)
        write_panoptic_npz(
            pano_dir / f"{stem}.npz",
            f.panoptic_seg, f.segment_ids, f.label_ids, f.scores,
        )

        # T_cam0_to_world = T_cam2_to_world @ T_cam0_to_cam2
        poses_cam0[i] = f.T_cam_to_world @ T_cam0_to_cam2

    write_kitti_poses_txt(pose_path, poses_cam0)
    write_id2label_json(id2label_path, id2label_for_file)

    return KITTILayoutPaths(
        sequence_dir=sequence_dir,
        depth_dir=depth_dir,
        panoptic_dir=pano_dir,
        pose_path=pose_path,
        id2label_path=id2label_path,
    )


# ---------------------------------------------------------------------------
# Semantic-point-cloud fixture: synthesize a teammate-shaped lift output
# from a DummySequenceLoader, without depending on open3d / scipy.
# ---------------------------------------------------------------------------

def _backproject_static_pixels(loader: DummySequenceLoader) -> SemanticPointCloud:
    """Naïve concat aggregation of static pixels across all dummy frames.

    DEV / TEST ONLY — production initialisation comes from the teammate's
    ``scripts/lift_to_semantic_pointcloud.py``. This helper exists solely
    so :func:`materialize_dummy_as_lift_output` can produce realistic
    fake teammate output for hermetic tests, without pulling in scipy/open3d.
    """
    cam = loader[0].camera
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    H, W = cam.height, cam.width
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    xyz_list:    list[np.ndarray] = []
    rgb_list:    list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    # Per-pixel class lookup needs the segment_id -> label_id map of each frame.
    for i in range(len(loader)):
        f = loader[i]
        mask = f.static_mask & np.isfinite(f.depth) & (f.depth > 0)
        if not mask.any():
            continue

        # Per-pixel class IDs
        label_map = np.zeros((H, W), dtype=np.int32)
        for sid, lid in zip(f.segment_ids.tolist(), f.label_ids.tolist()):
            label_map[f.panoptic_seg == sid] = int(lid)

        Z = f.depth[mask].astype(np.float32)
        X = ((u_grid[mask] - cx) * Z / fx).astype(np.float32)
        Y = ((v_grid[mask] - cy) * Z / fy).astype(np.float32)

        xyz_cam = np.stack([X, Y, Z], axis=-1)                          # (M, 3)
        R = f.T_cam_to_world[:3, :3].astype(np.float32)
        t = f.T_cam_to_world[:3,  3].astype(np.float32)
        xyz_world = xyz_cam @ R.T + t                                   # (M, 3)

        xyz_list.append(xyz_world)
        rgb_list.append((f.rgb[mask].astype(np.float32) / 255.0))
        labels_list.append(label_map[mask].astype(np.int32))

    if not xyz_list:
        empty = np.zeros((0, 3), dtype=np.float32)
        return SemanticPointCloud(
            xyz=empty,
            rgb=empty.copy(),
            labels=np.zeros((0,), dtype=np.int32),
        )

    return SemanticPointCloud(
        xyz=np.concatenate(xyz_list, axis=0),
        rgb=np.concatenate(rgb_list, axis=0),
        labels=np.concatenate(labels_list, axis=0),
    )


@dataclass(frozen=True)
class LiftOutputPaths:
    """Paths returned by :func:`materialize_dummy_as_lift_output`."""
    ply_path: Path
    npz_path: Path


def materialize_dummy_as_lift_output(
    loader: DummySequenceLoader,
    out_dir: str | Path,
    stem: str = "static_cloud",
) -> tuple[LiftOutputPaths, SemanticPointCloud]:
    """Persist a dummy loader's frames as a teammate-shaped lift output.

    Writes ``<out_dir>/<stem>.ply`` AND ``<out_dir>/<stem>.npz`` in the
    exact formats produced by ``scripts/lift_to_semantic_pointcloud.py``.
    Returns the paths and the in-memory :class:`SemanticPointCloud` used
    to produce them (so tests can do bit-exact round-trip comparisons).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc = _backproject_static_pixels(loader)
    ply_path = out_dir / f"{stem}.ply"
    npz_path = out_dir / f"{stem}.npz"
    write_semantic_pointcloud_ply(ply_path, pc.xyz, pc.rgb, pc.labels)
    write_semantic_pointcloud_npz(npz_path, pc.xyz, pc.rgb, pc.labels)
    return LiftOutputPaths(ply_path=ply_path, npz_path=npz_path), pc




