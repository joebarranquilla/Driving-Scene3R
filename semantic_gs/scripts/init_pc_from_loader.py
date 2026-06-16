"""Build a GS init point cloud by back-projecting a loader's static pixels.

POC alternative to ``scripts/lift_to_semantic_pointcloud.py`` for the
**SAM 3-only** path (the lift script is Mask2Former-only). It back-projects
the static pixels of a strided subset of frames into world space and writes
the teammate-format semantic NPZ (``xyz`` / ``colors`` / ``labels``) that
``train_gs --init-pc`` consumes.

This is intentionally simple (naive concatenation + optional subsample, no
voxel/ICP aggregation). It exists so the SAM 3 path can be run end-to-end
before a SAM 3-aware lift step is written. ``labels`` are set to ``-1``
("unknown / static") because the SAM 3 static mask carries no per-class
semantics; Phase 3 (RGB-only) training ignores labels anyway.

Example
-------
    python -m semantic_gs.scripts.init_pc_from_loader \\
        --kitti-odom-seq /storage/.../sequences/04 \\
        --depth-dir      /usr/prakt/<u>/depth_predictions/04 \\
        --sam3-dir       /usr/prakt/<u>/sam3_predictions/04 \\
        --pose-path      /storage/.../sequences/04/04.txt \\
        --out            /usr/prakt/<u>/semantic_clouds/seq04_sam3_static.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from semantic_gs.data.adapters.kitti_odom_sam3 import KITTISam3SequenceLoader
from semantic_gs.data.frame import Frame


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Back-project a SAM 3 loader's static pixels into a GS init cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kitti-odom-seq", required=True, metavar="SEQ_DIR")
    p.add_argument("--depth-dir",      required=True, metavar="DIR")
    p.add_argument("--sam3-dir",       required=True, metavar="DIR")
    p.add_argument("--pose-path",      required=True, metavar="FILE")
    p.add_argument("--concepts-path",  default=None, metavar="FILE")
    p.add_argument("--out",            required=True, type=Path)

    p.add_argument("--frame-stride",  type=int,   default=5,
                   help="Use every Nth frame (lower = denser cloud, slower).")
    p.add_argument("--pixel-stride",  type=int,   default=2,
                   help="Keep every Nth static pixel per frame (subsample).")
    p.add_argument("--far-plane",     type=float, default=80.0,
                   help="Discard points beyond this depth (metres). Drops sky.")
    p.add_argument("--near-plane",    type=float, default=0.5,
                   help="Discard points closer than this depth (metres).")
    p.add_argument("--boundary-margin", type=int, default=2,
                   help="Erode this many px around instance edges before masking.")
    p.add_argument("--max-points",    type=int,   default=3_000_000,
                   help="Random-subsample the final cloud to at most this many points.")
    p.add_argument("--seed",          type=int,   default=0)
    return p.parse_args()


def _backproject_frame(
    f: Frame, near: float, far: float, pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz_world (M,3) float32, rgb (M,3) float32 [0,1]) for one frame."""
    cam = f.camera
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    H, W = cam.height, cam.width

    keep = (
        f.static_mask
        & np.isfinite(f.depth)
        & (f.depth > near)
        & (f.depth < far)
    )
    if pixel_stride > 1:
        stride_mask = np.zeros((H, W), dtype=bool)
        stride_mask[::pixel_stride, ::pixel_stride] = True
        keep &= stride_mask

    if not keep.any():
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty.copy()

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    Z = f.depth[keep].astype(np.float32)
    X = ((u_grid[keep] - cx) * Z / fx).astype(np.float32)
    Y = ((v_grid[keep] - cy) * Z / fy).astype(np.float32)
    xyz_cam = np.stack([X, Y, Z], axis=-1)                       # (M, 3)

    R = f.T_cam_to_world[:3, :3].astype(np.float32)
    t = f.T_cam_to_world[:3,  3].astype(np.float32)
    xyz_world = xyz_cam @ R.T + t                                # (M, 3)

    rgb = f.rgb[keep].astype(np.float32) / 255.0
    return xyz_world.astype(np.float32), rgb


def main() -> int:
    args = _parse_args()

    loader = KITTISam3SequenceLoader(
        sequence_dir   = args.kitti_odom_seq,
        depth_dir      = args.depth_dir,
        sam3_dir       = args.sam3_dir,
        pose_path      = args.pose_path,
        concepts_path  = args.concepts_path,
        boundary_margin= args.boundary_margin,
    )
    print(f"[init] loader: {loader.name}  ({len(loader)} frames)")

    frame_indices = list(range(0, len(loader), max(1, args.frame_stride)))
    print(f"[init] back-projecting {len(frame_indices)} frames "
          f"(stride {args.frame_stride}, pixel-stride {args.pixel_stride}, "
          f"depth {args.near_plane}-{args.far_plane} m)")

    xyz_list: list[np.ndarray] = []
    rgb_list: list[np.ndarray] = []
    for k, idx in enumerate(frame_indices):
        f = loader[idx]
        xyz, rgb = _backproject_frame(f, args.near_plane, args.far_plane, args.pixel_stride)
        xyz_list.append(xyz)
        rgb_list.append(rgb)
        if k % 10 == 0 or k == len(frame_indices) - 1:
            print(f"  frame {f.frame_id}: {len(xyz):>8,} pts "
                  f"(running total {sum(len(a) for a in xyz_list):,})")

    xyz = np.concatenate(xyz_list, axis=0) if xyz_list else np.zeros((0, 3), np.float32)
    rgb = np.concatenate(rgb_list, axis=0) if rgb_list else np.zeros((0, 3), np.float32)

    if len(xyz) == 0:
        raise SystemExit("[error] no static points produced — check inputs / depth range.")

    if len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(xyz), size=args.max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        print(f"[init] subsampled to max-points={args.max_points:,}")

    labels = np.full((len(xyz),), -1, dtype=np.int32)  # static / unknown class

    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    print(f"[init] final cloud: {len(xyz):,} points")
    print(f"[init] bbox min ({mn[0]:.1f}, {mn[1]:.1f}, {mn[2]:.1f}) "
          f"max ({mx[0]:.1f}, {mx[1]:.1f}, {mx[2]:.1f}) m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xyz=xyz, colors=rgb, labels=labels)
    print(f"[✓] wrote init cloud -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
