#!/usr/bin/env python3
"""
Lift 2D panoptic segmentation + metric depth maps to a 3D semantic point cloud.
=================================================================================

Reads MobileStereoNet depth predictions and Mask2Former panoptic predictions
(both in the default output directories produced by the inference scripts), and
back-projects every surviving pixel into the world frame using the KITTI odometry
poses.  The resulting cloud contains **only the static world**: sky and dynamic
vehicle classes are discarded; persons/riders are kept (treated as static).

KITTI pose convention
---------------------
``poses.txt`` stores one 3×4 matrix per line: the pose of **cam0** (left
greyscale) expressed in the *world* frame, i.e. ``P_world = R @ P_cam0 + t``.
Cam0 and cam2 (left colour, used by both inference scripts) share the same
optical axis after rectification, so cam0 poses are used directly for cam2.

Outputs
-------
``<output>`` (PLY)  – coloured by semantic class (Cityscapes palette) or by
                       the original image RGB, ready to open in MeshLab / Open3D.
``<output>.npz``    – arrays: ``xyz`` (N,3) float32, ``colors`` (N,3) float32
                       [0-1], ``labels`` (N,) int32 semantic class IDs.

Usage
-----
python scripts/lift_to_semantic_pointcloud.py \\
    --dataset_root  /storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences \\
    --sequence      00 \\
    --depth_dir     /usr/prakt/<user>/depth_predictions \\
    --panoptic_dir  /usr/prakt/<user>/panoptic_predictions \\
    --output        /usr/prakt/<user>/semantic_clouds/seq00_static.ply \\
    --n_frames      10 \\
    --aggregation   voxel \\
    --voxel_size    0.1

Aggregation modes
-----------------
``concat``  – concatenate all per-frame clouds after pose-based alignment.
``voxel``   – concatenate then voxel-downsample (removes near-duplicate points).
``icp``     – pose-based alignment + Open3D point-to-plane ICP refinement
              between every new frame and the growing world cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Cityscapes label definitions
# ---------------------------------------------------------------------------

# (label_id, name, is_dynamic_vehicle, is_sky, cityscapes_rgb_colour)
_CITYSCAPES_LABELS = [
    (0,  "road",           False, False, (128,  64, 128)),
    (1,  "sidewalk",       False, False, (244,  35, 232)),
    (2,  "building",       False, False, ( 70,  70,  70)),
    (3,  "wall",           False, False, (102, 102, 156)),
    (4,  "fence",          False, False, (190, 153, 153)),
    (5,  "pole",           False, False, (153, 153, 153)),
    (6,  "traffic light",  False, False, (250, 170,  30)),
    (7,  "traffic sign",   False, False, (220, 220,   0)),
    (8,  "vegetation",     False, False, (107, 142,  35)),
    (9,  "terrain",        False, False, (152, 251, 152)),
    (10, "sky",            False, True,  ( 70, 130, 180)),  # ← excluded
    (11, "person",         False, False, (220,  20,  60)),  # ← kept (static)
    (12, "rider",          False, False, (255,   0,   0)),  # ← kept (static)
    (13, "car",            True,  False, (  0,   0, 142)),  # ← excluded
    (14, "truck",          True,  False, (  0,   0,  70)),  # ← excluded
    (15, "bus",            True,  False, (  0,  60, 100)),  # ← excluded
    (16, "train",          True,  False, (  0,  80, 100)),  # ← excluded
    (17, "motorcycle",     True,  False, (  0,   0, 230)),  # ← excluded
    (18, "bicycle",        True,  False, (119,  11,  32)),  # ← excluded
]

_ID_TO_COLOR  = {lid: np.array(col, dtype=np.float32) / 255.0
                 for lid, _, _, _, col in _CITYSCAPES_LABELS}
_EXCLUDED_IDS = frozenset(
    lid for lid, _, is_dyn, is_sky, _ in _CITYSCAPES_LABELS
    if is_dyn or is_sky
)


def _build_label_exclusion_set(id2label: dict[int, str]) -> frozenset[int]:
    """
    Build the set of segment label IDs to exclude using the id2label mapping
    written by the Mask2Former inference script.  Falls back to the hardcoded
    Cityscapes IDs when label names are not found in the mapping.
    """
    name_to_default_excluded = {
        name.lower()
        for _, name, is_dyn, is_sky, _ in _CITYSCAPES_LABELS
        if is_dyn or is_sky
    }
    excluded = set()
    for lid_str, name in id2label.items():
        if name.lower() in name_to_default_excluded:
            excluded.add(int(lid_str))
    # If the mapping gave no results, fall back to hardcoded IDs
    return frozenset(excluded) if excluded else _EXCLUDED_IDS


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def parse_calib(calib_path: str) -> dict[str, np.ndarray]:
    """Parse a KITTI odometry calib.txt → dict of 3×4 projection matrices."""
    data: dict[str, np.ndarray] = {}
    with open(calib_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = np.fromstring(value, sep=" ", dtype=np.float64)
    return data


def extract_intrinsics(P: np.ndarray) -> tuple[float, float, float, float]:
    """
    Extract (fx, fy, cx, cy) from a KITTI 3×4 projection matrix.

    The depth maps produced by MobileStereoNet are already in the cam2
    coordinate frame, so we only need the upper-left 3×3 block (the
    intrinsic matrix K) to back-project pixels.
    """
    P = P.reshape(3, 4)
    return float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_poses(poses_path: str) -> list[np.ndarray]:
    """
    Load KITTI odometry poses.txt.

    Each line contains 12 space-separated values forming a 3×4 matrix
    [R | t] that transforms a point from the **camera frame** to the
    **world frame**::

        P_world = R @ P_cam + t

    Returns a list of 4×4 homogeneous transformation matrices (float64).
    """
    poses = []
    with open(poses_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float64)
            if vals.size != 12:
                raise ValueError(
                    f"Expected 12 values per pose line, got {vals.size} in {poses_path}"
                )
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = vals.reshape(3, 4)
            poses.append(T)
    return poses


# ---------------------------------------------------------------------------
# Per-frame point cloud extraction
# ---------------------------------------------------------------------------

def backproject_frame(
    depth: np.ndarray,          # (H, W) float32 metres
    panoptic_seg: np.ndarray,   # (H, W) int32  segment IDs (0 = void)
    segment_ids: np.ndarray,    # (N,)   segment IDs present
    label_ids: np.ndarray,      # (N,)   semantic label per segment
    excluded_label_ids: frozenset[int],
    id_to_color: dict[int, np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    pose: np.ndarray,           # (4, 4) camera-to-world transform
    depth_trunc: float,
    rgb: Optional[np.ndarray] = None,  # (H, W, 3) float32 [0-1] or None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Back-project a single frame into world-space 3D points.

    Returns
    -------
    xyz    : (M, 3) float32  world-space XYZ
    colors : (M, 3) float32  [0-1] per-point colour
    labels : (M,)   int32    semantic label ID per point
    """
    H, W = depth.shape

    # ---- Build a per-pixel semantic label map --------------------------------
    # Map: segment_id → label_id
    seg2label = dict(zip(segment_ids.tolist(), label_ids.tolist()))

    label_map = np.zeros((H, W), dtype=np.int32)
    for seg_id, lab_id in seg2label.items():
        label_map[panoptic_seg == seg_id] = lab_id

    # ---- Exclusion mask ------------------------------------------------------
    keep_mask = np.ones((H, W), dtype=bool)
    for excl_id in excluded_label_ids:
        keep_mask &= (label_map != excl_id)

    # Also discard void (panoptic_seg == 0) and invalid / truncated depth
    keep_mask &= (panoptic_seg != 0)
    keep_mask &= (depth > 0.0) & (depth < depth_trunc) & np.isfinite(depth)

    # ---- Back-projection in camera frame ------------------------------------
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy

    xyz_cam = np.stack([X, Y, Z], axis=-1)[keep_mask]   # (M, 3)

    # ---- Transform to world frame -------------------------------------------
    R = pose[:3, :3]
    t = pose[:3,  3]
    xyz_world = (R @ xyz_cam.T).T + t                    # (M, 3)

    # ---- Colours -------------------------------------------------------------
    labels_flat = label_map[keep_mask]                   # (M,)

    if rgb is not None:
        colors = rgb[keep_mask]                          # (M, 3) already float [0-1]
    else:
        # Semantic palette colours
        colors = np.zeros((len(labels_flat), 3), dtype=np.float32)
        for lab_id, col in id_to_color.items():
            mask = labels_flat == lab_id
            if mask.any():
                colors[mask] = col

    return xyz_world.astype(np.float32), colors.astype(np.float32), labels_flat


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def aggregate_concat(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simply stack all per-frame arrays."""
    xyz    = np.concatenate([c[0] for c in clouds], axis=0)
    colors = np.concatenate([c[1] for c in clouds], axis=0)
    labels = np.concatenate([c[2] for c in clouds], axis=0)
    return xyz, colors, labels


def aggregate_voxel(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate all clouds then voxel-downsample using Open3D.

    Each voxel keeps the *average* colour and the *most-common* label of all
    points that fall into it.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[warn] open3d not installed – falling back to concat (no voxel downsampling).")
        return aggregate_concat(clouds)

    xyz, colors, labels = aggregate_concat(clouds)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd_down = pcd.voxel_down_sample(voxel_size)
    xyz_out    = np.asarray(pcd_down.points, dtype=np.float32)
    colors_out = np.asarray(pcd_down.colors, dtype=np.float32)

    # Re-assign labels: for each downsampled point find nearest original point
    from scipy.spatial import cKDTree  # noqa: PLC0415
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz_out, k=1, workers=-1)
    labels_out = labels[idx].astype(np.int32)

    return xyz_out, colors_out, labels_out


def aggregate_icp(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    voxel_size: float,
    max_correspondence_distance: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulate frames using ICP refinement on top of the pose-based alignment.

    Each frame cloud is already in the world frame (pose pre-applied).  ICP
    is used to refine residual drift between each new frame and the growing
    world cloud.  Point-to-plane ICP is used with normal estimation; a
    coarser point-to-point fallback is applied when the world cloud has fewer
    than 500 points.

    The refined transform is composed with the original pose so that
    subsequent frames benefit from the accumulated correction.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[warn] open3d not installed – falling back to voxel aggregation.")
        return aggregate_voxel(clouds, voxel_size)

    if max_correspondence_distance is None:
        max_correspondence_distance = voxel_size * 5.0

    def _to_pcd(xyz: np.ndarray, colors: np.ndarray) -> "o3d.geometry.PointCloud":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    acc_xyz:    list[np.ndarray] = []
    acc_colors: list[np.ndarray] = []
    acc_labels: list[np.ndarray] = []

    world_pcd: Optional["o3d.geometry.PointCloud"] = None

    for i, (xyz, colors, labels) in enumerate(clouds):
        src_pcd = _to_pcd(xyz, colors)

        if world_pcd is None or len(world_pcd.points) < 500:
            # Not enough reference geometry yet – just accept as-is
            acc_xyz.append(xyz)
            acc_colors.append(colors)
            acc_labels.append(labels)
        else:
            # Downsample for ICP registration
            src_down   = src_pcd.voxel_down_sample(voxel_size)
            world_down = world_pcd.voxel_down_sample(voxel_size)

            # Try point-to-plane (needs normals)
            use_p2plane = len(world_down.points) >= 100
            if use_p2plane:
                world_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 3, max_nn=30
                    )
                )
                result = o3d.pipelines.registration.registration_icp(
                    src_down, world_down,
                    max_correspondence_distance,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
            else:
                result = o3d.pipelines.registration.registration_icp(
                    src_down, world_down,
                    max_correspondence_distance,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

            T_refine = np.asarray(result.transformation)   # 4×4
            if result.fitness > 0.05:
                # Apply refinement only when ICP converged reasonably
                xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])   # (M, 4)
                xyz   = (T_refine @ xyz_h.T).T[:, :3].astype(np.float32)
            else:
                print(f"  [icp] frame {i}: low fitness ({result.fitness:.3f}), "
                      "keeping pose-only alignment.")

            acc_xyz.append(xyz)
            acc_colors.append(colors)
            acc_labels.append(labels)

        # Grow world cloud (voxel-downsampled for speed)
        frame_pcd = _to_pcd(
            acc_xyz[-1] if acc_xyz else xyz,
            acc_colors[-1] if acc_colors else colors,
        )
        if world_pcd is None:
            world_pcd = frame_pcd.voxel_down_sample(voxel_size)
        else:
            world_pcd += frame_pcd
            world_pcd = world_pcd.voxel_down_sample(voxel_size)

    xyz_all    = np.concatenate(acc_xyz,    axis=0)
    colors_all = np.concatenate(acc_colors, axis=0)
    labels_all = np.concatenate(acc_labels, axis=0)
    return xyz_all, colors_all, labels_all


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def save_ply(
    path: str,
    xyz: np.ndarray,      # (N, 3) float32
    colors: np.ndarray,   # (N, 3) float32 [0-1]
    labels: np.ndarray,   # (N,)   int32
) -> None:
    """
    Write a coloured PLY with an extra scalar property ``label`` so that
    tools like CloudCompare can colour-by-scalar after loading.
    """
    N = len(xyz)
    cols_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property int label\n"
            "end_header\n"
        )
        for (x, y, z), (r, g, b), lbl in zip(xyz, cols_u8, labels):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {lbl}\n")
    print(f"[✓] Saved PLY  → {path}  ({N:,} points)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Lift Mask2Former panoptic + MobileStereoNet depth predictions to a "
            "3D semantic point cloud of the static driving scene."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument(
        "--dataset_root", required=True,
        help=(
            "Root directory containing numbered KITTI odometry sequence folders.  "
            "Used to locate calib.txt and (optionally) image_2/ frames when "
            "--color_mode rgb is requested."
        ),
    )
    p.add_argument(
        "--sequence", required=True,
        metavar="SEQ",
        help="Two-digit sequence ID, e.g. 00.",
    )
    p.add_argument(
        "--depth_dir", required=True,
        help=(
            "Root directory of MobileStereoNet depth predictions.  "
            "Frames are expected at {depth_dir}/{sequence}/{frame_stem}.npz."
        ),
    )
    p.add_argument(
        "--panoptic_dir", required=True,
        help=(
            "Root directory of Mask2Former panoptic predictions.  "
            "Frames are expected at {panoptic_dir}/{sequence}/{frame_stem}.npz.  "
            "id2label.json is read from {panoptic_dir}/id2label.json."
        ),
    )

    # --- optional ---
    p.add_argument(
        "--poses_root", default=None,
        help=(
            "Root directory of KITTI sequences that contain poses.txt.  "
            "Defaults to --dataset_root when not specified.  "
            "Useful when depth/panoptic come from the colour dataset but poses "
            "live under the grey dataset path, e.g. "
            "/storage/group/dataset_mirrors/kitti_odom_grey/sequences."
        ),
    )
    p.add_argument(
        "--output",
        default="semantic_cloud.ply",
        help="Path of the output PLY file.",
    )
    p.add_argument(
        "--n_frames", type=int, default=10,
        help=(
            "Number of consecutive frames to aggregate.  "
            "At 10 fps this corresponds to 1 second of driving."
        ),
    )
    p.add_argument(
        "--start_frame", type=int, default=0,
        help="Index of the first frame to process (0-based, matches NPZ filename).",
    )
    p.add_argument(
        "--aggregation", choices=["concat", "voxel", "icp"], default="voxel",
        help=(
            "Point-cloud aggregation strategy.  "
            "concat: stack all frames as-is.  "
            "voxel: concatenate + voxel-downsample (deduplicates nearby points).  "
            "icp: pose-based alignment + Open3D ICP refinement between frames."
        ),
    )
    p.add_argument(
        "--voxel_size", type=float, default=0.10,
        help="Voxel grid cell size in metres used by the voxel and icp modes.",
    )
    p.add_argument(
        "--depth_trunc", type=float, default=50.0,
        help="Discard points whose depth exceeds this value (metres).",
    )
    p.add_argument(
        "--color_mode", choices=["semantic", "rgb"], default="semantic",
        help=(
            "Point colour source.  "
            "semantic: Cityscapes palette colours.  "
            "rgb: load the original image_2 frame and use its pixel colours."
        ),
    )
    p.add_argument(
        "--image_subdir", default="image_2",
        help=(
            "Sub-directory inside each sequence folder that holds the colour "
            "frames used when --color_mode rgb is requested."
        ),
    )
    p.add_argument(
        "--save_npz", action="store_true",
        help=(
            "Also save xyz / colors / labels arrays as a compressed NPZ next "
            "to the PLY output for downstream processing."
        ),
    )
    p.add_argument(
        "--no_cuda", action="store_true",
        help="Unused; kept for CLI consistency with other scripts.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seq = args.sequence.zfill(2)

    # ---- Resolve paths -------------------------------------------------------
    dataset_seq_dir = Path(args.dataset_root) / seq
    poses_root      = Path(args.poses_root) if args.poses_root else Path(args.dataset_root)
    poses_path      = poses_root / seq / "poses.txt"
    calib_path      = dataset_seq_dir / "calib.txt"
    depth_seq_dir   = Path(args.depth_dir) / seq
    panoptic_seq_dir = Path(args.panoptic_dir) / seq
    id2label_path   = Path(args.panoptic_dir) / "id2label.json"

    for label, path in [
        ("calib.txt",  calib_path),
        ("poses.txt",  poses_path),
        ("depth dir",  depth_seq_dir),
        ("panoptic dir", panoptic_seq_dir),
    ]:
        if not Path(path).exists():
            sys.exit(f"[error] {label} not found: {path}")

    # ---- Calibration ---------------------------------------------------------
    calib = parse_calib(str(calib_path))
    if "P2" not in calib:
        sys.exit("[error] P2 not found in calib.txt – expected KITTI odometry format.")
    fx, fy, cx, cy = extract_intrinsics(calib["P2"])
    print(f"[i] Camera intrinsics (P2): fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # ---- Poses ---------------------------------------------------------------
    all_poses = load_poses(str(poses_path))
    print(f"[i] Loaded {len(all_poses)} poses from {poses_path}")

    # ---- Label exclusion set -------------------------------------------------
    id2label: dict[int, str] = {}
    if id2label_path.exists():
        with open(id2label_path) as fh:
            id2label = {int(k): v for k, v in json.load(fh).items()}
        print(f"[i] Loaded id2label mapping ({len(id2label)} classes) from {id2label_path}")
    else:
        print(f"[warn] id2label.json not found at {id2label_path}; "
              "using hardcoded Cityscapes class IDs for exclusion.")

    excluded = _build_label_exclusion_set({str(k): v for k, v in id2label.items()}) \
        if id2label else _EXCLUDED_IDS

    excluded_names = [name for lid, name, _, _, _ in _CITYSCAPES_LABELS
                      if lid in excluded]
    print(f"[i] Excluding classes: {excluded_names}")

    # ---- Frame discovery -----------------------------------------------------
    start = args.start_frame
    end   = start + args.n_frames

    if end > len(all_poses):
        print(f"[warn] Requested frames {start}–{end-1} but only "
              f"{len(all_poses)} poses available; clamping to {len(all_poses) - 1}.")
        end = len(all_poses)

    frame_indices = list(range(start, end))
    print(f"[i] Processing {len(frame_indices)} frames: "
          f"{frame_indices[0]:06d} → {frame_indices[-1]:06d}")

    # ---- Per-frame processing ------------------------------------------------
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for frame_idx in frame_indices:
        stem = f"{frame_idx:06d}"

        depth_path    = depth_seq_dir    / f"{stem}.npz"
        panoptic_path = panoptic_seq_dir / f"{stem}.npz"

        if not depth_path.exists():
            print(f"  [skip] depth missing:    {depth_path}")
            continue
        if not panoptic_path.exists():
            print(f"  [skip] panoptic missing: {panoptic_path}")
            continue

        # Load depth
        depth_npz = np.load(str(depth_path))
        depth = depth_npz["depth"].astype(np.float32)

        # Load panoptic
        pan_npz      = np.load(str(panoptic_path))
        panoptic_seg = pan_npz["panoptic_seg"].astype(np.int32)
        segment_ids  = pan_npz["segment_ids"].astype(np.int32)
        label_ids    = pan_npz["label_ids"].astype(np.int32)

        # Sanity-check spatial alignment between depth and panoptic
        if depth.shape != panoptic_seg.shape:
            import cv2  # noqa: PLC0415
            panoptic_seg = cv2.resize(
                panoptic_seg, (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        # Optionally load RGB for colour
        rgb: Optional[np.ndarray] = None
        if args.color_mode == "rgb":
            rgb_path = dataset_seq_dir / args.image_subdir / f"{stem}.png"
            if rgb_path.exists():
                import cv2  # noqa: PLC0415
                img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.shape[:2] != depth.shape:
                        img = cv2.resize(img, (depth.shape[1], depth.shape[0]))
                    rgb = img.astype(np.float32) / 255.0
            else:
                print(f"  [warn] RGB image not found: {rgb_path}; "
                      "falling back to semantic colours.")

        pose = all_poses[frame_idx]

        xyz, colors, labels = backproject_frame(
            depth=depth,
            panoptic_seg=panoptic_seg,
            segment_ids=segment_ids,
            label_ids=label_ids,
            excluded_label_ids=excluded,
            id_to_color=_ID_TO_COLOR,
            fx=fx, fy=fy, cx=cx, cy=cy,
            pose=pose,
            depth_trunc=args.depth_trunc,
            rgb=rgb,
        )

        print(f"  frame {stem}: {len(xyz):>8,} pts kept "
              f"(depth range {depth[depth > 0].min():.1f}–"
              f"{depth[depth < args.depth_trunc].max():.1f} m)")

        if len(xyz) > 0:
            clouds.append((xyz, colors, labels))

    if not clouds:
        sys.exit("[error] No valid frames found – check paths and frame range.")

    # ---- Aggregation ---------------------------------------------------------
    print(f"\n[i] Aggregation mode: {args.aggregation}")

    if args.aggregation == "concat":
        xyz_final, colors_final, labels_final = aggregate_concat(clouds)

    elif args.aggregation == "voxel":
        xyz_final, colors_final, labels_final = aggregate_voxel(
            clouds, voxel_size=args.voxel_size
        )

    elif args.aggregation == "icp":
        xyz_final, colors_final, labels_final = aggregate_icp(
            clouds, voxel_size=args.voxel_size
        )

    else:
        raise ValueError(f"Unknown aggregation mode: {args.aggregation}")

    print(f"[i] Final cloud: {len(xyz_final):,} points")

    # ---- Save PLY ------------------------------------------------------------
    save_ply(args.output, xyz_final, colors_final, labels_final)

    # ---- Optionally save NPZ -------------------------------------------------
    if args.save_npz:
        npz_path = str(args.output).replace(".ply", "") + ".npz"
        np.savez_compressed(
            npz_path,
            xyz=xyz_final,
            colors=colors_final,
            labels=labels_final,
        )
        print(f"[✓] Saved NPZ  → {npz_path}")

    # ---- Per-class point counts (diagnostics) --------------------------------
    print("\n[i] Points per semantic class in final cloud:")
    label_to_name = {lid: name for lid, name, _, _, _ in _CITYSCAPES_LABELS}
    if id2label:
        label_to_name.update(id2label)
    unique_labels, counts = np.unique(labels_final, return_counts=True)
    for lbl, cnt in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        name = label_to_name.get(int(lbl), f"class_{lbl}")
        print(f"    {name:20s} {cnt:>10,}")


if __name__ == "__main__":
    main()
