#!/usr/bin/env python3
"""
Extract drivable-road geometry from panoptic segmentation + metric depth, for
placing dynamic-object Gaussian splats and for path planning in the lifted
3D (eventually 4D-GS) world.
============================================================================

This is a sibling script to ``lift_to_semantic_pointcloud.py``: it reads the
exact same MobileStereoNet depth predictions, Mask2Former panoptic
predictions, KITTI calibration and poses, but instead of producing a full
semantic scene cloud it isolates the "road" class and turns it into several
representations that are each useful for a different downstream need:

  1. Road MESH (.ply / .obj)
     A 2.5D triangulated surface (Delaunay in the ground plane, height
     looked up per vertex) of the drivable area.  Use this for:
       - sampling an exact placement height + surface normal for a spawned
         asset at any (x, z) query point (so a car doesn't float or clip
         into a slope/camber),
       - visual fidelity if you want to render the road itself.

  2. Oriented road RECTANGLES (.json / .npz) — "plane parameters"
     The road is chopped into fixed-length segments along the driving
     trajectory; each segment gets a least-squares-fit local plane
     (centroid + normal) and an oriented bounding rectangle in that plane
     (forward axis = local driving heading, right axis = lateral, plus
     half-length / half-width extents).  Use this for:
       - cheap, dependency-free placement / collision queries
         (point-in-OBB test) without touching the mesh or point cloud,
       - coarse "is there road here" reasoning at the segment level,
       - seeding lane-width / road-width statistics.

  3. 2D occupancy grid + drivable-area polygon (.npz / .png)
     A top-down (bird's-eye) binary grid of drivable vs. non-drivable cells,
     plus the polygon boundary of the drivable region.  Use this for:
       - fast O(1) traversability lookups during path planning / collision
         checking (much cheaper than mesh raycasting or point-cloud KNN),
       - a debug visualization of exactly what was classified as "road".

  4. Centerline GRAPH (.json / .npz) — for path planning
     A graph of nodes (3D position on the fitted road surface + heading)
     connected sequentially along the trajectory, with branch points left
     as TODO hooks (KITTI odometry sequences are single-carriageway, so a
     simple polyline graph is sufficient for one sequence; the format
     supports adding extra edges later e.g. when merging multiple
     sequences or adding lane-change edges).  Use this directly as the
     roadmap for a planner (A*, RRT, etc.) over (node, heading, speed).

All outputs share the SAME coordinate frame as ``lift_to_semantic_pointcloud.py``
i.e. KITTI's pose convention: world frame = cam0 pose of frame 0, camera/world
axes are +X right, +Y down, +Z forward.  "Up" is therefore -Y, not +Y or +Z.

Usage
-----
python scripts/extract_road_layout.py \\
    --dataset_root  /storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences \\
    --sequence      00 \\
    --depth_dir     /usr/prakt/<user>/depth_predictions \\
    --panoptic_dir  /usr/prakt/<user>/panoptic_predictions \\
    --output_dir    /usr/prakt/<user>/road_layout/seq00 \\
    --n_frames      150 \\
    --segment_length 5.0 \\
    --grid_resolution 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Reuse the exact same loader utilities as the semantic-lifting script so the
# two scripts can never silently disagree about calibration / pose / panoptic
# parsing.  If your real utils.py lives next to lift_to_semantic_pointcloud.py
# this import just works; the except-block below is only a fallback so this
# file is still runnable/inspectable standalone.
# ---------------------------------------------------------------------------
try:
    from utils import (  # noqa: E402
        CITYSCAPES_LABELS,
        parse_kitti_calib,
        extract_intrinsics,
        load_poses,
        load_panoptic,
        load_id2label,
    )
except ImportError:
    print("[warn] Could not import project utils.py - using minimal local "
          "fallbacks. Place this script next to lift_to_semantic_pointcloud.py "
          "to use your real KITTI/panoptic loaders.", file=sys.stderr)

    # Minimal Cityscapes label tuples: (id, name, color_or_None, is_sky, is_dynamic)
    CITYSCAPES_LABELS = [
        (0,  "road",          (128, 64, 128), False, False),
        (1,  "sidewalk",      (244, 35, 232), False, False),
        (10, "sky",           (70, 130, 180), True,  False),
    ]

    def parse_kitti_calib(path: str) -> dict[str, np.ndarray]:
        calib = {}
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, vals = line.split(":", 1)
                nums = np.array([float(v) for v in vals.strip().split()])
                if nums.size == 12:
                    calib[key.strip()] = nums.reshape(3, 4)
        return calib

    def extract_intrinsics(P: np.ndarray) -> tuple[float, float, float, float]:
        return float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])

    def load_poses(path: str) -> list[np.ndarray]:
        poses = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = np.array([float(v) for v in line.split()])
                T = np.eye(4)
                T[:3, :4] = vals.reshape(3, 4)
                poses.append(T)
        return poses

    def load_panoptic(path: str) -> dict[str, np.ndarray]:
        npz = np.load(path)
        return {
            "panoptic_seg": npz["panoptic_seg"],
            "segment_ids": npz["segment_ids"],
            "label_ids": npz["label_ids"],
        }

    def load_id2label(path: str) -> dict[int, str]:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

# "Road" label id(s) in the Cityscapes taxonomy.  Only "road" is treated as
# drivable; "sidewalk", "terrain", "parking" are deliberately excluded since
# vehicles should not be planned/placed there.
ROAD_LABEL_NAMES = frozenset({"road"})
ROAD_IDS: frozenset[int] = frozenset(
    lid for lid, name, *_ in CITYSCAPES_LABELS if name in ROAD_LABEL_NAMES
)


# ---------------------------------------------------------------------------
# Per-frame road point extraction
# ---------------------------------------------------------------------------

def backproject_road_points(
    depth: np.ndarray,
    panoptic_seg: np.ndarray,
    segment_ids: np.ndarray,
    label_ids: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    pose: np.ndarray,
    depth_trunc: float,
    road_ids: frozenset[int] = ROAD_IDS,
) -> np.ndarray:
    """Back-project only the road-labelled pixels of a single frame to world XYZ."""
    H, W = depth.shape
    seg2label = dict(zip(segment_ids.tolist(), label_ids.tolist()))

    label_map = np.zeros((H, W), dtype=np.int32)
    for seg_id, lab_id in seg2label.items():
        label_map[panoptic_seg == seg_id] = lab_id

    road_mask = np.zeros((H, W), dtype=bool)
    for rid in road_ids:
        road_mask |= (label_map == rid)

    valid = (
        road_mask
        & (panoptic_seg != 0)
        & (depth > 0.0) & (depth < depth_trunc) & np.isfinite(depth)
    )
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy
    xyz_cam = np.stack([X, Y, Z], axis=-1)[valid]

    R = pose[:3, :3]
    t = pose[:3, 3]
    xyz_world = (R @ xyz_cam.T).T + t
    return xyz_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory helpers (shared across mesh / rectangles / centerline graph)
# ---------------------------------------------------------------------------

UP_VEC = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # KITTI convention: -Y is up


def trajectory_arc_length(traj_xyz: np.ndarray) -> np.ndarray:
    """Cumulative arc length (metres) along a (N,3) polyline."""
    diffs = np.diff(traj_xyz, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seglens)])


def resample_polyline(points: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (N,D) polyline at uniform arc-length spacing (metres)."""
    arc = trajectory_arc_length(points)
    total = arc[-1]
    n = max(2, int(total // spacing) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, points.shape[1]), dtype=points.dtype)
    for d in range(points.shape[1]):
        out[:, d] = np.interp(targets, arc, points[:, d])
    return out, targets


def heading_vectors(nodes_xz: np.ndarray) -> np.ndarray:
    """Unit forward-heading vectors (N,2) in the XZ ground plane via finite differences."""
    heading = np.zeros_like(nodes_xz)
    heading[:-1] = nodes_xz[1:] - nodes_xz[:-1]
    heading[-1] = heading[-2] if len(heading) > 1 else np.array([0.0, 1.0])
    norms = np.linalg.norm(heading, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return heading / norms


# ---------------------------------------------------------------------------
# 1. Road mesh (2.5D Delaunay)
# ---------------------------------------------------------------------------

def build_road_mesh(
    road_points: np.ndarray,
    max_edge_length: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate road points in the ground plane (XZ) and lift back to 3D.

    Long edges (which bridge real gaps - occlusions, missing frames, the far
    side of an intersection - rather than sampling the same contiguous patch
    of road) are discarded so the mesh doesn't paper over holes with bogus
    flat triangles.

    Returns
    -------
    vertices  : (V, 3) float32
    triangles : (T, 3) int32  vertex indices
    """
    from scipy.spatial import Delaunay

    if len(road_points) < 3:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    xz = road_points[:, [0, 2]].astype(np.float64)
    tri = Delaunay(xz)
    simplices = tri.simplices

    p0 = xz[simplices[:, 0]]
    p1 = xz[simplices[:, 1]]
    p2 = xz[simplices[:, 2]]
    e0 = np.linalg.norm(p0 - p1, axis=1)
    e1 = np.linalg.norm(p1 - p2, axis=1)
    e2 = np.linalg.norm(p2 - p0, axis=1)
    max_edge = np.maximum(np.maximum(e0, e1), e2)

    keep = max_edge < max_edge_length
    triangles = simplices[keep].astype(np.int32)

    return road_points.astype(np.float32), triangles


def save_mesh_ply(path: str, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Minimal ASCII PLY writer for a vertex+face mesh (no external deps)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def save_mesh_obj(path: str, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Minimal OBJ writer (1-indexed faces) - convenient for Blender/Unity/Unreal import."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


# ---------------------------------------------------------------------------
# 2. Oriented road rectangles ("plane parameters") along the trajectory
# ---------------------------------------------------------------------------

def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares plane fit via SVD.  Returns (centroid, normal); normal is
    oriented to point "up" (i.e. have negative Y, per the -Y-is-up convention)
    so downstream consumers can rely on a consistent sign.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Smallest singular vector of the centered points = plane normal.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal @ UP_VEC < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    return centroid.astype(np.float32), normal.astype(np.float32)


def build_road_rectangles(
    road_points: np.ndarray,
    traj_xyz: np.ndarray,
    segment_length: float = 5.0,
    min_points_per_segment: int = 50,
    max_lateral_assignment_dist: Optional[float] = None,
) -> list[dict]:
    """
    Chop the trajectory into fixed-length arc-length segments; for each
    segment, fit a local plane to the road points whose arc-length-projected
    position falls in that segment, and compute an oriented bounding
    rectangle in that plane using the segment's driving heading as the
    forward axis.

    Parameters
    ----------
    max_lateral_assignment_dist
        Road points are assigned to a trajectory arc-length bin via nearest
        XZ neighbour against the trajectory polyline.  The trajectory only
        spans the aggregated frames' camera positions, but road visible in
        the LAST frame can extend well beyond the trajectory's own endpoint
        (anything up to depth_trunc ahead of the final camera).  Those points
        have no real trajectory sample near them, yet a naive nearest-
        neighbour query still happily snaps them to the closest (= last)
        trajectory sample and dumps them all into the final bin, wildly
        over-populating and skewing it.  Points whose nearest trajectory
        sample is farther than this distance (metres) are therefore dropped
        from rectangle fitting entirely.  Defaults to ``2 * segment_length``,
        which is generous enough to keep legitimate off-centerline points
        (e.g. a wide intersection) while rejecting points with no nearby
        trajectory coverage at all.

    Returns a list of dicts (one per segment), each containing:
        centroid        (3,) world-space centre of the rectangle
        normal          (3,) plane normal ("up"), -Y-aligned
        forward         (3,) unit vector, local driving direction
        right            (3,) unit vector, lateral direction (forward x normal)
        half_length     float, half-extent along `forward`
        half_width      float, half-extent along `right`
        arc_start/end   float, arc-length range along the trajectory (m)
        n_points        int, number of road points supporting this fit
        corners         (4,3) world-space rectangle corners (for quick viz / OBB tests)
    """
    arc = trajectory_arc_length(traj_xyz)
    total = arc[-1]
    if total <= 0:
        return []

    if max_lateral_assignment_dist is None:
        max_lateral_assignment_dist = 2.0 * segment_length

    n_segs = max(1, int(np.ceil(total / segment_length)))
    bin_edges = np.linspace(0.0, total, n_segs + 1)

    # Assign each road point to the nearest trajectory sample's arc-length
    # position, via nearest-neighbour in XZ against the trajectory polyline.
    # Points too far from ANY trajectory sample (e.g. road visible far ahead
    # of the last camera, beyond where the trajectory polyline actually
    # extends) are excluded rather than being force-assigned to whichever
    # bin happens to be nearest - see max_lateral_assignment_dist above.
    from scipy.spatial import cKDTree
    traj_xz = traj_xyz[:, [0, 2]]
    tree = cKDTree(traj_xz)
    nn_dist, nearest_traj_idx = tree.query(road_points[:, [0, 2]], k=1, workers=-1)
    in_range = nn_dist < max_lateral_assignment_dist
    n_dropped = int((~in_range).sum())
    if n_dropped > 0:
        print(f"    [rectangles] dropping {n_dropped:,} road points with no "
              f"trajectory sample within {max_lateral_assignment_dist:.1f} m "
              "(likely road visible beyond the trajectory's own extent)")

    point_arc = arc[nearest_traj_idx]
    point_bin = np.full(len(road_points), -1, dtype=np.int64)
    point_bin[in_range] = np.clip(
        np.digitize(point_arc[in_range], bin_edges) - 1, 0, n_segs - 1
    )

    rectangles: list[dict] = []
    for seg_i in range(n_segs):
        seg_pts = road_points[point_bin == seg_i]
        if len(seg_pts) < min_points_per_segment:
            continue

        a0, a1 = bin_edges[seg_i], bin_edges[seg_i + 1]
        # Trajectory samples that fall in this arc-length range define the
        # segment's nominal driving heading (more robust than deriving
        # heading from the noisy road points themselves).
        traj_in_seg = (arc >= a0) & (arc <= a1)
        if traj_in_seg.sum() < 2:
            # Segment shorter than trajectory sampling - widen the window.
            idx = np.searchsorted(arc, (a0 + a1) / 2.0)
            idx0, idx1 = max(0, idx - 1), min(len(traj_xyz) - 1, idx + 1)
            traj_chunk = traj_xyz[idx0:idx1 + 1]
        else:
            traj_chunk = traj_xyz[traj_in_seg]

        fwd_xz = traj_chunk[-1, [0, 2]] - traj_chunk[0, [0, 2]]
        norm = np.linalg.norm(fwd_xz)
        if norm < 1e-6:
            continue
        fwd_xz /= norm
        forward = np.array([fwd_xz[0], 0.0, fwd_xz[1]])

        centroid, normal = fit_plane(seg_pts)

        right = np.cross(forward, normal)
        right /= (np.linalg.norm(right) + 1e-12)
        # Re-orthogonalise forward against the fitted normal (road may be
        # cambered/sloped, so raw trajectory-derived forward isn't exactly
        # in-plane).
        forward_inplane = np.cross(normal, right)
        forward_inplane /= (np.linalg.norm(forward_inplane) + 1e-12)

        rel = seg_pts - centroid
        proj_fwd = rel @ forward_inplane
        proj_right = rel @ right

        half_length = float((proj_fwd.max() - proj_fwd.min()) / 2.0)
        half_width = float((proj_right.max() - proj_right.min()) / 2.0)
        # Re-centre centroid on the projected extents' midpoint (the plane
        # centroid is the mean position, which may not be the bounding-box
        # centre if points are unevenly distributed across the segment).
        mid_fwd = (proj_fwd.max() + proj_fwd.min()) / 2.0
        mid_right = (proj_right.max() + proj_right.min()) / 2.0
        centroid = centroid + mid_fwd * forward_inplane + mid_right * right

        corners = np.stack([
            centroid + half_length * forward_inplane + half_width * right,
            centroid + half_length * forward_inplane - half_width * right,
            centroid - half_length * forward_inplane - half_width * right,
            centroid - half_length * forward_inplane + half_width * right,
        ], axis=0)

        rectangles.append({
            "centroid": centroid.astype(np.float32),
            "normal": normal.astype(np.float32),
            "forward": forward_inplane.astype(np.float32),
            "right": right.astype(np.float32),
            "half_length": half_length,
            "half_width": half_width,
            "arc_start": float(a0),
            "arc_end": float(a1),
            "n_points": int(len(seg_pts)),
            "corners": corners.astype(np.float32),
        })

    return rectangles


def save_rectangles_json(path: str, rectangles: list[dict]) -> None:
    serializable = []
    for r in rectangles:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        serializable.append(d)
    with open(path, "w") as f:
        json.dump({"rectangles": serializable}, f, indent=2)


def save_rectangles_npz(path: str, rectangles: list[dict]) -> None:
    if not rectangles:
        np.savez_compressed(path)
        return
    np.savez_compressed(
        path,
        centroid=np.stack([r["centroid"] for r in rectangles]),
        normal=np.stack([r["normal"] for r in rectangles]),
        forward=np.stack([r["forward"] for r in rectangles]),
        right=np.stack([r["right"] for r in rectangles]),
        half_length=np.array([r["half_length"] for r in rectangles], dtype=np.float32),
        half_width=np.array([r["half_width"] for r in rectangles], dtype=np.float32),
        arc_start=np.array([r["arc_start"] for r in rectangles], dtype=np.float32),
        arc_end=np.array([r["arc_end"] for r in rectangles], dtype=np.float32),
        n_points=np.array([r["n_points"] for r in rectangles], dtype=np.int32),
        corners=np.stack([r["corners"] for r in rectangles]),
    )


# ---------------------------------------------------------------------------
# 3. 2D occupancy grid + drivable-area polygon
# ---------------------------------------------------------------------------

def build_occupancy_grid(
    road_points: np.ndarray,
    resolution: float = 0.2,
    dilate_iters: int = 1,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Rasterise road points into a binary top-down occupancy grid in the XZ
    (ground) plane.

    Returns
    -------
    grid       : (rows, cols) bool array, True = drivable
    origin_xz  : (x_min, z_min) world coordinates of grid cell (0, 0)
    resolution : metres per cell (echoed back for convenience)
    """
    if len(road_points) == 0:
        return np.zeros((1, 1), dtype=bool), (0.0, 0.0), resolution

    xz = road_points[:, [0, 2]]
    x_min, z_min = xz.min(axis=0) - resolution
    x_max, z_max = xz.max(axis=0) + resolution

    cols = max(1, int(np.ceil((x_max - x_min) / resolution)))
    rows = max(1, int(np.ceil((z_max - z_min) / resolution)))

    col_idx = np.clip(((xz[:, 0] - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((xz[:, 1] - z_min) / resolution).astype(int), 0, rows - 1)

    grid = np.zeros((rows, cols), dtype=bool)
    grid[row_idx, col_idx] = True

    if dilate_iters > 0:
        try:
            from scipy.ndimage import binary_dilation
            grid = binary_dilation(grid, iterations=dilate_iters)
        except ImportError:
            pass  # cosmetic only - skip if scipy.ndimage unavailable

    return grid, (float(x_min), float(z_min)), resolution


def grid_to_polygon(grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the boundary of the largest connected drivable region as a
    polygon in grid-cell coordinates (row, col), via marching-squares-style
    contour finding.  Returns None if scipy/skimage contour tooling isn't
    available - the occupancy grid itself is still saved either way, this
    polygon is a convenience extra for planners that prefer a boundary
    representation over a raster.
    """
    try:
        from skimage import measure
    except ImportError:
        return None

    contours = measure.find_contours(grid.astype(float), level=0.5)
    if not contours:
        return None
    # Largest contour by point count as a simple proxy for "main road area".
    largest = max(contours, key=len)
    return largest.astype(np.float32)  # (N, 2) in (row, col)


def save_occupancy_grid(
    path_npz: str,
    path_png: Optional[str],
    grid: np.ndarray,
    origin_xz: tuple[float, float],
    resolution: float,
    polygon_rc: Optional[np.ndarray],
) -> None:
    np.savez_compressed(
        path_npz,
        grid=grid,
        origin_x=origin_xz[0],
        origin_z=origin_xz[1],
        resolution=resolution,
        polygon_rc=polygon_rc if polygon_rc is not None else np.zeros((0, 2), dtype=np.float32),
    )
    if path_png is not None:
        try:
            import cv2
            img = (grid.astype(np.uint8) * 255)
            if polygon_rc is not None and len(polygon_rc) > 1:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                pts = polygon_rc[:, [1, 0]].astype(np.int32)  # (col, row) for cv2
                cv2.polylines(img_color, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
                cv2.imwrite(path_png, img_color)
            else:
                cv2.imwrite(path_png, img)
        except ImportError:
            print("[warn] cv2 not available - skipping PNG visualisation of occupancy grid.")


# ---------------------------------------------------------------------------
# 4. Centerline graph for path planning
# ---------------------------------------------------------------------------

def build_centerline_graph(
    traj_xyz: np.ndarray,
    road_points: np.ndarray,
    node_spacing: float = 2.0,
    height_snap_k: int = 8,
) -> dict:
    """
    Resample the driving trajectory at uniform spacing and snap each node's
    height to the locally fitted road surface (rather than the raw camera
    trajectory height, which sits ~1.6 m above the true road due to camera
    mount height).

    Returns a dict with:
        nodes    (N,3) world-space positions, ON the road surface
        headings (N,2) unit forward-heading vectors in the XZ plane
        yaw      (N,)  heading angle (radians) from +Z axis
        edges    (E,2) int pairs of node indices (sequential polyline;
                  extend this list yourself if you merge multiple sequences
                  or add lane-change / intersection edges later)
        arc_length (N,) cumulative distance along the path (m) - useful for
                  speed-profile planning
    """
    nodes_raw, arc_len = resample_polyline(traj_xyz, node_spacing)
    headings = heading_vectors(nodes_raw[:, [0, 2]])
    yaw = np.arctan2(headings[:, 0], headings[:, 1])

    nodes = nodes_raw.copy()
    if len(road_points) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(road_points[:, [0, 2]])
        k = min(height_snap_k, len(road_points))
        _, idx = tree.query(nodes[:, [0, 2]], k=k, workers=-1)
        idx = idx.reshape(len(nodes), k)
        snapped_y = road_points[idx][:, :, 1].mean(axis=1)
        nodes[:, 1] = snapped_y
    else:
        print("[warn] No road points available for height snapping - "
              "centerline graph will use raw (camera-height) trajectory Y.")

    edges = np.stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))], axis=1)

    return {
        "nodes": nodes.astype(np.float32),
        "headings": headings.astype(np.float32),
        "yaw": yaw.astype(np.float32),
        "edges": edges.astype(np.int32),
        "arc_length": arc_len.astype(np.float32),
    }


def save_centerline_json(path: str, graph: dict) -> None:
    serializable = {
        "nodes": graph["nodes"].tolist(),
        "headings": graph["headings"].tolist(),
        "yaw": graph["yaw"].tolist(),
        "edges": graph["edges"].tolist(),
        "arc_length": graph["arc_length"].tolist(),
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_centerline_npz(path: str, graph: dict) -> None:
    np.savez_compressed(path, **graph)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract drivable-road geometry (mesh, oriented rectangles, "
            "occupancy grid, centerline graph) for placing dynamic-object "
            "Gaussian splats and path planning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset_root", required=True,
                    help="Root directory containing numbered KITTI odometry sequence folders.")
    p.add_argument("--sequence", required=True, metavar="SEQ",
                    help="Two-digit sequence ID, e.g. 00.")
    p.add_argument("--depth_dir", required=True,
                    help="Root directory of MobileStereoNet depth predictions.")
    p.add_argument("--panoptic_dir", required=True,
                    help="Root directory of Mask2Former panoptic predictions.")
    p.add_argument("--poses_root", default=None,
                    help="Root directory containing poses.txt; defaults to --dataset_root.")
    p.add_argument("--output_dir", required=True,
                    help="Directory to write all road-layout outputs into.")

    p.add_argument("--n_frames", type=int, default=150,
                    help="Number of consecutive frames to aggregate.")
    p.add_argument("--start_frame", type=int, default=0,
                    help="Index of the first frame to process.")
    p.add_argument("--depth_trunc", type=float, default=50.0,
                    help="Discard points whose depth exceeds this value (metres).")

    p.add_argument("--voxel_size", type=float, default=0.10,
                    help="Voxel size (metres) for downsampling raw road points before "
                         "meshing/fitting; requires open3d, falls back to no "
                         "downsampling if unavailable.")

    p.add_argument("--mesh_max_edge", type=float, default=1.5,
                    help="Discard mesh triangles whose longest edge exceeds this "
                         "(metres) - prevents bridging real gaps in the road.")
    p.add_argument("--mesh_format", choices=["ply", "obj", "both"], default="both",
                    help="Output format(s) for the road mesh.")

    p.add_argument("--segment_length", type=float, default=5.0,
                    help="Arc-length (metres) of each oriented road rectangle segment.")
    p.add_argument("--min_points_per_segment", type=int, default=50,
                    help="Minimum road points required to fit a rectangle for a segment.")
    p.add_argument("--max_lateral_assignment_dist", type=float, default=None,
                    help="Max distance (metres) from a road point to its nearest trajectory "
                         "sample for it to be assigned to a rectangle segment; points farther "
                         "than this (e.g. road visible beyond the trajectory's own extent) are "
                         "dropped from rectangle fitting. Defaults to 2x --segment_length.")

    p.add_argument("--grid_resolution", type=float, default=0.2,
                    help="Cell size (metres) of the top-down occupancy grid.")

    p.add_argument("--node_spacing", type=float, default=2.0,
                    help="Arc-length spacing (metres) between centerline graph nodes.")

    p.add_argument("--id2label_path", default=None,
                    help="Override path to id2label.json (defaults to "
                         "{panoptic_dir}/id2label.json).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seq = args.sequence.zfill(2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_seq_dir = Path(args.dataset_root) / seq
    poses_root = Path(args.poses_root) if args.poses_root else Path(args.dataset_root)

    poses_candidates = [
        poses_root / seq / "poses.txt",
        poses_root / f"{seq}.txt",
        poses_root / "poses" / f"{seq}.txt",
        Path(args.dataset_root) / seq / f"{seq}.txt",
    ]
    poses_path = next((c for c in poses_candidates if c.is_file() and c.stat().st_size > 0),
                       poses_candidates[0])

    calib_path = dataset_seq_dir / "calib.txt"
    depth_seq_dir = Path(args.depth_dir) / seq
    panoptic_seq_dir = Path(args.panoptic_dir) / seq
    id2label_path = Path(args.id2label_path) if args.id2label_path else \
        Path(args.panoptic_dir) / "id2label.json"

    for label, path in [("calib.txt", calib_path), ("poses.txt", poses_path),
                         ("depth dir", depth_seq_dir), ("panoptic dir", panoptic_seq_dir)]:
        if not Path(path).exists():
            sys.exit(f"[error] {label} not found: {path}")

    calib = parse_kitti_calib(str(calib_path))
    if "P2" not in calib:
        sys.exit("[error] P2 not found in calib.txt - expected KITTI odometry format.")
    fx, fy, cx, cy = extract_intrinsics(calib["P2"])
    print(f"[i] Camera intrinsics (P2): fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    all_poses = load_poses(str(poses_path))
    print(f"[i] Loaded {len(all_poses)} poses from {poses_path}")
    if len(all_poses) == 0:
        sys.exit(f"[error] 0 poses parsed from {poses_path}.")

    road_ids = ROAD_IDS
    if id2label_path.exists():
        id2label = load_id2label(str(id2label_path))
        road_ids = frozenset(lid for lid, name in id2label.items() if name in ROAD_LABEL_NAMES)
        if not road_ids:
            print(f"[warn] No '{ROAD_LABEL_NAMES}' label found in {id2label_path}; "
                  f"falling back to hardcoded Cityscapes road id(s): {ROAD_IDS}")
            road_ids = ROAD_IDS
    print(f"[i] Road label id(s): {sorted(road_ids)}")

    start = args.start_frame
    end = min(start + args.n_frames, len(all_poses))
    frame_indices = list(range(start, end))
    if not frame_indices:
        sys.exit(f"[error] No frames to process: start={start}, end={end}.")
    print(f"[i] Processing {len(frame_indices)} frames: "
          f"{frame_indices[0]:06d} -> {frame_indices[-1]:06d}")

    # ---- Per-frame road point extraction --------------------------------
    road_clouds: list[np.ndarray] = []
    traj_positions: list[np.ndarray] = []

    for frame_idx in frame_indices:
        stem = f"{frame_idx:06d}"
        depth_path = depth_seq_dir / f"{stem}.npz"
        panoptic_path = panoptic_seq_dir / f"{stem}.npz"

        if not depth_path.exists() or not panoptic_path.exists():
            print(f"  [skip] missing depth or panoptic for frame {stem}")
            continue

        depth = np.load(str(depth_path))["depth"].astype(np.float32)
        pan = load_panoptic(str(panoptic_path))
        panoptic_seg = pan["panoptic_seg"].astype(np.int32)
        segment_ids = pan["segment_ids"].astype(np.int32)
        label_ids = pan["label_ids"].astype(np.int32)

        if depth.shape != panoptic_seg.shape:
            import cv2
            panoptic_seg = cv2.resize(
                panoptic_seg, (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        pose = all_poses[frame_idx]
        traj_positions.append(pose[:3, 3].copy())

        pts = backproject_road_points(
            depth=depth, panoptic_seg=panoptic_seg,
            segment_ids=segment_ids, label_ids=label_ids,
            fx=fx, fy=fy, cx=cx, cy=cy, pose=pose,
            depth_trunc=args.depth_trunc, road_ids=road_ids,
        )
        if len(pts) > 0:
            road_clouds.append(pts)
        print(f"  frame {stem}: {len(pts):>7,} road pts")

    if not road_clouds:
        sys.exit("[error] No road points extracted from any frame - check road_ids / "
                  "panoptic label mapping.")

    road_points = np.concatenate(road_clouds, axis=0)
    traj_xyz = np.stack(traj_positions, axis=0)
    print(f"[i] Total road points before downsampling: {len(road_points):,}")

    # ---- Optional voxel downsample of raw road points --------------------
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(road_points.astype(np.float64))
        pcd_down = pcd.voxel_down_sample(args.voxel_size)
        road_points = np.asarray(pcd_down.points, dtype=np.float32)
        print(f"[i] Road points after voxel downsample ({args.voxel_size} m): "
              f"{len(road_points):,}")
    except ImportError:
        print("[warn] open3d not installed - using all raw road points "
              "(no downsampling). This may be slow for large n_frames.")

    # =======================================================================
    # 1. Road mesh
    # =======================================================================
    print("\n[i] Building road mesh ...")
    vertices, triangles = build_road_mesh(road_points, max_edge_length=args.mesh_max_edge)
    print(f"    {len(vertices):,} vertices, {len(triangles):,} triangles "
          f"(after gap filtering)")

    if args.mesh_format in ("ply", "both"):
        mesh_ply_path = out_dir / f"{seq}_road_mesh.ply"
        save_mesh_ply(str(mesh_ply_path), vertices, triangles)
        print(f"    [✓] {mesh_ply_path}")
    if args.mesh_format in ("obj", "both"):
        mesh_obj_path = out_dir / f"{seq}_road_mesh.obj"
        save_mesh_obj(str(mesh_obj_path), vertices, triangles)
        print(f"    [✓] {mesh_obj_path}")

    # =======================================================================
    # 2. Oriented road rectangles
    # =======================================================================
    print("\n[i] Fitting oriented road rectangles along trajectory ...")
    rectangles = build_road_rectangles(
        road_points, traj_xyz,
        segment_length=args.segment_length,
        min_points_per_segment=args.min_points_per_segment,
        max_lateral_assignment_dist=args.max_lateral_assignment_dist,
    )
    print(f"    {len(rectangles)} rectangle segments "
          f"(segment_length={args.segment_length} m)")
    if rectangles:
        widths = [2 * r["half_width"] for r in rectangles]
        print(f"    road width stats: min={min(widths):.2f} m, "
              f"median={float(np.median(widths)):.2f} m, max={max(widths):.2f} m")

    rect_json_path = out_dir / f"{seq}_road_rectangles.json"
    rect_npz_path = out_dir / f"{seq}_road_rectangles.npz"
    save_rectangles_json(str(rect_json_path), rectangles)
    save_rectangles_npz(str(rect_npz_path), rectangles)
    print(f"    [✓] {rect_json_path}")
    print(f"    [✓] {rect_npz_path}")

    # =======================================================================
    # 3. Occupancy grid + polygon
    # =======================================================================
    print("\n[i] Rasterising occupancy grid ...")
    grid, origin_xz, res = build_occupancy_grid(road_points, resolution=args.grid_resolution)
    print(f"    grid shape: {grid.shape}, resolution={res} m, "
          f"origin(x,z)={origin_xz}, drivable cells={int(grid.sum()):,}")
    polygon_rc = grid_to_polygon(grid)
    if polygon_rc is None:
        print("    [warn] skimage not available (or no contour found) - "
              "polygon boundary omitted, grid still saved.")
    else:
        print(f"    drivable-area boundary polygon: {len(polygon_rc)} points")

    grid_npz_path = out_dir / f"{seq}_occupancy_grid.npz"
    grid_png_path = out_dir / f"{seq}_occupancy_grid.png"
    save_occupancy_grid(str(grid_npz_path), str(grid_png_path), grid, origin_xz, res, polygon_rc)
    print(f"    [✓] {grid_npz_path}")
    print(f"    [✓] {grid_png_path}")

    # =======================================================================
    # 4. Centerline graph
    # =======================================================================
    print("\n[i] Building centerline graph ...")
    graph = build_centerline_graph(traj_xyz, road_points, node_spacing=args.node_spacing)
    print(f"    {len(graph['nodes'])} nodes, {len(graph['edges'])} edges "
          f"(node_spacing={args.node_spacing} m)")

    centerline_json_path = out_dir / f"{seq}_centerline_graph.json"
    centerline_npz_path = out_dir / f"{seq}_centerline_graph.npz"
    save_centerline_json(str(centerline_json_path), graph)
    save_centerline_npz(str(centerline_npz_path), graph)
    print(f"    [✓] {centerline_json_path}")
    print(f"    [✓] {centerline_npz_path}")

    print(f"\n[✓] All road-layout outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
