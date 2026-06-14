#!/usr/bin/env python3
"""
Shared utilities for the Driving-Scene3R pipeline.

Covers:
  - Cityscapes label palette and derived helpers
  - KITTI calibration / pose I/O
  - Panoptic prediction I/O
  - PLY writer
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cityscapes label definitions
# ---------------------------------------------------------------------------

# (label_id, name, is_dynamic_vehicle, is_sky, rgb_colour_uint8)
CITYSCAPES_LABELS: list[tuple] = [
    (0,  "road",          False, False, (128,  64, 128)),
    (1,  "sidewalk",      False, False, (244,  35, 232)),
    (2,  "building",      False, False, ( 70,  70,  70)),
    (3,  "wall",          False, False, (102, 102, 156)),
    (4,  "fence",         False, False, (190, 153, 153)),
    (5,  "pole",          False, False, (153, 153, 153)),
    (6,  "traffic light", False, False, (250, 170,  30)),
    (7,  "traffic sign",  False, False, (220, 220,   0)),
    (8,  "vegetation",    False, False, (107, 142,  35)),
    (9,  "terrain",       False, False, (152, 251, 152)),
    (10, "sky",           False, True,  ( 70, 130, 180)),  # excluded
    (11, "person",        False, False, (220,  20,  60)),  # kept (static)
    (12, "rider",         False, False, (255,   0,   0)),  # kept (static)
    (13, "car",           True,  False, (  0,   0, 142)),  # excluded
    (14, "truck",         True,  False, (  0,   0,  70)),  # excluded
    (15, "bus",           True,  False, (  0,  60, 100)),  # excluded
    (16, "train",         True,  False, (  0,  80, 100)),  # excluded
    (17, "motorcycle",    True,  False, (  0,   0, 230)),  # excluded
    (18, "bicycle",       True,  False, (119,  11,  32)),  # excluded
]

# label_id → float32 RGB [0, 1]  (for 3-D point-cloud colouring)
ID_TO_COLOR: dict[int, np.ndarray] = {
    lid: np.array(col, dtype=np.float32) / 255.0
    for lid, _, _, _, col in CITYSCAPES_LABELS
}

# label_id → uint8 RGB tuple  (for 2-D visualisation)
ID_TO_COLOR_U8: dict[int, tuple[int, int, int]] = {
    lid: col for lid, _, _, _, col in CITYSCAPES_LABELS
}

# Default exclusion set: sky + all dynamic vehicle classes
EXCLUDED_IDS: frozenset[int] = frozenset(
    lid for lid, _, is_dyn, is_sky, _ in CITYSCAPES_LABELS
    if is_dyn or is_sky
)


def color_from_label_id(label_id: int) -> np.ndarray:
    """Return the Cityscapes uint8 (3,) colour for *label_id*; neutral grey fallback."""
    return np.array(ID_TO_COLOR_U8.get(label_id, (128, 128, 128)), dtype=np.uint8)


def build_label_exclusion_set(id2label: dict[int, str]) -> frozenset[int]:
    """
    Build the exclusion set from a model's {label_id: class_name} mapping.

    Matches class names against the Cityscapes dynamic/sky labels
    (case-insensitive).  Falls back to ``EXCLUDED_IDS`` when no match is
    found (e.g. id2label is empty or uses different class names).
    """
    excluded_names = {
        name.lower()
        for _, name, is_dyn, is_sky, _ in CITYSCAPES_LABELS
        if is_dyn or is_sky
    }
    excluded = {
        int(lid)
        for lid, name in id2label.items()
        if name.lower() in excluded_names
    }
    return frozenset(excluded) if excluded else EXCLUDED_IDS


def semantic_boundary_mask(label_map: np.ndarray, margin: int) -> np.ndarray:
    """
    Return a boolean mask that is True within *margin* pixels of any semantic
    class boundary in *label_map*.

    These pixels are candidates for depth-bleeding artefacts produced by the
    stereo aggregation kernel straddling two different semantic classes.
    Returns an all-False mask (no-op) when *margin* ≤ 0.
    """
    if margin <= 0:
        return np.zeros(label_map.shape, dtype=bool)

    from scipy.ndimage import binary_dilation

    # 4-connected boundary: True wherever any axis-aligned neighbour differs.
    edge = np.zeros(label_map.shape, dtype=bool)
    edge[:-1, :] |= label_map[:-1, :] != label_map[1:,  :]
    edge[1:,  :] |= label_map[:-1, :] != label_map[1:,  :]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:,  1:]
    edge[:,  1:] |= label_map[:, :-1] != label_map[:,  1:]

    struct = np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)
    return binary_dilation(edge, structure=struct)


# ---------------------------------------------------------------------------
# KITTI calibration
# ---------------------------------------------------------------------------

def parse_kitti_calib(calib_path: str) -> dict[str, np.ndarray]:
    """
    Parse a KITTI odometry calib.txt and return a dict mapping each camera
    key (``P0``–``P3``, ``Tr``) to its raw float64 coefficient array.
    """
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
    """Return *(fx, fy, cx, cy)* from a KITTI 3×4 projection matrix."""
    P = P.reshape(3, 4)
    return float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])


def focal_and_baseline(calib_path: str) -> tuple[float, float]:
    """
    Return *(focal_length_px, baseline_m)* for the colour stereo pair
    (cameras 2 and 3) from a KITTI calib.txt.

    Used by MobileStereoNet to convert disparity maps to metric depth:
    ``depth = focal_length * baseline / disparity``.
    """
    data = parse_kitti_calib(calib_path)
    P2 = data["P2"].reshape(3, 4)
    P3 = data["P3"].reshape(3, 4)
    focal_length = float(P2[0, 0])
    baseline = float(abs(P2[0, 3] - P3[0, 3]) / focal_length)
    return focal_length, baseline


# ---------------------------------------------------------------------------
# KITTI pose loading
# ---------------------------------------------------------------------------

def load_poses(poses_path: str) -> list[np.ndarray]:
    """
    Load KITTI odometry poses.txt.

    Each line contains 12 space-separated values forming a 3×4 cam-to-world
    matrix [R | t].  Returns a list of 4×4 float64 homogeneous matrices.
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
# Panoptic prediction I/O
# ---------------------------------------------------------------------------

def load_panoptic(npz_path: str) -> dict:
    """
    Load one Mask2Former panoptic prediction NPZ and return its arrays.

    Keys: ``panoptic_seg`` (H,W) int32, ``segment_ids`` (N,) int32,
    ``label_ids`` (N,) int32, ``scores`` (N,) float32.
    """
    data = np.load(npz_path)
    required = ["panoptic_seg", "segment_ids", "label_ids", "scores"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Prediction file missing keys {missing}: {npz_path}. "
            "Make sure it was produced by run_mask2former_inference.py."
        )
    return {
        "panoptic_seg": data["panoptic_seg"],
        "segment_ids":  data["segment_ids"],
        "label_ids":    data["label_ids"],
        "scores":       data["scores"],
    }


def load_id2label(json_path: str) -> dict[int, str]:
    """Load an id2label.json file and return ``{int_id: class_name}``."""
    with open(json_path) as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def save_ply(
    path: str,
    xyz: np.ndarray,                      # (N, 3) float32
    colors: np.ndarray,                   # (N, 3) float32 [0-1]
    labels: Optional[np.ndarray] = None,  # (N,)   int32 – optional scalar
) -> None:
    """
    Write a coloured PLY file.

    When *labels* is provided an extra integer property ``label`` is appended
    so tools like CloudCompare / MeshLab can colour-by-scalar after loading.
    """
    N = len(xyz)
    cols_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with_labels = labels is not None
    with open(path, "w") as fh:
        fh.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        if with_labels:
            fh.write("property int label\n")
        fh.write("end_header\n")
        if with_labels:
            for (x, y, z), (r, g, b), lbl in zip(xyz, cols_u8, labels):
                fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {lbl}\n")
        else:
            for (x, y, z), (r, g, b) in zip(xyz, cols_u8):
                fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    print(f"[✓] Saved PLY  → {path}  ({N:,} points)")
