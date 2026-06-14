"""Strict reader for the teammate's optional semantic NPZ.

Produced by ``scripts/lift_to_semantic_pointcloud.py --save_npz``::

    xyz    : float32 (N, 3)   world-coordinate points (metres)
    colors : float32 (N, 3)   RGB in [0, 1]
    labels : int32   (N,)     Cityscapes class IDs

The NPZ is functionally identical to the PLY adapter's output but loads
~10x faster for clouds with millions of points. Prefer it when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semantic_gs.data.pointcloud import SemanticPointCloud


REQUIRED_KEYS = ("xyz", "colors", "labels")


def load_semantic_pointcloud_npz(path: str | Path) -> SemanticPointCloud:
    """Load a teammate semantic-NPZ file as :class:`SemanticPointCloud`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"semantic point-cloud NPZ not found: {path}")

    with np.load(path) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; found {data.files}")
        xyz    = np.asarray(data["xyz"])
        colors = np.asarray(data["colors"])
        labels = np.asarray(data["labels"])

    return SemanticPointCloud(
        xyz    = xyz.astype(np.float32, copy=False),
        rgb    = colors.astype(np.float32, copy=False),
        labels = labels.astype(np.int32, copy=False),
    )

