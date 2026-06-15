"""Strict reader for the teammate's PLY semantic point cloud.

Backed by the standard :mod:`plyfile` library so we transparently handle
both ASCII and binary PLYs (the teammate currently emits ASCII, but
nothing prevents them from switching).

Expected per-vertex properties (the schema
``scripts/utils.py::save_ply`` on the ``lift-semantic`` branch
produces)::

    x, y, z          (float32)
    red, green, blue (uchar)
    label            (int, OPTIONAL)

Anything else raises with a clear error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

from semantic_gs.data.pointcloud import SemanticPointCloud


# Property names we know how to read (in the order the writer emits them).
_REQUIRED_PROPS = ("x", "y", "z", "red", "green", "blue")


def load_semantic_pointcloud_ply(path: str | Path) -> SemanticPointCloud:
    """Load a teammate semantic-PLY file as :class:`SemanticPointCloud`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"semantic point-cloud PLY not found: {path}")

    try:
        ply = PlyData.read(str(path))
    except Exception as e:                            # plyfile raises various
        raise ValueError(f"{path}: failed to parse PLY ({e})") from e

    if "vertex" not in (el.name for el in ply.elements):
        raise ValueError(f"{path}: no 'vertex' element in PLY")

    vert = ply["vertex"]
    present = set(vert.data.dtype.names or ())

    missing = [p for p in _REQUIRED_PROPS if p not in present]
    if missing:
        raise ValueError(
            f"{path}: missing required vertex properties {missing} "
            f"(have {sorted(present)})"
        )

    n = len(vert)
    xyz = np.stack(
        [np.asarray(vert["x"]),
         np.asarray(vert["y"]),
         np.asarray(vert["z"])],
        axis=1,
    ).astype(np.float32, copy=False)
    rgb = np.stack(
        [np.asarray(vert["red"]),
         np.asarray(vert["green"]),
         np.asarray(vert["blue"])],
        axis=1,
    ).astype(np.float32, copy=False) / 255.0

    if "label" in present:
        labels = np.asarray(vert["label"]).astype(np.int32, copy=False)
    else:
        labels = np.zeros(n, dtype=np.int32)

    return SemanticPointCloud(xyz=xyz, rgb=rgb, labels=labels)


