"""Strict reader for the teammate's depth NPZ files.

Contract (from ``scripts/run_mobilestereonet_inference.py``):
    {output_dir}/{sequence}/{frame_stem}.npz
        "depth" -> float32 (H, W) in metres
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEPTH_KEY = "depth"


def load_depth_npz(path: str | Path) -> np.ndarray:
    """Load a teammate depth NPZ and return a ``float32 (H, W)`` array in metres.

    Validates strictly against the documented contract. Any deviation
    raises with a clear message so failures surface early.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"depth NPZ not found: {path}")

    with np.load(path) as data:
        if DEPTH_KEY not in data.files:
            raise KeyError(
                f"{path}: expected key '{DEPTH_KEY}', got {data.files}"
            )
        depth = np.asarray(data[DEPTH_KEY])

    if depth.ndim != 2:
        raise ValueError(
            f"{path}: depth must be 2-D (H, W); got shape {depth.shape}"
        )
    if depth.dtype != np.float32:
        # Cast but warn-in-message: keep deterministic dtype downstream.
        depth = depth.astype(np.float32, copy=False)

    # Sanity: no NaN-only depth maps slip through silently.
    if not np.isfinite(depth).any():
        raise ValueError(f"{path}: depth array has no finite values")

    return depth

