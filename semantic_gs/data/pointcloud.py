"""Sequence-aggregated semantic point-cloud contract.

Produced by the teammate's ``scripts/lift_to_semantic_pointcloud.py`` and
consumed by Phase 3 as the initialisation geometry for the static
semantic Gaussian Splatting model.

Coordinate-frame caveat
-----------------------
In the SAM 3-only pipeline this cloud is produced by
``semantic_gs.scripts.init_pc_from_loader``, which back-projects the static
pixels of :class:`~semantic_gs.data.adapters.kitti_odom_sam3.KITTISam3SequenceLoader`
frames using the very same ``Frame.T_cam_to_world`` that Phase-3 training uses
for supervision. Init geometry and training poses therefore share one
convention — no systematic cam-0/cam-2 baseline offset to correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SemanticPointCloud:
    """A sequence-aggregated static point cloud with semantic labels.

    Attributes
    ----------
    xyz
        ``float32 (N, 3)`` — world-coordinate points (metres).
    rgb
        ``float32 (N, 3)`` — RGB colour in ``[0, 1]``.
    labels
        ``int32 (N,)`` — Cityscapes semantic class ID per point.
    """

    xyz:    np.ndarray
    rgb:    np.ndarray
    labels: np.ndarray

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        n = self.xyz.shape[0]
        self._check("xyz",    self.xyz,    (n, 3), np.float32)
        self._check("rgb",    self.rgb,    (n, 3), np.float32)
        self._check("labels", self.labels, (n,),   np.int32)

        if n > 0:
            if not (self.rgb >= 0).all() or not (self.rgb <= 1).all():
                raise ValueError(
                    f"SemanticPointCloud.rgb must be in [0, 1]; "
                    f"got min={self.rgb.min()}, max={self.rgb.max()}"
                )
            if not np.isfinite(self.xyz).all():
                raise ValueError("SemanticPointCloud.xyz contains non-finite values")

    @staticmethod
    def _check(name: str, arr: np.ndarray, expected_shape: tuple, expected_dtype) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} != expected {expected_shape}")
        if arr.dtype != np.dtype(expected_dtype):
            raise TypeError(
                f"{name} dtype {arr.dtype} != expected {np.dtype(expected_dtype)}"
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min_xyz, max_xyz)`` world-space bounding box as ``float32 (3,)``."""
        if self.n_points == 0:
            zeros = np.zeros(3, dtype=np.float32)
            return zeros, zeros.copy()
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def class_counts(self) -> dict[int, int]:
        """``{class_id: n_points}`` — useful for diagnostics / class balancing."""
        if self.n_points == 0:
            return {}
        ids, counts = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

