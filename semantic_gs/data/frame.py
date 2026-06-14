"""The ``Frame`` dataclass — the single source of truth for what one
loaded frame contains in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_gs.geometry.cameras import PinholeCamera


@dataclass
class Frame:
    """One time-stamped frame of a driving sequence.

    All array shapes are validated in :meth:`__post_init__`. See
    ``semantic_gs/README.md`` for unit and convention details.

    Parameters
    ----------
    frame_id
        A short identifier, typically the original file stem (``"000042"``).
    rgb
        ``uint8`` ``(H, W, 3)`` — channel order **RGB** (not BGR).
    depth
        ``float32`` ``(H, W)`` — metres; ``0`` or ``NaN`` means invalid.
    panoptic_seg
        ``int32`` ``(H, W)`` — segment ID per pixel, ``0 = void``.
    segment_ids
        ``int32`` ``(N,)`` — unique segment IDs present this frame.
    label_ids
        ``int32`` ``(N,)`` — semantic class ID of each segment
        (in the dataset's id2label space, e.g. Cityscapes).
    scores
        ``float32`` ``(N,)`` — segmentation confidence per segment.
    static_mask
        ``bool`` ``(H, W)`` — ``True`` = pixel suitable for static GS init.
    camera
        Intrinsics of the camera that produced ``rgb`` / ``depth``.
    T_cam_to_world
        ``float64`` ``(4, 4)`` — camera-to-world transform (see README).
    """

    frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    panoptic_seg: np.ndarray
    segment_ids: np.ndarray
    label_ids: np.ndarray
    scores: np.ndarray
    static_mask: np.ndarray
    camera: PinholeCamera
    T_cam_to_world: np.ndarray

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        H, W = self.camera.height, self.camera.width

        self._check_array("rgb",          self.rgb,          (H, W, 3), np.uint8)
        self._check_array("depth",        self.depth,        (H, W),    np.float32)
        self._check_array("panoptic_seg", self.panoptic_seg, (H, W),    np.int32)
        self._check_array("static_mask",  self.static_mask,  (H, W),    np.bool_)

        # 1-D per-segment arrays must all have the same length.
        n = self.segment_ids.shape[0]
        self._check_array("segment_ids", self.segment_ids, (n,), np.int32)
        self._check_array("label_ids",   self.label_ids,   (n,), np.int32)
        self._check_array("scores",      self.scores,      (n,), np.float32)

        if self.T_cam_to_world.shape != (4, 4):
            raise ValueError(
                f"Frame[{self.frame_id}]: T_cam_to_world must be (4, 4), "
                f"got {self.T_cam_to_world.shape}"
            )
        if self.T_cam_to_world.dtype != np.float64:
            raise TypeError(
                f"Frame[{self.frame_id}]: T_cam_to_world must be float64, "
                f"got {self.T_cam_to_world.dtype}"
            )
        # Bottom row should be [0, 0, 0, 1] for a proper homogeneous transform.
        if not np.allclose(self.T_cam_to_world[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(
                f"Frame[{self.frame_id}]: T_cam_to_world bottom row is "
                f"{self.T_cam_to_world[3]}, expected [0, 0, 0, 1]"
            )

        # Depth sanity: no negative values, finite where the mask is True.
        if np.any(self.depth[np.isfinite(self.depth)] < 0):
            raise ValueError(f"Frame[{self.frame_id}]: depth contains negative values")

    # ------------------------------------------------------------------
    @staticmethod
    def _check_array(
        name: str,
        arr: np.ndarray,
        expected_shape: tuple,
        expected_dtype: type,
    ) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape {arr.shape} != expected {expected_shape}"
            )
        # np.bool_ comparison: dtype.type subclasses numpy.bool_ for bool arrays.
        if expected_dtype is np.bool_:
            if arr.dtype != np.bool_:
                raise TypeError(
                    f"{name} dtype {arr.dtype} != expected bool"
                )
        elif arr.dtype != np.dtype(expected_dtype):
            raise TypeError(
                f"{name} dtype {arr.dtype} != expected {np.dtype(expected_dtype)}"
            )

