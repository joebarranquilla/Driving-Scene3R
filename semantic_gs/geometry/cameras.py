"""Pinhole-camera dataclass.

Camera convention: **OpenCV** (+X right, +Y down, +Z forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """An ideal pinhole camera with no distortion.

    Attributes
    ----------
    width, height : int
        Image dimensions in pixels.
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"PinholeCamera: width/height must be positive, "
                f"got width={self.width}, height={self.height}"
            )
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"PinholeCamera: focal lengths must be positive, "
                f"got fx={self.fx}, fy={self.fy}"
            )
        # cx/cy are usually inside [0, W/H], but rectified cameras can have
        # principal points slightly outside the image; we therefore do not
        # constrain them.

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix, ``float64``."""
        return np.array(
            [
                [self.fx, 0.0,     self.cx],
                [0.0,     self.fy, self.cy],
                [0.0,     0.0,     1.0    ],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_kitti_P(cls, P: np.ndarray, width: int, height: int) -> "PinholeCamera":
        """Build a :class:`PinholeCamera` from a KITTI ``P2``-style 3x4 matrix.

        The translation column ``P[:, 3]`` encodes the rectified-camera offset
        from cam-0 and is NOT part of the intrinsics. It is dropped here.
        """
        P = np.asarray(P)
        if P.shape != (3, 4):
            raise ValueError(f"KITTI P matrix must be 3x4, got {P.shape}")
        return cls(
            width=int(width),
            height=int(height),
            fx=float(P[0, 0]),
            fy=float(P[1, 1]),
            cx=float(P[0, 2]),
            cy=float(P[1, 2]),
        )

