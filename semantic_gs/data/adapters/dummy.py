"""Fully-synthetic sequence loader for offline / pre-teammate testing.

Generates a deterministic 5-frame "scene" with four panoptic segments
(sky, building, road, car) so that every code path in the pipeline can
be exercised without any real KITTI/Mask2Former/MobileStereoNet outputs.

The synthetic class IDs match the standard Cityscapes 19-class trainId
mapping used by ``facebook/mask2former-swin-large-cityscapes-panoptic``,
so the static-mask filter has to actually remove the "car" segment to
pass the Phase-0 tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.data.static_mask import build_static_mask
from semantic_gs.geometry.cameras import PinholeCamera


# Cityscapes 19-class trainIds (subset relevant to the dummy scene).
_DUMMY_ID2LABEL: dict[int, str] = {
    0:  "road",
    2:  "building",
    8:  "vegetation",
    10: "sky",
    11: "person",
    13: "car",
}

# Per-segment colors (RGB, uint8) used to "render" the dummy RGB image.
_SEG_COLOR: dict[int, tuple[int, int, int]] = {
    1: (135, 206, 235),   # sky      — light blue
    2: (139, 119, 101),   # building — brown
    3: (90, 90, 90),      # road     — grey
    4: (200, 30, 30),     # car      — red
}
# Per-segment Cityscapes class IDs.
_SEG_LABEL_ID: dict[int, int] = {1: 10, 2: 2, 3: 0, 4: 13}
# Per-segment depth (metres) used when building the dummy depth map.
_SEG_DEPTH_M: dict[int, float] = {1: 1000.0, 2: 12.0, 3: 0.0, 4: 10.0}
# (road depth is overwritten with a vertical ramp below)


@dataclass(frozen=True)
class _DummySceneSpec:
    """Static layout of the dummy scene (independent of frame index)."""

    width: int = 320
    height: int = 96
    sky_rows: int = 25                                # rows [0, sky_rows)
    building_cols: int = 80                           # cols [0, building_cols)
    car_bbox: tuple[int, int, int, int] = (55, 150, 75, 200)  # (y0, x0, y1, x1)
    road_depth_near_m: float = 4.0   # row = height - 1 → close to camera
    road_depth_far_m: float = 50.0   # row = sky_rows  → far from camera


class DummySequenceLoader(SequenceLoader):
    """Deterministic synthetic loader producing :class:`Frame` objects.

    Parameters
    ----------
    num_frames
        Number of frames in the sequence (default: ``5``).
    width, height
        Override image size for fast unit tests.
    fx, fy, cx, cy
        Override intrinsics; defaults are sensible for the chosen size.
    forward_step_m
        How far the camera advances along +Z between consecutive frames.
    """

    def __init__(
        self,
        num_frames: int = 5,
        width: int | None = None,
        height: int | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        forward_step_m: float = 1.0,
    ) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")

        # If user overrides width/height, rescale the scene layout so the
        # car bbox and sky band stay roughly in the same fractional place.
        base = _DummySceneSpec()
        W = int(width)  if width  is not None else base.width
        H = int(height) if height is not None else base.height
        sy = lambda v: max(1, int(round(v * H / base.height)))   # noqa: E731
        sx = lambda v: max(1, int(round(v * W / base.width)))    # noqa: E731
        self._spec = _DummySceneSpec(
            width=W,
            height=H,
            sky_rows=sy(base.sky_rows),
            building_cols=sx(base.building_cols),
            car_bbox=(
                sy(base.car_bbox[0]),
                sx(base.car_bbox[1]),
                sy(base.car_bbox[2]),
                sx(base.car_bbox[3]),
            ),
            road_depth_near_m=base.road_depth_near_m,
            road_depth_far_m=base.road_depth_far_m,
        )

        self._camera = PinholeCamera(
            width=W,
            height=H,
            fx=float(fx) if fx is not None else 0.9 * W,
            fy=float(fy) if fy is not None else 0.9 * W,
            cx=float(cx) if cx is not None else 0.5 * W,
            cy=float(cy) if cy is not None else 0.5 * H,
        )
        self._num_frames = int(num_frames)
        self._forward_step_m = float(forward_step_m)

        # Pre-compute the per-pixel panoptic_seg / RGB / depth — they are
        # identical across frames in the dummy world (the camera just
        # slides forward, but the synthetic content is repainted from
        # the same template). This keeps Phase 0 about plumbing, not
        # physically-consistent re-projection (that arrives in Phase 2).
        self._tpl_panoptic, self._tpl_rgb, self._tpl_depth = self._build_template()

    # ------------------------------------------------------------------
    # SequenceLoader interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"dummy/{self._num_frames}f_{self._spec.width}x{self._spec.height}"

    @property
    def id2label(self) -> Mapping[int, str]:
        return dict(_DUMMY_ID2LABEL)

    @property
    def car_bbox(self) -> tuple[int, int, int, int]:
        """``(y0, x0, y1, x1)`` of the dynamic-car region, for tests."""
        return self._spec.car_bbox

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> Frame:
        if not 0 <= idx < self._num_frames:
            raise IndexError(f"frame index {idx} out of range [0, {self._num_frames})")

        # Per-segment metadata is identical across frames.
        segment_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        label_ids   = np.array(
            [_SEG_LABEL_ID[s] for s in segment_ids.tolist()], dtype=np.int32
        )
        scores      = np.array([0.99, 0.98, 0.99, 0.95], dtype=np.float32)

        panoptic_seg = self._tpl_panoptic.copy()
        rgb          = self._tpl_rgb.copy()
        depth        = self._tpl_depth.copy()

        static_mask = build_static_mask(
            panoptic_seg=panoptic_seg,
            segment_ids=segment_ids,
            label_ids=label_ids,
            id2label=_DUMMY_ID2LABEL,
        )

        # Pose: identity rotation, translate +Z by idx * step.
        T = np.eye(4, dtype=np.float64)
        T[2, 3] = idx * self._forward_step_m

        return Frame(
            frame_id=f"{idx:06d}",
            rgb=rgb,
            depth=depth,
            panoptic_seg=panoptic_seg,
            segment_ids=segment_ids,
            label_ids=label_ids,
            scores=scores,
            static_mask=static_mask,
            camera=self._camera,
            T_cam_to_world=T,
        )

    # ------------------------------------------------------------------
    # Scene synthesis
    # ------------------------------------------------------------------
    def _build_template(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self._spec
        H, W = s.height, s.width

        # Start everything as "road" (segment 3) and overwrite from back to
        # front: road → building → car → sky.
        panoptic = np.full((H, W), fill_value=3, dtype=np.int32)
        # Building on the left side, below the sky band.
        panoptic[s.sky_rows:, : s.building_cols] = 2
        # Car rectangle on the road.
        y0, x0, y1, x1 = s.car_bbox
        panoptic[y0:y1, x0:x1] = 4
        # Sky on top covers everything in those rows.
        panoptic[: s.sky_rows, :] = 1

        # RGB: paint per-segment flat colors.
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for seg_id, color in _SEG_COLOR.items():
            rgb[panoptic == seg_id] = color

        # Depth: per-segment, with a vertical ramp on the road so that
        # back-projection produces a non-degenerate point cloud later.
        depth = np.zeros((H, W), dtype=np.float32)
        for seg_id, d in _SEG_DEPTH_M.items():
            depth[panoptic == seg_id] = d
        # Road ramp: linear from (row = sky_rows → far) to (row = H-1 → near).
        road_mask = panoptic == 3
        row_indices = np.arange(H, dtype=np.float32)
        t = (row_indices - s.sky_rows) / max(1, (H - 1 - s.sky_rows))
        t = np.clip(t, 0.0, 1.0)
        ramp = s.road_depth_far_m + t * (s.road_depth_near_m - s.road_depth_far_m)
        ramp_2d = np.broadcast_to(ramp[:, None], (H, W))
        depth = np.where(road_mask, ramp_2d, depth).astype(np.float32)

        return panoptic, rgb, depth

