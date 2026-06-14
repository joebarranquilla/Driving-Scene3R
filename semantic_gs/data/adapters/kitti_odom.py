"""Real-data sequence loader for KITTI odometry.

Wires together the four inputs my module consumes and produces a
:class:`~semantic_gs.data.frame.Frame` per call:

* RGB frames        ``<sequence_dir>/image_2/<stem>.png``
* Intrinsics        ``<sequence_dir>/calib.txt`` -> ``P2``
* Camera poses      ``<pose_path>`` (cam-0 -> world, baseline-shifted to cam-2)
* Depth (teammate)  ``<depth_dir>/<stem>.npz``    key ``"depth"``
* Panoptic (mate)   ``<panoptic_dir>/<stem>.npz`` + sibling ``id2label.json``

This loader is read-only and lazy: only the requested frame is fetched
into memory, so iterating a 4000-frame sequence does not OOM.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import AbstractSet, Iterable

import numpy as np
from PIL import Image

from semantic_gs.data.adapters.depth_npz import load_depth_npz
from semantic_gs.data.adapters.panoptic_npz import (
    load_id2label_json,
    load_panoptic_npz,
)
from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.data.static_mask import (
    DEFAULT_UNRELIABLE_DEPTH_CLASSES,
    build_static_mask,
)
from semantic_gs.geometry.cameras import PinholeCamera
from semantic_gs.geometry.poses import (
    cam_i_to_world_from_cam0,
    parse_kitti_calib,
    parse_kitti_poses,
)


class KITTIOdomSequenceLoader(SequenceLoader):
    """Loads one KITTI odometry sequence with depth/panoptic from disk.

    Parameters
    ----------
    sequence_dir
        Path to a KITTI-odometry sequence dir, e.g.
        ``.../sequences/04``. Must contain ``calib.txt`` and a
        ``<image_subdir>/`` (default ``image_2``).
    depth_dir
        Directory holding per-frame depth NPZ files
        (output of ``run_mobilestereonet_inference.py``).
    panoptic_dir
        Directory holding per-frame panoptic NPZ files
        (output of ``run_mask2former_inference.py``).
    pose_path
        Path to the per-sequence KITTI poses ``.txt`` file
        (one 3x4 cam-0-to-world matrix per line).
    id2label_path
        Path to ``id2label.json``. Defaults to
        ``panoptic_dir.parent / "id2label.json"`` (matches the teammate
        script's output convention).
    camera_index
        Which ``P_i`` from ``calib.txt`` to use as the working camera.
        Default ``2`` = ``image_2`` (left colour).
    image_subdir
        Sub-directory inside ``sequence_dir`` holding the PNG frames.
        Default ``"image_2"``.
    dynamic_classes
        See :func:`~semantic_gs.data.static_mask.build_static_mask`.
    unreliable_depth_classes
        See :func:`~semantic_gs.data.static_mask.build_static_mask`.
        Defaults to :data:`~semantic_gs.data.static_mask.DEFAULT_UNRELIABLE_DEPTH_CLASSES`
        (which removes sky from the static mask).
    skip_incomplete_frames
        If ``True`` (default), frames missing a depth or panoptic NPZ are
        skipped with a one-line warning. If ``False``, the loader raises.
    frame_stems
        Optional explicit list of frame stems to load (e.g. ``["000000",
        "000010"]``). When ``None``, every PNG in ``image_subdir`` is
        considered.
    """

    def __init__(
        self,
        sequence_dir:             str | Path,
        depth_dir:                str | Path,
        panoptic_dir:             str | Path,
        pose_path:                str | Path,
        id2label_path:            str | Path | None = None,
        *,
        camera_index:             int = 2,
        image_subdir:             str = "image_2",
        dynamic_classes:          AbstractSet[str] | None = None,
        unreliable_depth_classes: AbstractSet[str] | None = DEFAULT_UNRELIABLE_DEPTH_CLASSES,
        skip_incomplete_frames:   bool = True,
        frame_stems:              Iterable[str] | None = None,
    ) -> None:
        # --- resolve & validate paths -----------------------------------
        self._sequence_dir = Path(sequence_dir)
        self._depth_dir    = Path(depth_dir)
        self._panoptic_dir = Path(panoptic_dir)
        self._pose_path    = Path(pose_path)
        self._image_dir    = self._sequence_dir / image_subdir
        self._calib_path   = self._sequence_dir / "calib.txt"

        if id2label_path is None:
            id2label_path = self._panoptic_dir.parent / "id2label.json"
        self._id2label_path = Path(id2label_path)

        for p in (self._image_dir, self._calib_path, self._depth_dir,
                  self._panoptic_dir, self._pose_path, self._id2label_path):
            if not p.exists():
                raise FileNotFoundError(f"required path not found: {p}")

        # --- intrinsics from calib.txt[P_i] -----------------------------
        calib = parse_kitti_calib(self._calib_path)
        p_key = f"P{int(camera_index)}"
        if p_key not in calib:
            raise KeyError(f"{self._calib_path}: missing {p_key} (have {list(calib)})")
        self._P_i: np.ndarray = calib[p_key]
        self._camera_index = int(camera_index)

        # --- poses (cam-0 -> world) -------------------------------------
        self._T_cam0_to_world: np.ndarray = parse_kitti_poses(self._pose_path)  # (N, 4, 4)

        # --- id2label ---------------------------------------------------
        self._id2label: dict[int, str] = load_id2label_json(self._id2label_path)

        # --- discover frames -------------------------------------------
        self._dynamic_classes          = dynamic_classes
        self._unreliable_depth_classes = unreliable_depth_classes
        self._skip_incomplete          = bool(skip_incomplete_frames)

        if frame_stems is None:
            stems = sorted(p.stem for p in self._image_dir.glob("*.png"))
        else:
            stems = sorted(frame_stems)
        if not stems:
            raise FileNotFoundError(f"no PNG files in {self._image_dir}")

        # We cannot determine intrinsic image size until we open the
        # first PNG. Cache it lazily on first __getitem__.
        self._camera: PinholeCamera | None = None

        self._frame_stems: list[str] = self._filter_and_validate_stems(stems)
        if not self._frame_stems:
            raise FileNotFoundError(
                f"no usable frames in {self._image_dir} after filtering "
                f"(missing depth/panoptic NPZs?)"
            )

    # ------------------------------------------------------------------
    # SequenceLoader interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"kitti_odom/{self._sequence_dir.name}"

    @property
    def id2label(self) -> Mapping[int, str]:
        return dict(self._id2label)

    def __len__(self) -> int:
        return len(self._frame_stems)

    def __getitem__(self, idx: int) -> Frame:
        if not 0 <= idx < len(self._frame_stems):
            raise IndexError(
                f"frame index {idx} out of range [0, {len(self._frame_stems)})"
            )
        stem = self._frame_stems[idx]
        pose_idx = int(stem)  # KITTI stems are zero-padded frame indices.
        if pose_idx >= self._T_cam0_to_world.shape[0]:
            raise IndexError(
                f"frame stem {stem!r} -> pose index {pose_idx} >= "
                f"#poses {self._T_cam0_to_world.shape[0]}"
            )

        rgb   = self._load_rgb(self._image_dir / f"{stem}.png")
        depth = load_depth_npz(self._depth_dir / f"{stem}.npz")
        pano  = load_panoptic_npz(self._panoptic_dir / f"{stem}.npz")

        # Lazy intrinsics cache (need PNG dims).
        H, W = rgb.shape[:2]
        if self._camera is None:
            self._camera = PinholeCamera.from_kitti_P(self._P_i, width=W, height=H)
        elif (self._camera.width, self._camera.height) != (W, H):
            raise ValueError(
                f"frame {stem}: RGB size ({W}x{H}) differs from cached "
                f"({self._camera.width}x{self._camera.height}); KITTI "
                f"frames in one sequence must share intrinsics."
            )
        cam: PinholeCamera = self._camera  # local alias for type checker

        if depth.shape != (H, W):
            raise ValueError(
                f"frame {stem}: depth shape {depth.shape} != RGB shape ({H}, {W})"
            )
        if pano.panoptic_seg.shape != (H, W):
            raise ValueError(
                f"frame {stem}: panoptic_seg shape {pano.panoptic_seg.shape} "
                f"!= RGB shape ({H}, {W})"
            )

        static_mask = build_static_mask(
            panoptic_seg            = pano.panoptic_seg,
            segment_ids             = pano.segment_ids,
            label_ids               = pano.label_ids,
            id2label                = self._id2label,
            dynamic_classes         = self._dynamic_classes,
            unreliable_depth_classes= self._unreliable_depth_classes,
        )

        T_cam2_to_world = cam_i_to_world_from_cam0(
            self._T_cam0_to_world[pose_idx], self._P_i
        )

        return Frame(
            frame_id      = stem,
            rgb           = rgb,
            depth         = depth,
            panoptic_seg  = pano.panoptic_seg,
            segment_ids   = pano.segment_ids,
            label_ids     = pano.label_ids,
            scores        = pano.scores,
            static_mask   = static_mask,
            camera        = cam,
            T_cam_to_world= T_cam2_to_world,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _filter_and_validate_stems(self, stems: list[str]) -> list[str]:
        """Keep only stems with matching depth + panoptic NPZs, with
        proper warnings."""
        kept: list[str] = []
        n_skipped = 0
        for s in stems:
            try:
                pose_idx = int(s)
            except ValueError:
                if self._skip_incomplete:
                    print(f"[WARN] kitti_odom: stem {s!r} is not an integer; skipping.")
                    n_skipped += 1
                    continue
                raise ValueError(f"non-integer KITTI stem: {s!r}")

            d_ok = (self._depth_dir    / f"{s}.npz").is_file()
            p_ok = (self._panoptic_dir / f"{s}.npz").is_file()
            pose_ok = pose_idx < self._T_cam0_to_world.shape[0]
            if d_ok and p_ok and pose_ok:
                kept.append(s)
                continue
            if not self._skip_incomplete:
                missing = []
                if not d_ok:    missing.append("depth")
                if not p_ok:    missing.append("panoptic")
                if not pose_ok: missing.append("pose")
                raise FileNotFoundError(
                    f"frame {s!r}: missing {', '.join(missing)} "
                    f"(set skip_incomplete_frames=True to ignore)"
                )
            n_skipped += 1
        if n_skipped:
            print(f"[INFO] kitti_odom: skipped {n_skipped} frame(s) with "
                  f"missing depth/panoptic/pose.")
        return kept




