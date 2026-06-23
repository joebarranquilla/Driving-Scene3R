"""Real-data sequence loader for KITTI odometry with **SAM 3** masks.

SAM 3 is the project's sole segmentation source. This loader wires the four
inputs into a :class:`~semantic_gs.data.frame.Frame`:

* RGB frames        ``<sequence_dir>/image_2/<stem>.png``
* Intrinsics        ``<sequence_dir>/calib.txt`` -> ``P2``
* Camera poses      ``<pose_path>`` (cam-0 -> world, baseline-shifted to cam-2)
* Depth (teammate)  ``<depth_dir>/<stem>.npz``  key ``"depth"``
* SAM 3 (me)        ``<sam3_dir>/<stem>.npz`` + sibling ``concepts.json``

The SAM 3 prompts cover both DYNAMIC agents (car/pedestrian/cyclist) and the
static drivable surface (``road``). The static mask therefore excludes only
the *dynamic* concepts (configurable via ``dynamic_concepts``) and keeps road +
unsegmented background — see :meth:`Sam3Prediction.static_mask`, which uses
``build_static_mask(..., void_is_static=True)``. Sky is not prompted; rely on
``far_plane`` / depth truncation downstream to drop it.

Read-only and lazy: only the requested frame is fetched into memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from semantic_gs.data.adapters.depth_npz import load_depth_npz
from semantic_gs.data.adapters.sam3_npz import (
    DEFAULT_SAM3_DYNAMIC_CONCEPTS,
    load_concepts_json,
    load_sam3_npz,
)
from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.geometry.cameras import PinholeCamera
from semantic_gs.geometry.poses import (
    cam_i_to_world_from_cam0,
    parse_kitti_calib,
    parse_kitti_poses,
)


class KITTISam3SequenceLoader(SequenceLoader):
    """Loads one KITTI odometry sequence with depth + SAM 3 masks from disk.

    Parameters
    ----------
    sequence_dir
        Path to a KITTI-odometry sequence dir, e.g. ``.../sequences/04``.
        Must contain ``calib.txt`` and an ``<image_subdir>/``.
    depth_dir
        Directory holding per-frame depth NPZ files
        (output of ``run_mobilestereonet_inference.py``).
    sam3_dir
        Directory holding per-frame SAM 3 instance NPZ files
        (output of ``scripts/run_sam3_inference.py``, i.e.
        ``<output_dir>/<sequence>``).
    pose_path
        Path to the per-sequence KITTI poses ``.txt`` file
        (one 3x4 cam-0-to-world matrix per line).
    concepts_path
        Path to ``concepts.json``. Defaults to
        ``sam3_dir.parent / "concepts.json"`` (matches the SAM 3 script's
        output convention).
    camera_index
        Which ``P_i`` from ``calib.txt`` to use. Default ``2`` = ``image_2``.
    image_subdir
        Sub-directory inside ``sequence_dir`` holding the PNG frames.
    boundary_margin
        Pixels to erode around every instance boundary before building the
        static mask (stereo depth bleeds across edges). ``0`` disables it.
    dynamic_concepts
        Concept names removed from the static mask. Defaults to
        :data:`~semantic_gs.data.adapters.sam3_npz.DEFAULT_SAM3_DYNAMIC_CONCEPTS`
        (car/pedestrian/cyclist/...). Static concepts like ``road`` are kept.
    skip_incomplete_frames
        If ``True`` (default), frames missing a depth or SAM 3 NPZ are
        skipped with a warning. If ``False``, the loader raises.
    frame_stems
        Optional explicit list of frame stems to load. When ``None``, every
        PNG in ``image_subdir`` is considered.
    """

    def __init__(
        self,
        sequence_dir:           str | Path,
        depth_dir:              str | Path,
        sam3_dir:               str | Path,
        pose_path:              str | Path,
        concepts_path:          str | Path | None = None,
        *,
        camera_index:           int = 2,
        image_subdir:           str = "image_2",
        dynamic_concepts:       frozenset[str] | None = None,
        boundary_margin:        int = 0,
        skip_incomplete_frames: bool = True,
        frame_stems:            Iterable[str] | None = None,
    ) -> None:
        # --- resolve & validate paths -----------------------------------
        self._sequence_dir = Path(sequence_dir)
        self._depth_dir    = Path(depth_dir)
        self._sam3_dir     = Path(sam3_dir)
        self._pose_path    = Path(pose_path)
        self._image_dir    = self._sequence_dir / image_subdir
        self._calib_path   = self._sequence_dir / "calib.txt"

        if concepts_path is None:
            concepts_path = self._sam3_dir.parent / "concepts.json"
        self._concepts_path = Path(concepts_path)

        for p in (self._image_dir, self._calib_path, self._depth_dir,
                  self._sam3_dir, self._pose_path, self._concepts_path):
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

        # --- concepts (label index -> concept name) ---------------------
        self._concepts: dict[int, str] = load_concepts_json(self._concepts_path)

        # --- config -----------------------------------------------------
        self._dynamic_concepts = (dynamic_concepts if dynamic_concepts is not None
                                  else DEFAULT_SAM3_DYNAMIC_CONCEPTS)
        self._boundary_margin = int(boundary_margin)
        self._skip_incomplete = bool(skip_incomplete_frames)

        # --- discover frames -------------------------------------------
        if frame_stems is None:
            stems = sorted(p.stem for p in self._image_dir.glob("*.png"))
        else:
            stems = sorted(frame_stems)
        if not stems:
            raise FileNotFoundError(f"no PNG files in {self._image_dir}")

        # Intrinsic image size is unknown until the first PNG is opened.
        self._camera: PinholeCamera | None = None

        self._frame_stems: list[str] = self._filter_and_validate_stems(stems)
        if not self._frame_stems:
            raise FileNotFoundError(
                f"no usable frames in {self._image_dir} after filtering "
                f"(missing depth/SAM 3 NPZs?)"
            )

    # ------------------------------------------------------------------
    # SequenceLoader interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"kitti_odom_sam3/{self._sequence_dir.name}"

    @property
    def id2label(self) -> Mapping[int, str]:
        return dict(self._concepts)

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
        sam3  = load_sam3_npz(self._sam3_dir / f"{stem}.npz")

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
        if sam3.instance_seg.shape != (H, W):
            raise ValueError(
                f"frame {stem}: instance_seg shape {sam3.instance_seg.shape} "
                f"!= RGB shape ({H}, {W})"
            )

        static_mask = sam3.static_mask(
            id2label        = self._concepts,
            dynamic_classes = self._dynamic_concepts,
            boundary_margin = self._boundary_margin,
        )

        T_cam2_to_world = cam_i_to_world_from_cam0(
            self._T_cam0_to_world[pose_idx], self._P_i
        )

        return Frame(
            frame_id      = stem,
            rgb           = rgb,
            depth         = depth,
            panoptic_seg  = sam3.instance_seg,   # int32 (H, W), 0 = background
            segment_ids   = sam3.segment_ids,    # track_ids + 1 (match pixel values)
            label_ids     = sam3.label_ids,      # concept indices (id2label = concepts)
            scores        = sam3.scores,
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
        """Keep only stems with matching depth + SAM 3 NPZs + a pose."""
        kept: list[str] = []
        n_skipped = 0
        for s in stems:
            try:
                pose_idx = int(s)
            except ValueError:
                if self._skip_incomplete:
                    print(f"[WARN] kitti_odom_sam3: stem {s!r} is not an integer; skipping.")
                    n_skipped += 1
                    continue
                raise ValueError(f"non-integer KITTI stem: {s!r}")

            d_ok = (self._depth_dir / f"{s}.npz").is_file()
            s_ok = (self._sam3_dir  / f"{s}.npz").is_file()
            pose_ok = pose_idx < self._T_cam0_to_world.shape[0]
            if d_ok and s_ok and pose_ok:
                kept.append(s)
                continue
            if not self._skip_incomplete:
                missing = []
                if not d_ok:    missing.append("depth")
                if not s_ok:    missing.append("sam3")
                if not pose_ok: missing.append("pose")
                raise FileNotFoundError(
                    f"frame {s!r}: missing {', '.join(missing)} "
                    f"(set skip_incomplete_frames=True to ignore)"
                )
            n_skipped += 1
        if n_skipped:
            print(f"[INFO] kitti_odom_sam3: skipped {n_skipped} frame(s) with "
                  f"missing depth/sam3/pose.")
        return kept
