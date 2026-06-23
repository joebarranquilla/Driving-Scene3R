"""Strict reader for the SAM 3 instance-segmentation NPZ files + ``concepts.json``.

Contract (from ``scripts/run_sam3_inference.py``):
    {output_dir}/concepts.json   -> {"0": "car", "1": "pedestrian", "2": "cyclist"}
    {output_dir}/{sequence}/{frame_stem}.npz
        "instance_seg" -> int32   (H, W)   (track_id + 1) per pixel (0 = background)
        "track_ids"    -> int32   (N,)     persistent tracking IDs this frame
        "label_ids"    -> int32   (N,)     concept index (into concepts.json)
        "scores"       -> float32 (N,)     detection confidence per instance

SAM 3 only segments the *dynamic* concepts it is prompted with (car,
pedestrian, cyclist), so every instance in a SAM 3 frame is dynamic. This
makes the static mask trivial: a pixel is static iff it belongs to no
instance, i.e. ``instance_seg == 0``. Contrast with the Mask2Former path
(``panoptic_npz.py`` + ``static_mask.build_static_mask``), which must match
dynamic *classes by name* because its panoptic output also covers the static
"stuff" classes (road, building, sky, ...).

``instance_seg`` already satisfies the ``Frame.panoptic_seg`` contract
(``int32 (H, W)``, ``0`` = void/background), so the SAM 3 → ``Frame`` mapping
is direct:

    panoptic_seg = instance_seg
    segment_ids  = track_ids + 1   # +1 so values match the instance_seg encoding
    label_ids    = label_ids       # concept indices; id2label = concepts.json
    scores       = scores
    static_mask  = instance_seg == 0

The ``+1`` offset on ``segment_ids`` mirrors the encoding baked into
``instance_seg`` (the SAM 3 tracker assigns IDs from 0, which would otherwise
collide with the background value 0). The original SAM 3 track id of a segment
is therefore ``segment_id - 1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_gs.data.static_mask import semantic_boundary_mask


SAM3_KEYS = ("instance_seg", "track_ids", "label_ids", "scores")


@dataclass(frozen=True)
class Sam3Prediction:
    """One frame's SAM 3 instance prediction loaded from disk."""

    instance_seg: np.ndarray   # int32   (H, W)  (track_id + 1), 0 = background
    track_ids:    np.ndarray   # int32   (N,)
    label_ids:    np.ndarray   # int32   (N,)    concept index
    scores:       np.ndarray   # float32 (N,)

    # ------------------------------------------------------------------
    # Frame-contract adapters
    # ------------------------------------------------------------------
    @property
    def segment_ids(self) -> np.ndarray:
        """Per-instance segment IDs consistent with ``instance_seg`` values.

        ``instance_seg`` stores ``track_id + 1`` per pixel, so the segment ID
        you would look up with ``instance_seg == segment_id`` is ``track_id + 1``.
        """
        return (self.track_ids + 1).astype(np.int32, copy=False)

    def static_mask(self, boundary_margin: int = 0) -> np.ndarray:
        """Return the ``bool (H, W)`` static mask for GS initialisation.

        A pixel is static iff it belongs to no SAM 3 instance
        (``instance_seg == 0``). Because every SAM 3 instance is a dynamic
        agent, this single rule removes all moving objects — no class-name
        lookup is needed (unlike the Mask2Former path).

        Parameters
        ----------
        boundary_margin
            If > 0, also discard pixels within this many pixels of any
            instance boundary. Stereo depth bleeds across instance edges,
            producing floating artefacts in the back-projected cloud; eroding
            the boundary is the same cheap fix the Mask2Former path applies
            (see ``static_mask.build_static_mask``). ``0`` disables it.
        """
        static = self.instance_seg == 0
        if boundary_margin > 0:
            boundary = semantic_boundary_mask(self.instance_seg, boundary_margin)
            static = static & ~boundary
        return static.astype(bool, copy=False)


def load_sam3_npz(path: str | Path) -> Sam3Prediction:
    """Load a SAM 3 instance NPZ as a :class:`Sam3Prediction`.

    Casts ``instance_seg`` / ``track_ids`` / ``label_ids`` to ``int32`` and
    ``scores`` to ``float32`` for a stable contract downstream.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SAM 3 NPZ not found: {path}")

    with np.load(path) as data:
        missing = [k for k in SAM3_KEYS if k not in data.files]
        if missing:
            raise KeyError(
                f"{path}: missing keys {missing}; found {data.files}"
            )
        instance_seg = np.asarray(data["instance_seg"])
        track_ids    = np.asarray(data["track_ids"])
        label_ids    = np.asarray(data["label_ids"])
        scores       = np.asarray(data["scores"])

    if instance_seg.ndim != 2:
        raise ValueError(
            f"{path}: instance_seg must be (H, W); got {instance_seg.shape}"
        )
    if not (track_ids.shape == label_ids.shape == scores.shape and track_ids.ndim == 1):
        raise ValueError(
            f"{path}: track_ids/label_ids/scores must be 1-D and same length; "
            f"got {track_ids.shape}, {label_ids.shape}, {scores.shape}"
        )

    return Sam3Prediction(
        instance_seg=instance_seg.astype(np.int32, copy=False),
        track_ids   =track_ids.astype(np.int32, copy=False),
        label_ids   =label_ids.astype(np.int32, copy=False),
        scores      =scores.astype(np.float32, copy=False),
    )


def load_concepts_json(path: str | Path) -> dict[int, str]:
    """Load the SAM 3 ``concepts.json`` (string-keyed) as a ``dict[int, str]``.

    This is the SAM 3 analogue of Mask2Former's ``id2label.json``: it maps
    each ``label_ids`` value to its concept name (e.g. ``{0: "car"}``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"concepts.json not found: {path}")
    with path.open("r") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: keys must be integer-parseable; {exc}") from exc
