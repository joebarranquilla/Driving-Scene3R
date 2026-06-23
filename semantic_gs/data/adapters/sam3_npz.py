"""Strict reader for the SAM 3 instance-segmentation NPZ files + ``concepts.json``.

Contract (from ``scripts/run_sam3_inference.py``):
    {output_dir}/concepts.json   -> {"0": "car", ..., "3": "road"}
    {output_dir}/{sequence}/{frame_stem}.npz
        "instance_seg" -> int32   (H, W)   (track_id + 1) per pixel (0 = background)
        "track_ids"    -> int32   (N,)     persistent tracking IDs this frame
        "label_ids"    -> int32   (N,)     concept index (into concepts.json)
        "scores"       -> float32 (N,)     detection confidence per instance

SAM 3 is the project's sole segmentation source. It segments the *prompted*
concepts only — both DYNAMIC agents (car / pedestrian / cyclist) and at least
one STATIC concept, the drivable surface (``road``). So "every instance is
dynamic" is NO LONGER true: the static mask must exclude only the dynamic
concepts (by name) and KEEP road + unsegmented background.

``instance_seg`` already satisfies the ``Frame.panoptic_seg`` contract
(``int32 (H, W)``, ``0`` = background), so the SAM 3 → ``Frame`` mapping is:

    panoptic_seg = instance_seg
    segment_ids  = track_ids + 1   # +1 so values match the instance_seg encoding
    label_ids    = label_ids       # concept indices; id2label = concepts.json
    scores       = scores
    static_mask  = build_static_mask(..., void_is_static=True)  # see below

The static mask reuses ``static_mask.build_static_mask`` (the same name-based
exclusion used for the old Mask2Former path) with **``void_is_static=True``**:
SAM 3 only labels prompted concepts, so background (``instance_seg == 0``) is
the rest of the static scene and must be KEPT (the opposite of Mask2Former,
where void = uncertain). Only segments whose concept is in the dynamic set are
removed; road and background stay.

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

from semantic_gs.data.static_mask import build_static_mask


SAM3_KEYS = ("instance_seg", "track_ids", "label_ids", "scores")

# SAM 3 prompt concepts that are dynamic agents (removed from the static scene).
# Name-based (lower-cased) like ``static_mask.DEFAULT_DYNAMIC_CLASSES`` so it is
# robust to concept re-ordering; uses the SAM 3 prompt vocabulary.
DEFAULT_SAM3_DYNAMIC_CONCEPTS: frozenset[str] = frozenset({
    "car", "pedestrian", "cyclist", "truck", "bus", "motorcycle", "bicycle",
})

# Concept names that denote the drivable surface (robust to phrasing).
DEFAULT_ROAD_CONCEPTS: frozenset[str] = frozenset({
    "road", "drivable surface", "street",
})


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

    def static_mask(
        self,
        id2label,
        dynamic_classes=None,
        boundary_margin: int = 0,
    ) -> np.ndarray:
        """Return the ``bool (H, W)`` static mask for GS initialisation.

        A pixel is static iff it does NOT belong to a *dynamic* concept
        instance. Background (``instance_seg == 0``) and static concepts such
        as ``road`` are kept. Delegates to
        :func:`~semantic_gs.data.static_mask.build_static_mask` with
        ``void_is_static=True`` (SAM 3 background is the rest of the static
        scene, not "uncertain").

        Parameters
        ----------
        id2label
            Concept-index → name map (``concepts.json``).
        dynamic_classes
            Concept names to remove. Defaults to
            :data:`DEFAULT_SAM3_DYNAMIC_CONCEPTS`. Pass ``frozenset()`` to keep
            everything.
        boundary_margin
            If > 0, also erode this many pixels around every instance boundary
            (stereo depth-bleeding fix), same as the Mask2Former path.
        """
        return build_static_mask(
            panoptic_seg     = self.instance_seg,
            segment_ids      = self.segment_ids,
            label_ids        = self.label_ids,
            id2label         = id2label,
            dynamic_classes  = (dynamic_classes if dynamic_classes is not None
                                else DEFAULT_SAM3_DYNAMIC_CONCEPTS),
            void_is_static   = True,
            boundary_margin  = boundary_margin,
        )

    def concept_mask(self, concept_name: str, id2label) -> np.ndarray:
        """Return a ``bool (H, W)`` mask of all instances of ``concept_name``.

        Unions every segment whose concept (resolved via ``id2label``) matches
        ``concept_name`` (case-insensitive). SAM 3 often splits a "stuff"
        concept like the road into several instances/lanes — they are all
        merged here, so lane-splitting does not matter.
        """
        target = concept_name.lower()
        matching_label_ids = {
            int(lid) for lid, name in id2label.items()
            if str(name).lower() == target
        }
        if not matching_label_ids:
            return np.zeros(self.instance_seg.shape, dtype=bool)
        seg_ids = [
            int(t) + 1 for t, l in zip(self.track_ids, self.label_ids)
            if int(l) in matching_label_ids
        ]
        if not seg_ids:
            return np.zeros(self.instance_seg.shape, dtype=bool)
        return np.isin(self.instance_seg, np.asarray(seg_ids, dtype=np.int32))

    def road_mask(self, id2label, road_names=DEFAULT_ROAD_CONCEPTS) -> np.ndarray:
        """Return a ``bool (H, W)`` mask of the drivable surface.

        ORs :meth:`concept_mask` over every name in ``road_names`` so it works
        regardless of which road phrasing was prompted.
        """
        mask = np.zeros(self.instance_seg.shape, dtype=bool)
        for name in road_names:
            mask |= self.concept_mask(name, id2label)
        return mask


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
    each ``label_ids`` value to its concept name (e.g. ``{0: "car", 3: "road"}``).
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
