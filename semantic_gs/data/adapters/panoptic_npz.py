"""Strict reader for the teammate's panoptic NPZ files + ``id2label.json``.

Contract (from ``scripts/run_mask2former_inference.py``):
    {output_dir}/id2label.json   -> {"0": "road", "2": "building", ...}
    {output_dir}/{sequence}/{frame_stem}.npz
        "panoptic_seg" -> int32  (H, W)   (0 = void)
        "segment_ids"  -> int32  (N,)
        "label_ids"    -> int32  (N,)
        "scores"       -> float32 (N,)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PANOPTIC_KEYS = ("panoptic_seg", "segment_ids", "label_ids", "scores")


@dataclass(frozen=True)
class PanopticPrediction:
    """One frame's panoptic prediction loaded from disk."""

    panoptic_seg: np.ndarray   # int32 (H, W)
    segment_ids:  np.ndarray   # int32 (N,)
    label_ids:    np.ndarray   # int32 (N,)
    scores:       np.ndarray   # float32 (N,)


def load_panoptic_npz(path: str | Path) -> PanopticPrediction:
    """Load a teammate panoptic NPZ as a :class:`PanopticPrediction`.

    Casts ``panoptic_seg`` / ``segment_ids`` / ``label_ids`` to ``int32``
    and ``scores`` to ``float32`` for a stable contract downstream.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"panoptic NPZ not found: {path}")

    with np.load(path) as data:
        missing = [k for k in PANOPTIC_KEYS if k not in data.files]
        if missing:
            raise KeyError(
                f"{path}: missing keys {missing}; found {data.files}"
            )
        panoptic_seg = np.asarray(data["panoptic_seg"])
        segment_ids  = np.asarray(data["segment_ids"])
        label_ids    = np.asarray(data["label_ids"])
        scores       = np.asarray(data["scores"])

    if panoptic_seg.ndim != 2:
        raise ValueError(
            f"{path}: panoptic_seg must be (H, W); got {panoptic_seg.shape}"
        )
    if not (segment_ids.shape == label_ids.shape == scores.shape and segment_ids.ndim == 1):
        raise ValueError(
            f"{path}: segment_ids/label_ids/scores must be 1-D and same length; "
            f"got {segment_ids.shape}, {label_ids.shape}, {scores.shape}"
        )

    return PanopticPrediction(
        panoptic_seg=panoptic_seg.astype(np.int32, copy=False),
        segment_ids =segment_ids.astype(np.int32, copy=False),
        label_ids   =label_ids.astype(np.int32, copy=False),
        scores      =scores.astype(np.float32, copy=False),
    )


def load_id2label_json(path: str | Path) -> dict[int, str]:
    """Load the teammate's ``id2label.json`` (string-keyed) and return
    a ``dict[int, str]``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"id2label.json not found: {path}")
    with path.open("r") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: keys must be integer-parseable; {exc}") from exc

