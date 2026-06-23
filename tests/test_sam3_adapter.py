"""Tests for the SAM 3 instance-NPZ adapter (reader + static mask + concepts)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_gs.data.adapters.sam3_npz import (
    SAM3_KEYS,
    Sam3Prediction,
    load_concepts_json,
    load_sam3_npz,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_sam3_npz(path, instance_seg, track_ids, label_ids, scores):
    np.savez_compressed(
        path,
        instance_seg=instance_seg.astype(np.int32),
        track_ids=track_ids.astype(np.int32),
        label_ids=label_ids.astype(np.int32),
        scores=scores.astype(np.float32),
    )


def _toy_frame():
    """A 4x5 frame with two instances: track 0 (car) and track 4 (car).

    instance_seg stores track_id + 1, so track 0 -> 1, track 4 -> 5.
    """
    instance_seg = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 5],
            [0, 0, 0, 0, 5],
        ],
        dtype=np.int32,
    )
    track_ids = np.array([0, 4], dtype=np.int32)
    label_ids = np.array([0, 0], dtype=np.int32)
    scores = np.array([0.84, 0.95], dtype=np.float32)
    return instance_seg, track_ids, label_ids, scores


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_sam3_npz_roundtrip(tmp_path):
    instance_seg, track_ids, label_ids, scores = _toy_frame()
    p = tmp_path / "000000.npz"
    _write_sam3_npz(p, instance_seg, track_ids, label_ids, scores)

    pred = load_sam3_npz(p)
    assert isinstance(pred, Sam3Prediction)
    assert pred.instance_seg.dtype == np.int32
    assert pred.track_ids.dtype == np.int32
    assert pred.label_ids.dtype == np.int32
    assert pred.scores.dtype == np.float32
    np.testing.assert_array_equal(pred.instance_seg, instance_seg)
    np.testing.assert_array_equal(pred.track_ids, track_ids)


def test_sam3_npz_casts_dtypes(tmp_path):
    p = tmp_path / "000001.npz"
    np.savez_compressed(
        p,
        instance_seg=np.zeros((3, 3), dtype=np.int64),
        track_ids=np.array([0], dtype=np.int64),
        label_ids=np.array([1], dtype=np.int64),
        scores=np.array([0.5], dtype=np.float64),
    )
    pred = load_sam3_npz(p)
    assert pred.instance_seg.dtype == np.int32
    assert pred.track_ids.dtype == np.int32
    assert pred.label_ids.dtype == np.int32
    assert pred.scores.dtype == np.float32


def test_sam3_npz_rejects_missing_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, instance_seg=np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(KeyError, match="missing"):
        load_sam3_npz(p)


def test_sam3_npz_rejects_non_2d_instance_seg(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(
        p,
        instance_seg=np.zeros((2, 2, 2), dtype=np.int32),
        track_ids=np.array([0], dtype=np.int32),
        label_ids=np.array([0], dtype=np.int32),
        scores=np.array([0.5], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="must be"):
        load_sam3_npz(p)


def test_sam3_npz_rejects_mismatched_lengths(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(
        p,
        instance_seg=np.zeros((2, 2), dtype=np.int32),
        track_ids=np.array([0, 1], dtype=np.int32),
        label_ids=np.array([0], dtype=np.int32),     # wrong length
        scores=np.array([0.5, 0.5], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="same length"):
        load_sam3_npz(p)


def test_sam3_keys_constant():
    assert SAM3_KEYS == ("instance_seg", "track_ids", "label_ids", "scores")


# ---------------------------------------------------------------------------
# Frame-contract mapping
# ---------------------------------------------------------------------------

def test_segment_ids_offset_matches_instance_seg(tmp_path):
    instance_seg, track_ids, label_ids, scores = _toy_frame()
    p = tmp_path / "000000.npz"
    _write_sam3_npz(p, instance_seg, track_ids, label_ids, scores)
    pred = load_sam3_npz(p)

    # segment_ids must be the values you actually find in instance_seg.
    np.testing.assert_array_equal(pred.segment_ids, np.array([1, 5], dtype=np.int32))
    present = set(np.unique(pred.instance_seg).tolist()) - {0}
    assert set(pred.segment_ids.tolist()) == present
    assert pred.segment_ids.dtype == np.int32


# ---------------------------------------------------------------------------
# Static mask
# ---------------------------------------------------------------------------

def test_static_mask_excludes_all_instances(tmp_path):
    instance_seg, track_ids, label_ids, scores = _toy_frame()
    p = tmp_path / "000000.npz"
    _write_sam3_npz(p, instance_seg, track_ids, label_ids, scores)
    pred = load_sam3_npz(p)

    mask = pred.static_mask()
    assert mask.dtype == np.bool_
    assert mask.shape == instance_seg.shape
    # Every instance pixel is removed; every background pixel is kept.
    np.testing.assert_array_equal(mask, instance_seg == 0)


def test_static_mask_boundary_margin_erodes_more(tmp_path):
    instance_seg, track_ids, label_ids, scores = _toy_frame()
    p = tmp_path / "000000.npz"
    _write_sam3_npz(p, instance_seg, track_ids, label_ids, scores)
    pred = load_sam3_npz(p)

    base = pred.static_mask(boundary_margin=0)
    eroded = pred.static_mask(boundary_margin=1)
    # Boundary erosion can only remove static pixels, never add them.
    assert eroded.sum() <= base.sum()
    assert not np.any(eroded & ~base)


# ---------------------------------------------------------------------------
# concepts.json
# ---------------------------------------------------------------------------

def test_load_concepts_json(tmp_path):
    p = tmp_path / "concepts.json"
    p.write_text(json.dumps({"0": "car", "1": "pedestrian", "2": "cyclist"}))
    concepts = load_concepts_json(p)
    assert concepts == {0: "car", 1: "pedestrian", 2: "cyclist"}
    assert all(isinstance(k, int) for k in concepts)


def test_load_concepts_json_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_concepts_json(tmp_path / "nope.json")
