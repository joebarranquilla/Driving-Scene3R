"""Tests for the teammate-NPZ adapters (depth + panoptic + id2label)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_gs.data.adapters.depth_npz import load_depth_npz
from semantic_gs.data.adapters.mock_teammate_outputs import (
    write_depth_npz,
    write_id2label_json,
    write_panoptic_npz,
)
from semantic_gs.data.adapters.panoptic_npz import (
    PANOPTIC_KEYS,
    PanopticPrediction,
    load_id2label_json,
    load_panoptic_npz,
)


# ---------------------------------------------------------------------------
# Depth NPZ
# ---------------------------------------------------------------------------

def test_depth_npz_roundtrip(tmp_path):
    depth = np.linspace(2.0, 60.0, 24, dtype=np.float32).reshape(4, 6)
    p = tmp_path / "000000.npz"
    write_depth_npz(p, depth)
    out = load_depth_npz(p)
    assert out.shape == (4, 6)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, depth)


def test_depth_npz_casts_to_float32(tmp_path):
    depth_f64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p = tmp_path / "000001.npz"
    np.savez_compressed(p, depth=depth_f64)
    out = load_depth_npz(p)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, depth_f64)


def test_depth_npz_rejects_missing_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, disparity=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="'depth'"):
        load_depth_npz(p)


def test_depth_npz_rejects_bad_shape(tmp_path):
    p = tmp_path / "bad.npz"
    write_depth_npz(p, np.zeros((2,), dtype=np.float32).reshape(2, 1))
    # Force a 3-D array via direct savez_compressed (bypassing the writer's
    # own shape guard) — the loader must catch it.
    np.savez_compressed(p, depth=np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        load_depth_npz(p)


def test_depth_npz_rejects_nan_only(tmp_path):
    p = tmp_path / "nan.npz"
    np.savez_compressed(p, depth=np.full((3, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="no finite values"):
        load_depth_npz(p)


def test_depth_npz_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_depth_npz(tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# Panoptic NPZ
# ---------------------------------------------------------------------------

def _make_panoptic_arrays():
    panoptic_seg = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    segment_ids  = np.array([1, 2], dtype=np.int32)
    label_ids    = np.array([0, 13], dtype=np.int32)
    scores       = np.array([0.95, 0.80], dtype=np.float32)
    return panoptic_seg, segment_ids, label_ids, scores


def test_panoptic_npz_roundtrip(tmp_path):
    pano, sids, lids, scores = _make_panoptic_arrays()
    p = tmp_path / "000000.npz"
    write_panoptic_npz(p, pano, sids, lids, scores)

    out = load_panoptic_npz(p)
    assert isinstance(out, PanopticPrediction)
    np.testing.assert_array_equal(out.panoptic_seg, pano)
    np.testing.assert_array_equal(out.segment_ids,  sids)
    np.testing.assert_array_equal(out.label_ids,    lids)
    np.testing.assert_array_equal(out.scores,       scores)
    assert out.panoptic_seg.dtype == np.int32
    assert out.scores.dtype       == np.float32


def test_panoptic_npz_rejects_missing_keys(tmp_path):
    p = tmp_path / "broken.npz"
    np.savez_compressed(p, panoptic_seg=np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(KeyError) as exc_info:
        load_panoptic_npz(p)
    msg = str(exc_info.value)
    for missing in PANOPTIC_KEYS[1:]:
        assert missing in msg


def test_panoptic_npz_rejects_shape_mismatch(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(
        p,
        panoptic_seg=np.zeros((2, 2), dtype=np.int32),
        segment_ids =np.array([1, 2], dtype=np.int32),
        label_ids   =np.array([0], dtype=np.int32),      # length mismatch
        scores      =np.array([0.5, 0.5], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="same length"):
        load_panoptic_npz(p)


# ---------------------------------------------------------------------------
# id2label.json
# ---------------------------------------------------------------------------

def test_id2label_roundtrip(tmp_path):
    mapping = {0: "road", 2: "building", 13: "car"}
    p = tmp_path / "id2label.json"
    write_id2label_json(p, mapping)

    # File on disk must use string keys (teammate-script convention).
    with p.open("r") as fh:
        raw = json.load(fh)
    assert all(isinstance(k, str) for k in raw)

    out = load_id2label_json(p)
    assert out == mapping
    assert all(isinstance(k, int) for k in out)


def test_id2label_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not_an_int": "road"}))
    with pytest.raises(ValueError, match="integer"):
        load_id2label_json(p)

