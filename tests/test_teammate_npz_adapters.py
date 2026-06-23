"""Tests for the teammate depth-NPZ adapter."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.adapters.depth_npz import load_depth_npz
from semantic_gs.data.adapters.mock_teammate_outputs import write_depth_npz


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
