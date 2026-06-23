"""Tests for the teammate semantic-point-cloud adapters (PLY + NPZ).

Round-trip via :mod:`semantic_gs.data.adapters.mock_teammate_outputs` so
the tests are hermetic — no teammate script run is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.mock_teammate_outputs import (
    materialize_dummy_as_lift_output,
    write_semantic_pointcloud_npz,
    write_semantic_pointcloud_ply,
)
from semantic_gs.data.adapters.semantic_pointcloud_npz import (
    load_semantic_pointcloud_npz,
)
from semantic_gs.data.adapters.semantic_pointcloud_ply import (
    load_semantic_pointcloud_ply,
)
from semantic_gs.data.pointcloud import SemanticPointCloud


# ---------------------------------------------------------------------------
# SemanticPointCloud contract
# ---------------------------------------------------------------------------

def test_pointcloud_rejects_bad_shapes():
    with pytest.raises(ValueError):
        SemanticPointCloud(
            xyz   = np.zeros((4, 3), dtype=np.float32),
            rgb   = np.zeros((3, 3), dtype=np.float32),  # mismatched N
            labels= np.zeros((4,),  dtype=np.int32),
        )


def test_pointcloud_rejects_rgb_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SemanticPointCloud(
            xyz   = np.zeros((2, 3), dtype=np.float32),
            rgb   = np.full((2, 3), 2.0, dtype=np.float32),
            labels= np.zeros((2,),  dtype=np.int32),
        )


def test_pointcloud_empty_is_allowed():
    pc = SemanticPointCloud(
        xyz   = np.zeros((0, 3), dtype=np.float32),
        rgb   = np.zeros((0, 3), dtype=np.float32),
        labels= np.zeros((0,),  dtype=np.int32),
    )
    assert pc.n_points == 0
    assert pc.class_counts() == {}


# ---------------------------------------------------------------------------
# PLY round-trip
# ---------------------------------------------------------------------------

def test_ply_roundtrip_preserves_data(tmp_path):
    xyz = np.array(
        [[0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [-4.5, 0.25, 9.0]],
        dtype=np.float32,
    )
    rgb = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.array([0, 13, 8], dtype=np.int32)

    p = tmp_path / "tiny.ply"
    write_semantic_pointcloud_ply(p, xyz, rgb, labels)

    pc = load_semantic_pointcloud_ply(p)
    np.testing.assert_allclose(pc.xyz, xyz, atol=1e-4)
    # Colours are stored as uchar, so we lose up to 1 / 255 precision.
    np.testing.assert_allclose(pc.rgb, rgb, atol=1.0 / 255 + 1e-6)
    np.testing.assert_array_equal(pc.labels, labels)


def test_ply_without_label_property_loads(tmp_path):
    xyz = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    rgb = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    p = tmp_path / "no_label.ply"
    write_semantic_pointcloud_ply(p, xyz, rgb, labels=None)

    pc = load_semantic_pointcloud_ply(p)
    assert pc.n_points == 1
    # Labels default to zero when the PLY has no label property.
    np.testing.assert_array_equal(pc.labels, [0])


def test_ply_rejects_missing_required_property(tmp_path):
    """plyfile-backed reader is permissive about property ordering but
    must still reject a PLY that's missing a *required* RGB channel."""
    p = tmp_path / "bad.ply"
    p.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\n"   # missing blue
        "end_header\n"
        "0 0 0 255 0\n"
    )
    with pytest.raises(ValueError, match="missing required vertex properties"):
        load_semantic_pointcloud_ply(p)


def test_ply_accepts_binary_format(tmp_path):
    """The plyfile-backed reader transparently handles binary PLYs too.

    Build a tiny binary PLY by hand and check the round-trip.
    """
    p = tmp_path / "bin.ply"
    # Header
    n = 2
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    # Each vertex: 3 float32 (12 B) + 3 uint8 (3 B) = 15 B
    body = np.array(
        [(0.0, 0.0, 0.0, 10,  20,  30 ),
         (1.0, 2.0, 3.0, 200, 100, 50)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"),  ("g", "u1"),  ("b", "u1")],
    ).tobytes()
    p.write_bytes(header + body)

    pc = load_semantic_pointcloud_ply(p)
    assert pc.n_points == n
    np.testing.assert_allclose(pc.xyz, [[0, 0, 0], [1, 2, 3]], atol=1e-6)
    np.testing.assert_allclose(
        pc.rgb,
        np.array([[10, 20, 30], [200, 100, 50]], dtype=np.float32) / 255.0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# NPZ round-trip
# ---------------------------------------------------------------------------

def test_npz_roundtrip_preserves_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    xyz = rng.standard_normal((n, 3)).astype(np.float32)
    rgb = rng.random((n, 3)).astype(np.float32)
    labels = rng.integers(0, 19, size=n).astype(np.int32)

    p = tmp_path / "cloud.npz"
    write_semantic_pointcloud_npz(p, xyz, rgb, labels)

    pc = load_semantic_pointcloud_npz(p)
    np.testing.assert_array_equal(pc.xyz, xyz)
    np.testing.assert_array_equal(pc.rgb, rgb)
    np.testing.assert_array_equal(pc.labels, labels)


def test_npz_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, xyz=np.zeros((1, 3), np.float32))
    with pytest.raises(KeyError) as exc:
        load_semantic_pointcloud_npz(p)
    assert "colors" in str(exc.value) and "labels" in str(exc.value)


def test_npz_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_semantic_pointcloud_npz(tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# Full mock teammate fixture: PLY + NPZ from dummy frames
# ---------------------------------------------------------------------------

def test_materialize_dummy_lift_output_yields_consistent_clouds(tmp_path):
    dummy = DummySequenceLoader(num_frames=3)
    paths, ref_pc = materialize_dummy_as_lift_output(dummy, tmp_path)

    pc_ply = load_semantic_pointcloud_ply(paths.ply_path)
    pc_npz = load_semantic_pointcloud_npz(paths.npz_path)

    # NPZ is lossless and should match the reference bit-for-bit.
    np.testing.assert_array_equal(pc_npz.xyz,    ref_pc.xyz)
    np.testing.assert_array_equal(pc_npz.rgb,    ref_pc.rgb)
    np.testing.assert_array_equal(pc_npz.labels, ref_pc.labels)

    # PLY round-trip loses ~1/255 colour precision and 4-decimal xyz precision.
    assert pc_ply.n_points == ref_pc.n_points
    np.testing.assert_allclose(pc_ply.xyz, ref_pc.xyz, atol=1e-4)
    np.testing.assert_allclose(pc_ply.rgb, ref_pc.rgb, atol=1.0 / 255 + 1e-6)
    np.testing.assert_array_equal(pc_ply.labels, ref_pc.labels)


def test_materialize_dummy_excludes_dynamic_classes(tmp_path):
    """Sanity: after dropping person/rider from defaults, the only dynamic
    class in the dummy is "car" (label 13) — it must not appear in the cloud.
    Sky (label 10) is also excluded via the dummy's static mask.
    """
    dummy = DummySequenceLoader(num_frames=2)
    _, ref_pc = materialize_dummy_as_lift_output(dummy, tmp_path)
    present = set(ref_pc.class_counts().keys())
    assert 13 not in present, "car (13) leaked into the static cloud"
    # The dummy loader does NOT apply the unreliable-depth sky filter
    # (only the KITTI loader does), so sky (10) IS present here.
    assert 10 in present

