"""Tests for ``semantic_gs.data.adapters.dummy.DummySequenceLoader``."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.frame import Frame


def test_length_and_iteration():
    loader = DummySequenceLoader(num_frames=5)
    assert len(loader) == 5
    frames = list(loader)
    assert len(frames) == 5
    assert all(isinstance(f, Frame) for f in frames)


def test_default_frame_contract():
    loader = DummySequenceLoader(num_frames=2)
    f = loader[0]

    H, W = f.camera.height, f.camera.width
    assert f.rgb.shape          == (H, W, 3) and f.rgb.dtype          == np.uint8
    assert f.depth.shape        == (H, W)    and f.depth.dtype        == np.float32
    assert f.panoptic_seg.shape == (H, W)    and f.panoptic_seg.dtype == np.int32
    assert f.static_mask.shape  == (H, W)    and f.static_mask.dtype  == np.bool_

    assert f.segment_ids.dtype == np.int32
    assert f.label_ids.dtype   == np.int32
    assert f.scores.dtype      == np.float32
    assert f.segment_ids.shape == f.label_ids.shape == f.scores.shape

    assert f.T_cam_to_world.shape == (4, 4)
    assert f.T_cam_to_world.dtype == np.float64
    np.testing.assert_allclose(f.T_cam_to_world[3], [0, 0, 0, 1])


def test_small_size_override():
    loader = DummySequenceLoader(num_frames=2, width=64, height=32)
    f = loader[0]
    assert f.rgb.shape == (32, 64, 3)
    assert f.camera.width == 64 and f.camera.height == 32


def test_static_mask_excludes_car_region():
    loader = DummySequenceLoader(num_frames=1)
    f = loader[0]
    y0, x0, y1, x1 = loader.car_bbox

    # The static mask must drop *every* pixel of the car bbox.
    assert not f.static_mask[y0:y1, x0:x1].any(), \
        "Static mask still contains dynamic-car pixels."

    # And it must keep most road/building/sky pixels.
    assert f.static_mask.mean() > 0.5, \
        f"Static mask is too sparse ({f.static_mask.mean():.3f})"


def test_static_mask_label_ids_match_id2label():
    loader = DummySequenceLoader(num_frames=1)
    f = loader[0]
    # Every label ID present must be resolvable to a class name.
    for lid in f.label_ids.tolist():
        assert int(lid) in loader.id2label, f"label_id {lid} missing in id2label"


def test_poses_advance_strictly_forward():
    step = 0.7
    loader = DummySequenceLoader(num_frames=4, forward_step_m=step)
    zs = [loader[i].T_cam_to_world[2, 3] for i in range(len(loader))]
    diffs = np.diff(zs)
    np.testing.assert_allclose(diffs, step, atol=1e-9)
    # Rotations are identity in the dummy world.
    for i in range(len(loader)):
        np.testing.assert_allclose(loader[i].T_cam_to_world[:3, :3], np.eye(3))


def test_depth_is_non_negative_and_finite_where_static():
    loader = DummySequenceLoader(num_frames=1)
    f = loader[0]
    assert np.all(f.depth >= 0.0)
    # Wherever static_mask is True the depth value should be a finite,
    # positive metric measurement (i.e. usable for back-projection).
    static_depths = f.depth[f.static_mask]
    assert np.all(np.isfinite(static_depths))
    assert np.all(static_depths > 0.0)


def test_out_of_range_index_raises():
    loader = DummySequenceLoader(num_frames=3)
    with pytest.raises(IndexError):
        _ = loader[3]
    with pytest.raises(IndexError):
        _ = loader[-1]


def test_id2label_is_a_copy():
    loader = DummySequenceLoader(num_frames=1)
    m1 = loader.id2label
    # Verify the loader hands back a concrete (mutable) dict that is a
    # defensive copy, so callers cannot poison the loader's internal map.
    assert isinstance(m1, dict), "expected concrete dict, got read-only Mapping"
    m1[999] = "tamper"  # type: ignore[index]
    assert 999 not in loader.id2label



