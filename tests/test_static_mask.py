"""Tests for ``semantic_gs.data.static_mask.build_static_mask``."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.static_mask import (
    DEFAULT_DYNAMIC_CLASSES,
    build_static_mask,
)


def test_filters_dynamic_segment_by_class_name():
    # Layout:
    #   row 0: segment 1 (road) | segment 1 (road) | segment 2 (car)
    #   row 1: segment 1 (road) | segment 3 (veg ) | segment 3 (veg)
    panoptic = np.array([[1, 1, 2],
                         [1, 3, 3]], dtype=np.int32)
    segment_ids = np.array([1, 2, 3], dtype=np.int32)
    label_ids   = np.array([0, 13, 8], dtype=np.int32)  # road, car, vegetation
    id2label    = {0: "road", 13: "car", 8: "vegetation"}

    mask = build_static_mask(panoptic, segment_ids, label_ids, id2label)

    expected = np.array([[True, True, False],
                         [True, True, True ]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_void_excluded_by_default_but_kept_when_requested():
    panoptic = np.array([[0, 1]], dtype=np.int32)
    seg_ids  = np.array([1], dtype=np.int32)
    lab_ids  = np.array([0], dtype=np.int32)
    id2label = {0: "road"}

    default_mask = build_static_mask(panoptic, seg_ids, lab_ids, id2label)
    assert default_mask.tolist() == [[False, True]]

    void_ok_mask = build_static_mask(
        panoptic, seg_ids, lab_ids, id2label, void_is_static=True
    )
    assert void_ok_mask.tolist() == [[True, True]]


def test_class_name_matching_is_case_insensitive():
    panoptic = np.array([[1]], dtype=np.int32)
    seg_ids  = np.array([1], dtype=np.int32)
    lab_ids  = np.array([13], dtype=np.int32)
    id2label = {13: "Car"}  # capital C — should still be detected as dynamic
    mask = build_static_mask(panoptic, seg_ids, lab_ids, id2label)
    assert mask.tolist() == [[False]]


def test_custom_dynamic_classes_override_defaults():
    panoptic = np.array([[1, 2]], dtype=np.int32)
    seg_ids  = np.array([1, 2], dtype=np.int32)
    lab_ids  = np.array([13, 8], dtype=np.int32)
    id2label = {13: "car", 8: "vegetation"}

    # Make vegetation dynamic, allow cars to stay.
    mask = build_static_mask(
        panoptic, seg_ids, lab_ids, id2label,
        dynamic_classes={"vegetation"},
    )
    assert mask.tolist() == [[True, False]]


def test_no_segments_returns_all_non_void_static():
    panoptic = np.array([[0, 1, 1]], dtype=np.int32)
    mask = build_static_mask(
        panoptic,
        segment_ids=np.array([], dtype=np.int32),
        label_ids=np.array([], dtype=np.int32),
        id2label={},
    )
    assert mask.tolist() == [[False, True, True]]


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        build_static_mask(
            panoptic_seg=np.zeros((2, 2, 2), dtype=np.int32),
            segment_ids=np.array([], dtype=np.int32),
            label_ids=np.array([], dtype=np.int32),
            id2label={},
        )
    with pytest.raises(ValueError):
        build_static_mask(
            panoptic_seg=np.zeros((2, 2), dtype=np.int32),
            segment_ids=np.array([1, 2], dtype=np.int32),
            label_ids=np.array([1], dtype=np.int32),  # length mismatch
            id2label={},
        )


def test_default_dynamic_classes_contents():
    # Vehicles MUST stay dynamic.
    assert "car" in DEFAULT_DYNAMIC_CLASSES
    assert "truck" in DEFAULT_DYNAMIC_CLASSES
    assert "bus" in DEFAULT_DYNAMIC_CLASSES
    assert "bicycle" in DEFAULT_DYNAMIC_CLASSES
    # Person and rider are intentionally NOT dynamic by default
    # (matches scripts/lift_to_semantic_pointcloud.py policy — see
    # semantic_gs/data/static_mask.py for rationale).
    assert "person" not in DEFAULT_DYNAMIC_CLASSES
    assert "rider" not in DEFAULT_DYNAMIC_CLASSES
    # Static stuff classes must never be in here.
    assert "road" not in DEFAULT_DYNAMIC_CLASSES
    assert "building" not in DEFAULT_DYNAMIC_CLASSES


def test_person_pixels_now_kept_static_by_default():
    """Regression guard for the Phase-2 policy change."""
    panoptic = np.array([[1, 2]], dtype=np.int32)
    seg_ids  = np.array([1, 2], dtype=np.int32)
    lab_ids  = np.array([11, 13], dtype=np.int32)  # person, car
    id2label = {11: "person", 13: "car"}

    mask = build_static_mask(panoptic, seg_ids, lab_ids, id2label)
    # person stays static (True), car is dropped (False).
    assert mask.tolist() == [[True, False]]


def test_boundary_margin_removes_edge_pixels():
    """A 1-pixel margin around every class boundary must be dropped."""
    # 6x4 frame split: rows 0-2 = road (id 0), rows 3-5 = building (id 2).
    # 4-connected boundary lives at rows {2, 3}; with margin=1 the dilation
    # also pulls in rows {1, 4}, leaving only rows {0, 5} static.
    panoptic = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ], dtype=np.int32)
    seg_ids  = np.array([1, 2], dtype=np.int32)
    lab_ids  = np.array([0, 2], dtype=np.int32)
    id2label = {0: "road", 2: "building"}

    without = build_static_mask(panoptic, seg_ids, lab_ids, id2label)
    with_eroded = build_static_mask(
        panoptic, seg_ids, lab_ids, id2label, boundary_margin=1,
    )

    # Without erosion: every pixel is static.
    assert without.sum() == 24
    # With margin=1: only the two rows furthest from the boundary survive.
    assert with_eroded.sum() == 8
    assert with_eroded[0, :].all()
    assert with_eroded[5, :].all()
    assert not with_eroded[1:5, :].any()



