"""End-to-end test for :class:`KITTISam3SequenceLoader`.

Reuses ``materialize_dummy_as_kitti_layout`` for the shared KITTI parts
(calib, poses, RGB, depth), then fabricates SAM 3 instance NPZs + a
``concepts.json`` alongside, and loads everything back through the SAM 3
loader. Verifies the SAM 3 → ``Frame`` contract and the
``instance_seg == 0`` static-mask rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.kitti_odom_sam3 import KITTISam3SequenceLoader
from semantic_gs.data.adapters.mock_teammate_outputs import (
    materialize_dummy_as_kitti_layout,
)
from semantic_gs.data.frame import Frame


CONCEPTS = {0: "car", 1: "pedestrian", 2: "cyclist"}


def _write_sam3_frame(path: Path, H: int, W: int):
    """Fabricate a SAM 3 NPZ with two instances: track 0 (car), track 3 (cyclist).

    instance_seg stores track_id + 1, so track 0 -> 1, track 3 -> 4.
    Returns the arrays written for assertions.
    """
    instance_seg = np.zeros((H, W), dtype=np.int32)
    instance_seg[1, 1] = 1          # track 0
    instance_seg[H - 1, W - 1] = 4  # track 3
    track_ids = np.array([0, 3], dtype=np.int32)
    label_ids = np.array([0, 2], dtype=np.int32)   # car, cyclist
    scores    = np.array([0.9, 0.8], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        instance_seg=instance_seg,
        track_ids=track_ids,
        label_ids=label_ids,
        scores=scores,
    )
    return instance_seg, track_ids, label_ids, scores


@pytest.fixture
def sam3_layout(tmp_path):
    """Build a 4-frame KITTI+SAM3 fixture; return (paths_dict, dummy)."""
    dummy = DummySequenceLoader(num_frames=4)
    kitti = materialize_dummy_as_kitti_layout(dummy, tmp_path, seq_id="07")

    cam = dummy[0].camera
    H, W = cam.height, cam.width

    sam3_dir = tmp_path / "sam3" / "07"
    concepts_path = tmp_path / "sam3" / "concepts.json"
    for i in range(len(dummy)):
        _write_sam3_frame(sam3_dir / f"{i:06d}.npz", H, W)
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_path.write_text(json.dumps({str(k): v for k, v in CONCEPTS.items()}))

    paths = {
        "sequence_dir": kitti.sequence_dir,
        "depth_dir":    kitti.depth_dir,
        "sam3_dir":     sam3_dir,
        "pose_path":    kitti.pose_path,
    }
    return paths, dummy


# ---------------------------------------------------------------------------
# Discovery / contract
# ---------------------------------------------------------------------------

def test_loader_discovers_all_frames(sam3_layout):
    paths, dummy = sam3_layout
    loader = KITTISam3SequenceLoader(**paths)
    assert len(loader) == len(dummy) == 4
    assert loader.name == "kitti_odom_sam3/07"
    assert loader.id2label == CONCEPTS


def test_loader_returns_valid_frames(sam3_layout):
    paths, _ = sam3_layout
    loader = KITTISam3SequenceLoader(**paths)
    for i in range(len(loader)):
        f = loader[i]
        assert isinstance(f, Frame)          # Frame.__post_init__ validates contracts
        assert f.frame_id == f"{i:06d}"


def test_segment_ids_are_track_ids_plus_one(sam3_layout):
    paths, _ = sam3_layout
    loader = KITTISam3SequenceLoader(**paths)
    f = loader[0]
    np.testing.assert_array_equal(f.segment_ids, np.array([1, 4], dtype=np.int32))
    np.testing.assert_array_equal(f.label_ids, np.array([0, 2], dtype=np.int32))
    # segment_ids must be exactly the non-zero values present in panoptic_seg.
    present = set(np.unique(f.panoptic_seg).tolist()) - {0}
    assert set(f.segment_ids.tolist()) == present


# ---------------------------------------------------------------------------
# Static mask
# ---------------------------------------------------------------------------

def test_static_mask_excludes_every_instance(sam3_layout):
    paths, _ = sam3_layout
    loader = KITTISam3SequenceLoader(**paths)
    f = loader[0]
    # Every instance pixel dropped; every background pixel kept.
    np.testing.assert_array_equal(f.static_mask, f.panoptic_seg == 0)
    assert not f.static_mask[f.panoptic_seg > 0].any()


def test_boundary_margin_erodes_more(sam3_layout):
    paths, _ = sam3_layout
    base = KITTISam3SequenceLoader(**paths)[0]
    eroded = KITTISam3SequenceLoader(**paths, boundary_margin=1)[0]
    assert eroded.static_mask.sum() <= base.static_mask.sum()
    assert not np.any(eroded.static_mask & ~base.static_mask)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_loader_skips_incomplete_frames(sam3_layout):
    paths, _ = sam3_layout
    (paths["sam3_dir"] / "000002.npz").unlink()
    loader = KITTISam3SequenceLoader(**paths)
    assert len(loader) == 3
    assert "000002" not in [loader[i].frame_id for i in range(len(loader))]


def test_loader_raises_when_strict(sam3_layout):
    paths, _ = sam3_layout
    (paths["sam3_dir"] / "000002.npz").unlink()
    with pytest.raises(FileNotFoundError):
        KITTISam3SequenceLoader(**paths, skip_incomplete_frames=False)


def test_loader_raises_on_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        KITTISam3SequenceLoader(
            sequence_dir=tmp_path / "nope",
            depth_dir=tmp_path,
            sam3_dir=tmp_path,
            pose_path=tmp_path / "nope.txt",
        )


def test_concepts_default_path_resolution(sam3_layout):
    """concepts.json defaults to sam3_dir.parent / concepts.json."""
    paths, _ = sam3_layout
    # Not passing concepts_path → must auto-resolve to the sibling file.
    loader = KITTISam3SequenceLoader(**paths)
    assert loader.id2label == CONCEPTS
