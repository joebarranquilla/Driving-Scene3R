"""End-to-end test for :class:`KITTIOdomSequenceLoader`.

We materialise the existing dummy adapter's frames in the **exact** KITTI
layout the teammates produce on disk, then load them back through the
real KITTI loader and verify the round-trip preserves contracts and
geometry (intrinsics, baseline-shifted poses, static mask).
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.kitti_odom import KITTIOdomSequenceLoader
from semantic_gs.data.adapters.mock_teammate_outputs import (
    materialize_dummy_as_kitti_layout,
)
from semantic_gs.data.frame import Frame


# ---------------------------------------------------------------------------
# Fixture: a fully-populated KITTI-shaped directory tree
# ---------------------------------------------------------------------------

@pytest.fixture
def kitti_layout(tmp_path):
    """Build a 4-frame fixture and return ``(paths, dummy_loader)``."""
    dummy = DummySequenceLoader(num_frames=4)
    paths = materialize_dummy_as_kitti_layout(dummy, tmp_path, seq_id="07")
    return paths, dummy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loader_discovers_all_frames(kitti_layout):
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    assert len(loader) == len(dummy) == 4
    assert loader.name == "kitti_odom/07"


def test_loader_returns_valid_frames(kitti_layout):
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )

    for i in range(len(loader)):
        f = loader[i]
        assert isinstance(f, Frame)
        assert f.frame_id == f"{i:06d}"
        # Contracts re-validated by Frame.__post_init__; success means OK.


def test_loader_preserves_arrays_through_roundtrip(kitti_layout):
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    src = dummy[0]
    out = loader[0]

    np.testing.assert_array_equal(out.rgb,          src.rgb)
    np.testing.assert_array_equal(out.depth,        src.depth)
    np.testing.assert_array_equal(out.panoptic_seg, src.panoptic_seg)
    np.testing.assert_array_equal(out.segment_ids,  src.segment_ids)
    np.testing.assert_array_equal(out.label_ids,    src.label_ids)
    np.testing.assert_array_equal(out.scores,       src.scores)


def test_loader_intrinsics_match_dummy(kitti_layout):
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    cam_in  = dummy[0].camera
    cam_out = loader[0].camera
    for attr in ("width", "height", "fx", "fy", "cx", "cy"):
        assert getattr(cam_in, attr) == pytest.approx(getattr(cam_out, attr))


def test_loader_poses_match_dummy_after_baseline_shift(kitti_layout):
    """The mock writer encodes dummy poses (which are cam-2 poses) as cam-0
    poses on disk; the loader must reapply the cam-0→cam-2 shift and
    recover the original cam-2 poses exactly.
    """
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    for i in range(len(loader)):
        np.testing.assert_allclose(
            loader[i].T_cam_to_world,
            dummy[i].T_cam_to_world,
            atol=1e-9,
            err_msg=f"pose mismatch at frame {i}",
        )


def test_loader_static_mask_excludes_car_and_sky(kitti_layout):
    """KITTI loader defaults to unreliable_depth_classes={'sky'}, so on
    the dummy scene it must drop BOTH the car (dynamic) AND the sky band.
    """
    paths, dummy = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    f = loader[0]
    pano = f.panoptic_seg

    # No static pixel may belong to the car segment (id=4) or sky segment (id=1).
    assert not f.static_mask[pano == 4].any(), "car pixels still in static mask"
    assert not f.static_mask[pano == 1].any(), "sky pixels still in static mask"

    # The dummy loader (no sky-exclusion) keeps sky, so KITTI's mask must
    # be strictly smaller.
    assert f.static_mask.sum() < dummy[0].static_mask.sum()


def test_loader_static_mask_can_disable_sky_exclusion(kitti_layout):
    paths, _ = kitti_layout
    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
        unreliable_depth_classes=frozenset(),
    )
    f = loader[0]
    # Sky pixels (segment 1) are now allowed back into the static mask.
    assert f.static_mask[f.panoptic_seg == 1].all()


def test_loader_skips_incomplete_frames(kitti_layout, capsys):
    paths, _ = kitti_layout
    # Delete one of the depth files; loader should skip its frame.
    missing = paths.depth_dir / "000002.npz"
    missing.unlink()

    loader = KITTIOdomSequenceLoader(
        sequence_dir=paths.sequence_dir,
        depth_dir=paths.depth_dir,
        panoptic_dir=paths.panoptic_dir,
        pose_path=paths.pose_path,
    )
    assert len(loader) == 3
    assert "000002" not in [loader[i].frame_id for i in range(len(loader))]


def test_loader_raises_when_strict(kitti_layout):
    paths, _ = kitti_layout
    (paths.depth_dir / "000002.npz").unlink()
    with pytest.raises(FileNotFoundError):
        KITTIOdomSequenceLoader(
            sequence_dir=paths.sequence_dir,
            depth_dir=paths.depth_dir,
            panoptic_dir=paths.panoptic_dir,
            pose_path=paths.pose_path,
            skip_incomplete_frames=False,
        )


def test_loader_raises_on_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        KITTIOdomSequenceLoader(
            sequence_dir=tmp_path / "nope",
            depth_dir=tmp_path,
            panoptic_dir=tmp_path,
            pose_path=tmp_path / "nope.txt",
        )

