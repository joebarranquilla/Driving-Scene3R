"""Tests for :mod:`semantic_gs.scripts.extract_track_trajectories`.

Fixture design (reuses ``materialize_dummy_as_kitti_layout`` like the SAM 3
loader test, then overwrites the depth NPZs with controlled values):

- dummy poses translate the camera +1 m in +Z per frame (identity rotation);
- "moving" car (track 0): same pixel block + constant depth every frame
  → its world position rides along with the ego at 1 m/frame (+Z);
- "parked" car (track 1): same pixel block, depth shrinking 1 m/frame
  → its world position is constant;
- "short" car (track 9): present in only 2 frames → filtered out;
- road segment (track 7): wrong concept → ignored entirely.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.mock_teammate_outputs import (
    materialize_dummy_as_kitti_layout,
    write_depth_npz,
)
from semantic_gs.scripts.extract_track_trajectories import (
    TrackObservations,
    _finalize_track,
    _smooth,
    main,
)

CONCEPTS = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "road"}
N_FRAMES = 6
DEPTH_M = 10.0

MOVING_BLOCK = (slice(40, 60), slice(200, 240))   # track 0 → seg id 1
SHORT_BLOCK  = (slice(70, 80), slice(280, 310))   # track 9 → seg id 10


@pytest.fixture
def traj_layout(tmp_path):
    dummy = DummySequenceLoader(num_frames=N_FRAMES)
    kitti = materialize_dummy_as_kitti_layout(dummy, tmp_path, seq_id="07")
    cam = dummy[0].camera
    H, W = cam.height, cam.width

    # The parked car keeps the SAME pixels while its depth shrinks with the
    # approaching ego. That is only world-consistent for pixels on the optical
    # axis (X_cam = (u - cx) * Z / fx varies with Z otherwise), so the block
    # is centered on cx. track 1 → seg id 2.
    cx = int(round(cam.cx))
    parked_block = (slice(40, 60), slice(cx - 20, cx + 20))

    sam3_dir = tmp_path / "sam3" / "07"
    sam3_dir.mkdir(parents=True)
    (tmp_path / "sam3" / "concepts.json").write_text(
        json.dumps({str(k): v for k, v in CONCEPTS.items()})
    )

    for i in range(N_FRAMES):
        # Depth: constant 10 m, except the parked car's block approaches the
        # camera exactly as fast as the ego drives forward (world-static).
        depth = np.full((H, W), DEPTH_M, dtype=np.float32)
        depth[parked_block] = DEPTH_M - i * 1.0
        write_depth_npz(kitti.depth_dir / f"{i:06d}.npz", depth)

        inst = np.zeros((H, W), dtype=np.int32)
        inst[MOVING_BLOCK] = 1                    # track 0 (car)
        inst[parked_block] = 2                    # track 1 (car)
        inst[0, 0] = 8                            # track 7 (road)
        track_ids = [0, 1, 7]
        label_ids = [0, 0, 3]
        if i < 2:
            inst[SHORT_BLOCK] = 10                # track 9 (car), 2 frames only
            track_ids.append(9)
            label_ids.append(0)
        np.savez_compressed(
            sam3_dir / f"{i:06d}.npz",
            instance_seg=inst,
            track_ids=np.asarray(track_ids, dtype=np.int32),
            label_ids=np.asarray(label_ids, dtype=np.int32),
            scores=np.full(len(track_ids), 0.9, dtype=np.float32),
        )

    out_dir = tmp_path / "trajectories"
    argv = [
        "extract_track_trajectories",
        "--kitti-odom-seq", str(kitti.sequence_dir),
        "--depth-dir",      str(kitti.depth_dir),
        "--sam3-dir",       str(sam3_dir),
        "--pose-path",      str(kitti.pose_path),
        "--out",            str(out_dir),
        "--min-track-frames", "4",
        "--min-pixels",       "10",
        "--boundary-margin",  "0",
        "--smooth-window",    "3",
        "--no-plot",
    ]
    return argv, out_dir


# ---------------------------------------------------------------------------
# End-to-end via main()
# ---------------------------------------------------------------------------

@pytest.fixture
def traj_results(traj_layout, monkeypatch):
    argv, out_dir = traj_layout
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    summary = json.loads((out_dir / "tracks_summary.json").read_text())
    return out_dir, summary


def test_short_and_offconcept_tracks_filtered(traj_results):
    out_dir, summary = traj_results
    assert summary["n_tracks_kept"] == 2
    kept_ids = {t["track_id"] for t in summary["tracks"]}
    assert kept_ids == {0, 1}                      # not 9 (short), not 7 (road)
    assert not (out_dir / "track_009_trajectory.json").exists()


def test_moving_track_rides_with_ego(traj_results):
    out_dir, _ = traj_results
    tr = json.loads((out_dir / "track_000_trajectory.json").read_text())
    assert tr["is_moving"] is True
    z = tr["trajectory"]["z"]
    x = tr["trajectory"]["x"]
    # +1 m per frame in +Z (smoothing shrinks the endpoints, interior exact).
    assert z[3] - z[2] == pytest.approx(1.0, abs=0.05)
    assert max(x) - min(x) < 0.05                  # no lateral drift
    # Heading along +Z → theta == atan2(0, +dz) == 0; speed 1 m / 0.1 s.
    assert tr["trajectory"]["theta"][2] == pytest.approx(0.0, abs=0.02)
    assert tr["trajectory"]["v"][2] == pytest.approx(10.0, rel=0.05)


def test_parked_track_is_static_in_world(traj_results):
    out_dir, _ = traj_results
    tr = json.loads((out_dir / "track_001_trajectory.json").read_text())
    assert tr["is_moving"] is False
    z = np.asarray(tr["trajectory"]["z"])
    assert np.ptp(z) < 0.1                         # world-static despite ego motion
    assert tr["net_displacement_m"] < 0.1


def test_json_schema_matches_consumer_contract(traj_results):
    """Ece's place_object_from_trajectory_json requires trajectory.{x,y,z,theta}."""
    out_dir, _ = traj_results
    tr = json.loads((out_dir / "track_000_trajectory.json").read_text())
    traj = tr["trajectory"]
    for key in ("x", "y", "z", "theta"):
        assert key in traj and traj[key] is not None
    n = tr["n_steps"]
    assert {len(traj[k]) for k in ("x", "y", "z", "theta", "v")} == {n}
    assert all(np.isfinite(traj[k]).all() for k in ("x", "y", "z", "theta"))


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------

def test_smooth_preserves_length_and_constants():
    a = np.full(9, 3.5)
    out = _smooth(a, 5)
    assert out.shape == a.shape
    np.testing.assert_allclose(out, a)
    assert _smooth(np.array([1.0, 2.0]), 7).shape == (2,)   # tiny input untouched


def test_finalize_track_theta_for_lateral_motion():
    """Motion in +X must give theta ≈ +π/2 (yaw measured from +Z toward +X)."""
    obs = TrackObservations(track_id=4, concept="car")
    for i in range(5):
        obs.frame_indices.append(i)
        obs.frame_ids.append(f"{i:06d}")
        obs.centroids.append(np.array([float(i), -1.0, 20.0]))
        obs.n_pixels.append(100)
    tr = _finalize_track(obs, smooth_window=1, moving_threshold=2.0)
    assert tr["is_moving"] is True
    # theta is rounded to 5 decimals in the JSON payload.
    assert tr["trajectory"]["theta"][1] == pytest.approx(np.pi / 2, abs=1e-4)
    assert tr["trajectory"]["v"][1] == pytest.approx(10.0, rel=1e-6)
    assert tr["path_length_m"] == pytest.approx(4.0, abs=1e-3)
