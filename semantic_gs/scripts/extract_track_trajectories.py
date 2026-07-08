"""Extract observed 3D trajectories of SAM 3 tracks (the SAM 3 downstream task).

For every persistent SAM 3 ``track_id`` of a dynamic concept (default: car),
unproject its masked depth pixels per frame into the KITTI world frame and
reduce them to a robust per-frame centroid. Stringing the centroids together
gives the *observed* trajectory of each real agent — something Mask2Former
cannot provide (no identity across frames).

Output JSONs use the same schema as ``plan_trajectory.py``'s
``optimal_trajectory.json`` (teammate branch ``drivable-road``), so they are
drop-in inputs for ``combine.py``'s ``place_object_from_trajectory_json``
(teammate branch ``dynamic``): a generated car asset can be placed into the
trained splat world at any step of a *real* car's path, with matching heading.

Example
-------
    python -m semantic_gs.scripts.extract_track_trajectories \\
        --kitti-odom-seq /storage/.../sequences/04 \\
        --depth-dir      /usr/prakt/<u>/depth_predictions/04 \\
        --sam3-dir       /usr/prakt/<u>/sam3_predictions_road/04 \\
        --pose-path      /storage/.../sequences/04/04.txt \\
        --out            /usr/prakt/<u>/track_trajectories/04
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semantic_gs.data.adapters.kitti_odom_sam3 import KITTISam3SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.data.static_mask import semantic_boundary_mask
from semantic_gs.geometry.cameras import unproject_pixels

KITTI_FPS = 10.0  # KITTI odometry is captured at 10 Hz


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-track observed 3D trajectories from SAM 3 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kitti-odom-seq", required=True, metavar="SEQ_DIR")
    p.add_argument("--depth-dir",      required=True, metavar="DIR")
    p.add_argument("--sam3-dir",       required=True, metavar="DIR")
    p.add_argument("--pose-path",      required=True, metavar="FILE")
    p.add_argument("--concepts-path",  default=None, metavar="FILE")
    p.add_argument("--out",            required=True, type=Path,
                   help="Output directory (per-track JSONs + summary + plot).")

    p.add_argument("--concepts", nargs="+", default=["car"],
                   help="Concept names whose tracks are extracted.")
    p.add_argument("--min-track-frames", type=int, default=10,
                   help="Drop tracks observed in fewer frames than this.")
    p.add_argument("--min-pixels", type=int, default=50,
                   help="Skip a frame's observation below this many valid pixels.")
    p.add_argument("--near-plane", type=float, default=0.5,
                   help="Discard depths closer than this (metres).")
    p.add_argument("--far-plane", type=float, default=80.0,
                   help="Discard depths beyond this (metres).")
    p.add_argument("--boundary-margin", type=int, default=2,
                   help="Px eroded around segment edges (cuts depth bleeding).")
    p.add_argument("--depth-pct", type=float, nargs=2, default=(20.0, 80.0),
                   metavar=("LO", "HI"),
                   help="Keep only mask pixels inside this depth-percentile band "
                        "(robustness against background bleed at mask edges).")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Centered moving-average window (frames) for x/y/z.")
    p.add_argument("--moving-threshold", type=float, default=2.0,
                   help="Net world-frame displacement (m) above which a track "
                        "is flagged as moving.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip writing the top-down PNG (no matplotlib needed).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-frame observation
# ---------------------------------------------------------------------------

@dataclass
class TrackObservations:
    """Raw per-frame observations of one persistent track."""
    track_id: int                                  # raw SAM 3 id (segment_id - 1)
    concept: str
    frame_indices: list[int] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)
    centroids: list[np.ndarray] = field(default_factory=list)  # (3,) world, m
    n_pixels: list[int] = field(default_factory=list)


def _wanted_seg_ids(f: Frame, id2label, wanted: set[str]) -> list[tuple[int, str]]:
    """Segments of this frame whose concept name is in ``wanted``.

    Returns ``[(segment_id, concept_name), ...]`` — remember
    ``segment_id == track_id + 1`` for the SAM 3 loader.
    """
    out = []
    for seg_id, lid in zip(f.segment_ids.tolist(), f.label_ids.tolist()):
        name = str(id2label.get(int(lid), "")).lower()
        if name in wanted:
            out.append((int(seg_id), name))
    return out


def _mask_centroid_world(
    f: Frame,
    mask: np.ndarray,
    near: float,
    far: float,
    depth_pct: tuple[float, float],
    min_pixels: int,
) -> tuple[np.ndarray, int] | None:
    """Robust world-frame centroid of one instance mask, or None.

    Same unprojection math as ``init_pc_from_loader._backproject_frame``, but
    gated to one mask and reduced to a median. The percentile band handles the
    depth bleeding that survives boundary erosion (a car mask that includes a
    sliver of background at 60 m would otherwise drag the mean away).
    """
    keep = mask & np.isfinite(f.depth) & (f.depth > near) & (f.depth < far)
    if int(keep.sum()) < min_pixels:
        return None

    # np.nonzero yields row-major pixel coords in the same order as
    # f.depth[keep], so the percentile band indexes all three consistently.
    vs, us = np.nonzero(keep)
    Z_all = f.depth[keep].astype(np.float64)
    lo, hi = np.percentile(Z_all, sorted(depth_pct))
    band = (Z_all >= lo) & (Z_all <= hi)
    if int(band.sum()) < max(1, min_pixels // 2):
        return None

    xyz_cam = unproject_pixels(f.camera, us[band], vs[band], Z_all[band])
    R = f.T_cam_to_world[:3, :3]
    t = f.T_cam_to_world[:3, 3]
    xyz_world = xyz_cam @ R.T + t
    return np.median(xyz_world, axis=0), int(band.sum())


# ---------------------------------------------------------------------------
# Trajectory post-processing
# ---------------------------------------------------------------------------

def _smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflected edges (length preserved)."""
    if window <= 1 or len(a) < 3:
        return a.astype(np.float64)
    w = min(window if window % 2 == 1 else window + 1, 2 * (len(a) // 2) + 1)
    half = w // 2
    padded = np.pad(a.astype(np.float64), half, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def _finalize_track(obs: TrackObservations, smooth_window: int,
                    moving_threshold: float) -> dict:
    """Smooth, derive heading + speed, and package one track as a JSON dict.

    ``trajectory.{x,y,z,theta}`` follows plan_trajectory.py's schema exactly
    (KITTI world frame; theta = yaw in radians = atan2(dx, dz), i.e. measured
    from +Z toward +X) so combine.py's place_object_from_trajectory_json can
    consume the file unchanged.
    """
    xyz = np.stack(obs.centroids, axis=0)  # (N, 3)
    fi = np.asarray(obs.frame_indices, dtype=np.int64)

    x = _smooth(xyz[:, 0], smooth_window)
    y = _smooth(xyz[:, 1], smooth_window)
    z = _smooth(xyz[:, 2], smooth_window)

    dx, dz = np.diff(x), np.diff(z)
    # Heading of each step, last value repeated so theta has N entries. For a
    # parked car the diffs are noise-sized and theta is meaningless — flagged
    # via is_moving below.
    theta = np.arctan2(dx, dz)
    theta = np.append(theta, theta[-1] if len(theta) else 0.0)

    dt = np.diff(fi).astype(np.float64) / KITTI_FPS  # gaps in frames → real dt
    step = np.hypot(dx, dz)
    v = np.append(step / np.maximum(dt, 1e-9), 0.0)

    path_length = float(step.sum())
    net_disp = float(np.hypot(x[-1] - x[0], z[-1] - z[0]))
    is_moving = net_disp >= moving_threshold

    return {
        "source": "extract_track_trajectories (observed SAM 3 track)",
        "track_id": obs.track_id,
        "concept": obs.concept,
        "n_steps": int(len(x)),
        "frame_indices": fi.tolist(),
        "frame_ids": obs.frame_ids,
        # dt_s is the sequence's NOMINAL frame period; steps of gappy tracks
        # are NOT uniformly spaced — use trajectory.t for real timestamps.
        "dt_s": 1.0 / KITTI_FPS,
        "has_gaps": bool(np.any(np.diff(fi) != 1)) if len(fi) > 1 else False,
        "path_length_m": round(path_length, 3),
        "net_displacement_m": round(net_disp, 3),
        "is_moving": bool(is_moving),
        "mean_pixels": int(np.mean(obs.n_pixels)),
        "trajectory": {
            "x": [round(float(a), 4) for a in x],
            "y": [round(float(a), 4) for a in y],
            "z": [round(float(a), 4) for a in z],
            "theta": [round(float(a), 5) for a in theta],
            "v": [round(float(a), 3) for a in v],
            # Real per-step timestamps (s, from the first observation) —
            # honest even when the track has frame gaps. Extra keys are
            # ignored by combine.py's reader.
            "t": [round(float(a), 3) for a in
                  (fi - fi[0]).astype(np.float64) / KITTI_FPS],
        },
    }


# ---------------------------------------------------------------------------
# Top-down plot
# ---------------------------------------------------------------------------

# Validated categorical palette (dataviz reference, light mode) — fixed slot
# order, assigned per track in first-seen order, never cycled: tracks beyond
# slot 8 render in the muted "other" gray.
_SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_EGO_GRAY = "#8a8a86"
_OTHER_GRAY = "#c3c2b7"


def _plot_topdown(tracks: list[dict], ego_xz: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ego_xz[:, 0], ego_xz[:, 1], color=_EGO_GRAY, lw=2,
            label="ego (camera path)", zorder=1)

    for i, tr in enumerate(sorted(tracks, key=lambda t: -t["path_length_m"])):
        color = _SERIES[i] if i < len(_SERIES) else _OTHER_GRAY
        x = np.asarray(tr["trajectory"]["x"])
        z = np.asarray(tr["trajectory"]["z"])
        style = "-" if tr["is_moving"] else ":"
        ax.plot(x, z, style, color=color, lw=2, zorder=2,
                label=f"track {tr['track_id']} "
                      f"({'moving' if tr['is_moving'] else 'parked'}, "
                      f"{tr['path_length_m']:.0f} m)")
        ax.plot(x[0], z[0], "o", color=color, ms=8, zorder=3)
        if i < 4:  # selective direct labels, not every series
            ax.annotate(f"{tr['track_id']}", (x[0], z[0]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=9, color="#40403e")

    ax.set_xlabel("x (m, world)")
    ax.set_ylabel("z (m, world)")
    ax.set_title("Observed SAM 3 track trajectories — top-down (KITTI world frame)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e5e4dd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    wanted = {c.lower() for c in args.concepts}

    loader = KITTISam3SequenceLoader(
        sequence_dir  = args.kitti_odom_seq,
        depth_dir     = args.depth_dir,
        sam3_dir      = args.sam3_dir,
        pose_path     = args.pose_path,
        concepts_path = args.concepts_path,
    )
    id2label = loader.id2label
    print(f"[tracks] loader: {loader.name}  ({len(loader)} frames), "
          f"concepts: {sorted(wanted)}")

    tracks: dict[int, TrackObservations] = {}
    ego_xz: list[tuple[float, float]] = []

    for idx in range(len(loader)):
        f = loader[idx]
        ego_xz.append((float(f.T_cam_to_world[0, 3]), float(f.T_cam_to_world[2, 3])))

        boundary = semantic_boundary_mask(f.panoptic_seg, args.boundary_margin)
        for seg_id, concept in _wanted_seg_ids(f, id2label, wanted):
            mask = (f.panoptic_seg == seg_id) & ~boundary
            res = _mask_centroid_world(
                f, mask, args.near_plane, args.far_plane,
                tuple(args.depth_pct), args.min_pixels,
            )
            if res is None:
                continue
            centroid, n_px = res
            raw_id = seg_id - 1  # loader convention: segment_id = track_id + 1
            obs = tracks.setdefault(raw_id, TrackObservations(raw_id, concept))
            # ABSOLUTE KITTI frame number (the file stem), not the loader's
            # enumeration index — consumers index poses/times by this, and
            # the two diverge when the loader skips an incomplete frame.
            obs.frame_indices.append(int(f.frame_id))
            obs.frame_ids.append(f.frame_id)
            obs.centroids.append(centroid)
            obs.n_pixels.append(n_px)

        if idx % 50 == 0 or idx == len(loader) - 1:
            print(f"  frame {f.frame_id}: {len(tracks)} tracks so far")

    kept = [o for o in tracks.values() if len(o.frame_indices) >= args.min_track_frames]
    dropped = len(tracks) - len(kept)
    print(f"[tracks] {len(tracks)} raw tracks -> {len(kept)} kept "
          f"(>= {args.min_track_frames} frames), {dropped} short ones dropped")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for obs in sorted(kept, key=lambda o: o.track_id):
        tr = _finalize_track(obs, args.smooth_window, args.moving_threshold)
        results.append(tr)
        out_json = args.out / f"track_{obs.track_id:03d}_trajectory.json"
        out_json.write_text(json.dumps(tr, indent=2))
        print(f"  track {obs.track_id:>3} ({obs.concept}): "
              f"{tr['n_steps']:>3} frames "
              f"[{obs.frame_ids[0]}..{obs.frame_ids[-1]}], "
              f"path {tr['path_length_m']:7.1f} m, "
              f"{'MOVING' if tr['is_moving'] else 'parked'} -> {out_json.name}")

    summary = {
        "loader": loader.name,
        "n_frames": len(loader),
        "concepts": sorted(wanted),
        "min_track_frames": args.min_track_frames,
        "n_tracks_raw": len(tracks),
        "n_tracks_kept": len(kept),
        "n_moving": sum(1 for t in results if t["is_moving"]),
        "tracks": [
            {k: t[k] for k in ("track_id", "concept", "n_steps", "path_length_m",
                               "net_displacement_m", "is_moving", "mean_pixels")}
            for t in results
        ],
    }
    (args.out / "tracks_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[✓] wrote {len(results)} trajectory JSONs + tracks_summary.json "
          f"-> {args.out}")

    if not args.no_plot and results:
        plot_path = args.out / "trajectories_topdown.png"
        _plot_topdown(results, np.asarray(ego_xz), plot_path)
        print(f"[✓] wrote top-down plot -> {plot_path}")

    if results and not any(t["is_moving"] for t in results):
        print("[note] no MOVING tracks survived — the sequence may only contain "
              "parked cars; consider a livelier sequence for the replay demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
