"""Replay renderer: animate a car asset along an observed trajectory in a splat.

Phase 2 of the SAM 3 downstream task. Takes a trained 3DGS world PLY, a
gaussian car asset (SAM 3D Objects output), and a
``track_<id>_trajectory.json`` from ``extract_track_trajectories``, then for
every trajectory step poses the car at the observed position/heading and
renders the composited scene from the *real* ego camera pose of the same
KITTI frame. Frames are written as PNGs and stitched into an MP4 (optionally
side-by-side with the real camera images).

Requires CUDA (gsplat). Run ``module load cuda/13.0.1`` first on the TUM
workstations so gsplat can JIT-compile its kernels.

Example
-------
    python -m semantic_gs.scripts.render_replay \\
        --world-ply       ~/afm_shared/3dgs_gaussians/gaussians_20260609_try_5_dome.ply \\
        --object-ply      ~/afm_shared/sam3d/car_example/car.ply \\
        --trajectory-json ~/track_trajectories/04/track_005_trajectory.json \\
        --kitti-odom-seq  /storage/.../sequences/04 \\
        --pose-path       /storage/.../sequences/04/04.txt \\
        --out             ~/replay_runs/seq04_track005 --side-by-side
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from semantic_gs.export.ply import load_gaussian_ply  # noqa: F401  (re-exported;
# also imported from here by local visualization tooling)
from semantic_gs.geometry.cameras import PinholeCamera
from semantic_gs.geometry.poses import (
    cam_i_to_world_from_cam0,
    parse_kitti_calib,
    parse_kitti_poses,
)

# SAM 3D Objects / TriPoSR assets come out in OpenCV object space; this maps
# them to the KITTI camera/world orientation (same convention as the
# teammate's combine.py, where it fixes "upside-down and backwards" assets).
# diag(-1,-1,1) is a 180-degree rotation about +Z.
_M_ALIGN = np.diag([-1.0, -1.0, 1.0])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a car asset driving its observed trajectory "
                    "inside a trained 3DGS world.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--world-ply",       required=True, metavar="PLY")
    p.add_argument("--object-ply",      required=True, metavar="PLY")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectory-json", nargs="+", metavar="JSON",
                     help="One or more track_*_trajectory.json files.")
    src.add_argument("--tracks-dir", metavar="DIR",
                     help="Load every track_*_trajectory.json in this dir.")
    p.add_argument("--min-path-length", type=float, default=0.0,
                   help="Skip tracks with path_length_m below this (metres).")
    p.add_argument("--kitti-odom-seq",  required=True, metavar="SEQ_DIR",
                   help="Sequence dir (calib.txt + image_2 for --side-by-side).")
    p.add_argument("--pose-path",       required=True, metavar="FILE")
    p.add_argument("--out",             required=True, type=Path)

    p.add_argument("--target-size", type=float, default=4.5,
                   help="Longest side of the placed car (metres).")
    p.add_argument("--object-yaw-offset", type=float, default=180.0,
                   help="Asset yaw at theta=0 (deg). 180 makes the stock "
                        "SAM 3D car asset face +Z (its travel direction).")
    p.add_argument("--y-offset", type=float, default=0.0,
                   help="Extra vertical offset (m, +Y is DOWN in KITTI).")
    p.add_argument("--frames", type=int, default=0,
                   help="Render only the first N frames (0 = all).")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Render every Nth frame.")
    p.add_argument("--near-plane", type=float, default=0.1)
    p.add_argument("--far-plane", type=float, default=400.0,
                   help="Keep beyond the sky-dome radius or the sky vanishes.")
    p.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--no-video", action="store_true",
                   help="Skip the ffmpeg MP4 step (PNGs only).")
    p.add_argument("--side-by-side", action="store_true",
                   help="Also write rendered|real composite frames + MP4.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Object posing (pure numpy — unit-tested on CPU)
# ---------------------------------------------------------------------------

def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product ``q * r`` for wxyz quaternions; r may be (N, 4)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def canonicalize_object(g: dict[str, np.ndarray], target_size: float,
                        yaw_offset_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Center the asset, scale its longest side to ``target_size`` metres and
    pre-rotate from the asset convention into KITTI orientation (_M_ALIGN,
    then an extra yaw of ``yaw_offset_deg`` about +Y).

    The yaw offset defines which way the asset's nose points at theta = 0;
    the stock SAM 3D car asset needs 180 so it faces +Z (its direction of
    travel) instead of showing its headlights to the ego camera.
    """
    means = g["means"].astype(np.float64)
    centroid = means.mean(axis=0)
    means = means - centroid

    extent = means.max(axis=0) - means.min(axis=0)
    s = target_size / max(float(extent.max()), 1e-9)
    means = means * s

    phi = np.deg2rad(yaw_offset_deg)
    c, sn = np.cos(phi), np.sin(phi)
    R_yaw = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    R = R_yaw @ _M_ALIGN
    means = means @ R.T

    # _M_ALIGN is a 180-degree rotation about +Z -> quaternion (0, 0, 0, 1);
    # compose the yaw-offset quaternion (about +Y) on top.
    q_align = np.array([0.0, 0.0, 0.0, 1.0])
    q_yaw = np.array([np.cos(phi / 2.0), 0.0, np.sin(phi / 2.0), 0.0])
    q_pre = _quat_multiply(q_yaw, q_align[None, :])[0]
    quats = _quat_multiply(q_pre, g["quats"].astype(np.float64))

    out = dict(g)
    out["means"] = means.astype(np.float32)
    out["scales"] = (g["scales"] * s).astype(np.float32)
    out["quats"] = quats.astype(np.float32)
    return out


def pose_object(g: dict[str, np.ndarray], x: float, y: float, z: float,
                theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Yaw a canonicalized asset by ``theta`` and place it at (x, y, z).

    theta follows the trajectory convention: yaw from +Z toward +X
    (theta = atan2(dx, dz)), i.e. a right-hand rotation about +Y.
    Returns (means (N,3), quats (N,4) wxyz).
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    means = g["means"].astype(np.float64) @ R.T + np.array([x, y, z])
    q_yaw = np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0])
    quats = _quat_multiply(q_yaw, g["quats"].astype(np.float64))
    return means.astype(np.float32), quats.astype(np.float32)


def build_schedule(
    trajs: list[dict],
) -> tuple[dict[int, list[tuple[int, int]]], list[int], int]:
    """Map each KITTI frame index to the tracks visible in it.

    Returns ``(schedule, frame_range, max_active)`` where
    ``schedule[kitti_frame_idx] = [(traj_idx, step), ...]``,
    ``frame_range`` is the sorted union of all tracks' frame indices, and
    ``max_active`` is the largest number of simultaneously visible tracks.
    """
    schedule: dict[int, list[tuple[int, int]]] = {}
    for ti, traj in enumerate(trajs):
        for step, fidx in enumerate(traj["frame_indices"]):
            schedule.setdefault(int(fidx), []).append((ti, step))
    frame_range = sorted(schedule)
    max_active = max((len(v) for v in schedule.values()), default=0)
    return schedule, frame_range, max_active


def _load_trajectories(args: argparse.Namespace) -> list[dict]:
    if args.tracks_dir:
        paths = sorted(Path(args.tracks_dir).glob("track_*_trajectory.json"))
        if not paths:
            raise SystemExit(f"[error] no track_*_trajectory.json in {args.tracks_dir}")
    else:
        paths = [Path(p) for p in args.trajectory_json]

    trajs = []
    for p in paths:
        traj = json.loads(p.read_text())
        if traj.get("path_length_m", 0.0) < args.min_path_length:
            print(f"[replay] skipping track {traj.get('track_id')} "
                  f"({traj.get('path_length_m')} m < {args.min_path_length} m)")
            continue
        trajs.append(traj)
    if not trajs:
        raise SystemExit("[error] no trajectories left after filtering.")
    return trajs


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _ffmpeg_stitch(frames_dir: Path, fps: int, out_mp4: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"[✓] wrote {out_mp4}")


# ---------------------------------------------------------------------------
# Entry point (CUDA)
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("[error] render_replay needs CUDA (gsplat). "
                         "On TUM workstations: module load cuda/13.0.1")
    from gsplat import rasterization
    device = torch.device("cuda")

    trajs = _load_trajectories(args)
    schedule, frame_range, max_active = build_schedule(trajs)
    frame_range = frame_range[::max(1, args.frame_step)]
    if args.frames > 0:
        frame_range = frame_range[:args.frames]
    fidx_to_fid = {
        int(fi): fid
        for traj in trajs
        for fi, fid in zip(traj["frame_indices"], traj["frame_ids"])
    }
    print(f"[replay] {len(trajs)} tracks "
          f"({', '.join(str(t['track_id']) for t in trajs)}), "
          f"{len(frame_range)} frames, max {max_active} simultaneous cars")

    # Cameras: intrinsics from calib P2, per-frame cam2->world from poses.
    seq_dir = Path(args.kitti_odom_seq)
    calib = parse_kitti_calib(seq_dir / "calib.txt")
    P2 = calib["P2"]
    poses_cam0 = parse_kitti_poses(args.pose_path)
    first_fid = fidx_to_fid[frame_range[0]]
    with Image.open(seq_dir / "image_2" / f"{first_fid}.png") as im:
        W, H = im.size
    cam = PinholeCamera.from_kitti_P(P2, width=W, height=H)
    K = torch.tensor(cam.K, dtype=torch.float32, device=device)

    # World + max_active copies of the canonicalized car in one combined GPU
    # buffer; per frame only the car slots are rewritten. Slots without an
    # active track that frame get opacity 0.
    print(f"[replay] loading world: {args.world_ply}")
    world = load_gaussian_ply(args.world_ply)
    print(f"[replay] loading car  : {args.object_ply}")
    car = canonicalize_object(load_gaussian_ply(args.object_ply),
                              args.target_size, args.object_yaw_offset)
    n_w, n_c = len(world["means"]), len(car["means"])
    print(f"[replay] world {n_w:,} + {max_active} x {n_c:,} car gaussians")

    buf = {}
    for key in ("means", "colors", "opacities", "scales", "quats"):
        buf[key] = torch.from_numpy(np.concatenate(
            [world[key]] + [car[key]] * max_active, axis=0
        )).to(device)
    del world
    car_opac = torch.from_numpy(car["opacities"]).to(device)

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sbs_dir = args.out / "frames_sbs"
    if args.side_by_side:
        sbs_dir.mkdir(parents=True, exist_ok=True)

    bg = torch.tensor([list(args.bg)], dtype=torch.float32, device=device)
    t_start = time.time()
    for n, fidx in enumerate(frame_range):
        fid = fidx_to_fid[fidx]
        active = schedule[fidx]
        for slot in range(max_active):
            lo, hi = n_w + slot * n_c, n_w + (slot + 1) * n_c
            if slot < len(active):
                ti, step = active[slot]
                t = trajs[ti]["trajectory"]
                means_c, quats_c = pose_object(
                    car, t["x"][step], t["y"][step] + args.y_offset,
                    t["z"][step], t["theta"][step],
                )
                buf["means"][lo:hi] = torch.from_numpy(means_c).to(device)
                buf["quats"][lo:hi] = torch.from_numpy(quats_c).to(device)
                buf["opacities"][lo:hi] = car_opac
            else:
                buf["opacities"][lo:hi] = 0.0

        # NOTE: poses are indexed by ABSOLUTE KITTI frame number (the file
        # stem), not by the trajectory's loader position — the two differ as
        # soon as the loader skipped an incomplete frame.
        T_c2w = cam_i_to_world_from_cam0(poses_cam0[int(fid)], P2)
        viewmat = torch.tensor(np.linalg.inv(T_c2w), dtype=torch.float32,
                               device=device)

        rgb, _, _ = rasterization(
            means=buf["means"], quats=buf["quats"], scales=buf["scales"],
            opacities=buf["opacities"], colors=buf["colors"],
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=W, height=H, sh_degree=None,
            near_plane=args.near_plane, far_plane=args.far_plane,
            packed=False, backgrounds=bg,
        )
        img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(frames_dir / f"{n:06d}.png")

        if args.side_by_side:
            with Image.open(seq_dir / "image_2" / f"{fid}.png") as real_im:
                real = np.asarray(real_im.convert("RGB"))
            sbs = np.concatenate([img, real], axis=0)  # rendered on top
            Image.fromarray(sbs).save(sbs_dir / f"{n:06d}.png")

        if n % 25 == 0 or n == len(frame_range) - 1:
            dt = (time.time() - t_start) / (n + 1)
            print(f"  frame {n + 1}/{len(frame_range)} (kitti {fid}, "
                  f"{len(active)} car(s))  {dt * 1000:.0f} ms/frame",
                  flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[replay] done: {len(frame_range)} frames, VRAM peak {peak_gb:.1f} GB")

    if not args.no_video:
        _ffmpeg_stitch(frames_dir, args.fps, args.out / "replay.mp4")
        if args.side_by_side:
            _ffmpeg_stitch(sbs_dir, args.fps, args.out / "replay_sbs.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
