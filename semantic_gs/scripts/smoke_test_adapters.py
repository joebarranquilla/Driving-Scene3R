"""Phase 0/1 smoke test: load N frames from an adapter and dump a 4-panel
visualisation (RGB | depth | panoptic | static mask) for visual inspection.

Examples
--------
    # Pure synthetic data — no teammate output, no datasets needed.
    python -m semantic_gs.scripts.smoke_test_adapters --dummy --out smoke_test_out

    # Real KITTI odometry (Phase 1):
    python -m semantic_gs.scripts.smoke_test_adapters \\
        --kitti-odom-seq /storage/.../sequences/04 \\
        --depth-dir      /storage/user/<user>/depth_predictions/04 \\
        --pano-dir       /storage/user/<user>/panoptic_predictions/04 \\
        --pose-path      /storage/.../poses/04.txt \\
        --frames 0 50 100 --out smoke_test_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.kitti_odom import KITTIOdomSequenceLoader
from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Smoke-test data adapters by printing the contract of each loaded "
            "frame and saving a 4-panel visualisation per frame."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dummy", action="store_true",
                     help="Use the synthetic DummySequenceLoader.")
    src.add_argument("--kitti-odom-seq", metavar="SEQ_DIR",
                     help="Path to a KITTI-odometry sequence dir "
                          "(must contain image_2/ and calib.txt).")

    # --- KITTI-only flags (validated only if --kitti-odom-seq is used) ---
    p.add_argument("--depth-dir",  metavar="DIR",
                   help="(KITTI) directory with per-frame depth .npz files.")
    p.add_argument("--pano-dir",   metavar="DIR",
                   help="(KITTI) directory with per-frame panoptic .npz files.")
    p.add_argument("--pose-path",  metavar="FILE",
                   help="(KITTI) per-sequence poses .txt file.")
    p.add_argument("--id2label",   metavar="FILE", default=None,
                   help="(KITTI) optional override for id2label.json. "
                        "Defaults to <pano-dir>/../id2label.json.")
    p.add_argument("--camera-index", type=int, default=2, choices=(2, 3),
                   help="(KITTI) which P_i to use (2 = image_2, 3 = image_3).")

    # --- shared flags ---
    p.add_argument("--frames", type=int, nargs="+", default=[0],
                   help="Frame indices to inspect.")
    p.add_argument("--num-frames", type=int, default=5,
                   help="(dummy only) number of frames the loader exposes.")
    p.add_argument("--out", type=Path, default=Path("smoke_test_out"),
                   help="Output directory for the visualisation PNGs.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip writing the PNGs (only print the contract).")

    args = p.parse_args()

    if args.kitti_odom_seq:
        missing = [name for name, val in
                   (("--depth-dir", args.depth_dir),
                    ("--pano-dir",  args.pano_dir),
                    ("--pose-path", args.pose_path))
                   if val is None]
        if missing:
            p.error(f"--kitti-odom-seq requires: {', '.join(missing)}")
    return args


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_frame_contract(loader: SequenceLoader, frame: Frame) -> None:
    cam = frame.camera
    print(f"=== Frame {frame.frame_id}  (loader: {loader.name}) ===")
    print(f"  camera        : {cam.width}x{cam.height}  "
          f"fx={cam.fx:.2f}  fy={cam.fy:.2f}  cx={cam.cx:.2f}  cy={cam.cy:.2f}")
    print(f"  rgb           : shape={frame.rgb.shape} dtype={frame.rgb.dtype}")
    print(f"  depth         : shape={frame.depth.shape} dtype={frame.depth.dtype} "
          f"min={float(np.min(frame.depth)):.3f} "
          f"max={float(np.max(frame.depth)):.3f} (m)")
    print(f"  panoptic_seg  : shape={frame.panoptic_seg.shape} "
          f"dtype={frame.panoptic_seg.dtype} "
          f"unique={np.unique(frame.panoptic_seg).tolist()}")
    print(f"  N_segments    : {len(frame.segment_ids)}")
    print(f"  label_ids     : {frame.label_ids.tolist()}  "
          f"-> {[loader.id2label.get(int(l), '?') for l in frame.label_ids]}")
    print(f"  static_mask   : True frac = {frame.static_mask.mean():.3f}")
    print(f"  T_cam_to_world translation = {frame.T_cam_to_world[:3, 3].tolist()}")
    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_viz(loader: SequenceLoader, frame: Frame, out_path: Path) -> None:
    # Import matplotlib lazily so --no-viz works without it being installed.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{loader.name}  |  frame {frame.frame_id}", fontsize=11)

    axes[0].imshow(frame.rgb)
    axes[0].set_title("RGB")

    # Depth visualisation: clip to the 95th percentile of *static-mask*
    # depths so far-away sky / invalid pixels do not dominate the colour
    # scale and squash all real geometry into a single hue.
    depth_vis = frame.depth.copy()
    valid = np.isfinite(depth_vis) & (depth_vis > 0) & frame.static_mask
    if not valid.any():
        valid = np.isfinite(depth_vis) & (depth_vis > 0)
    vmax = float(np.percentile(depth_vis[valid], 95)) if valid.any() else 1.0
    # Hide invalid / out-of-range pixels so they appear as background grey.
    depth_show = np.where(valid, depth_vis, np.nan)
    im = axes[1].imshow(depth_show, vmin=0.0, vmax=vmax, cmap="turbo")
    axes[1].set_title(f"Depth (m, static-clip @ {vmax:.1f})")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panoptic: deterministic random colors per segment ID.
    seg = frame.panoptic_seg
    max_id = int(seg.max()) if seg.size else 0
    rng = np.random.default_rng(seed=0)
    colors = rng.random((max_id + 1, 3))
    colors[0] = (0.0, 0.0, 0.0)  # void = black
    axes[2].imshow(ListedColormap(colors)(seg))
    axes[2].set_title("Panoptic seg")

    axes[3].imshow(frame.static_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Static mask (frac={frame.static_mask.mean():.2f})")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_loader(args: argparse.Namespace) -> SequenceLoader:
    if args.dummy:
        return DummySequenceLoader(num_frames=args.num_frames)
    return KITTIOdomSequenceLoader(
        sequence_dir   = args.kitti_odom_seq,
        depth_dir      = args.depth_dir,
        panoptic_dir   = args.pano_dir,
        pose_path      = args.pose_path,
        id2label_path  = args.id2label,
        camera_index   = args.camera_index,
    )


def main() -> int:
    args = _parse_args()
    loader = _build_loader(args)

    print(f"Loader: {loader.name}  |  {len(loader)} frames available")

    for idx in args.frames:
        if not 0 <= idx < len(loader):
            print(f"WARN: frame index {idx} out of range [0, {len(loader)}), skipping.",
                  file=sys.stderr)
            continue
        frame = loader[idx]
        _print_frame_contract(loader, frame)
        if not args.no_viz:
            safe_name = loader.name.replace('/', '_')
            out_path = args.out / f"{safe_name}_frame_{frame.frame_id}.png"
            _save_viz(loader, frame, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

