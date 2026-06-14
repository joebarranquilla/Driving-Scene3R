"""Phase 3 CLI: train a vanilla 3DGS model on a sequence + init point cloud.

Examples
--------
    # Real KITTI (this is what ``run_3dgs_e2e.sh`` invokes for Stage 10):
    python -m semantic_gs.scripts.train_gs \\
        --kitti-odom-seq /storage/.../sequences/04 \\
        --depth-dir      /storage/user/<u>/depth_predictions/04 \\
        --pano-dir       /storage/user/<u>/panoptic_predictions/04 \\
        --pose-path      /storage/.../sequences/04/poses.txt \\
        --init-pc        /storage/user/<u>/semantic_clouds/seq04_static.npz \\
        --out            /storage/user/<u>/semantic_gs_runs/seq04 \\
        --max-iters      1500

    # GPU smoke test (no KITTI, no teammate output — needs CUDA for gsplat):
    python -m semantic_gs.scripts.train_gs --dummy --max-iters 200 \\
        --out runs/dummy
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from semantic_gs.data.adapters.dummy import DummySequenceLoader
from semantic_gs.data.adapters.kitti_odom import KITTIOdomSequenceLoader
from semantic_gs.data.adapters.semantic_pointcloud_npz import (
    load_semantic_pointcloud_npz,
)
from semantic_gs.data.adapters.semantic_pointcloud_ply import (
    load_semantic_pointcloud_ply,
)
from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.data.pointcloud import SemanticPointCloud
from semantic_gs.train.trainer import TrainConfig, train


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a vanilla 3DGS model (Phase 3, RGB only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dummy", action="store_true",
        help="Synthetic 5-frame scene + synthetic init cloud (GPU smoke test).",
    )
    src.add_argument(
        "--kitti-odom-seq", metavar="SEQ_DIR",
        help="Path to a KITTI-odometry sequence dir.",
    )

    # KITTI-only flags ----------------------------------------------------
    p.add_argument("--depth-dir",    metavar="DIR")
    p.add_argument("--pano-dir",     metavar="DIR")
    p.add_argument("--pose-path",    metavar="FILE")
    p.add_argument("--id2label",     metavar="FILE", default=None)
    p.add_argument("--camera-index", type=int, default=2, choices=(2, 3))

    # Init point cloud (required for KITTI; defaults to synthetic for dummy)
    p.add_argument(
        "--init-pc", metavar="PATH",
        help="Initial semantic point cloud (.ply or .npz). "
             "Required for --kitti-odom-seq.",
    )

    # Training hyper-params ---------------------------------------------
    p.add_argument("--out",          type=Path, default=Path("runs/poc"))
    p.add_argument("--max-iters",    type=int,  default=1500)
    p.add_argument("--eval-every",   type=int,  default=500)
    p.add_argument("--ckpt-every",   type=int,  default=1500)
    p.add_argument("--eval-stride",  type=int,  default=10,
                   help="Every Nth loader frame is held out for eval.")
    p.add_argument("--lambda-ssim",  type=float, default=0.2)
    p.add_argument("--near-plane",   type=float, default=0.1)
    p.add_argument("--far-plane",    type=float, default=200.0)
    p.add_argument("--max-frames",   type=int,  default=0,
                   help="Cap the number of loader frames (0 = use all).")
    p.add_argument("--seed",         type=int,  default=0)
    p.add_argument("--no-renders",   action="store_true",
                   help="Skip writing per-eval-iter render PNGs (still saves PLY + JSON).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loader / init helpers
# ---------------------------------------------------------------------------

def _build_loader(args: argparse.Namespace) -> SequenceLoader:
    if args.dummy:
        return DummySequenceLoader(num_frames=5)
    missing = [name for name, val in
               (("--depth-dir", args.depth_dir),
                ("--pano-dir",  args.pano_dir),
                ("--pose-path", args.pose_path))
               if val is None]
    if missing:
        raise SystemExit(
            f"--kitti-odom-seq requires {missing}; pass them on the CLI."
        )
    return KITTIOdomSequenceLoader(
        sequence_dir = args.kitti_odom_seq,
        depth_dir    = args.depth_dir,
        panoptic_dir = args.pano_dir,
        pose_path    = args.pose_path,
        id2label_path= args.id2label,
        camera_index = args.camera_index,
    )


def _build_init_pc(
    args: argparse.Namespace, loader: SequenceLoader,
) -> SemanticPointCloud:
    if args.init_pc:
        path = Path(args.init_pc)
        if path.suffix.lower() == ".ply":
            return load_semantic_pointcloud_ply(path)
        if path.suffix.lower() == ".npz":
            return load_semantic_pointcloud_npz(path)
        raise SystemExit(
            f"--init-pc must be a .ply or .npz file, got {path.suffix}"
        )
    if args.dummy:
        # Synthesize a tiny cloud from the dummy frames so the GPU smoke
        # test works end-to-end without --init-pc.
        from semantic_gs.data.adapters.mock_teammate_outputs import (
            _backproject_static_pixels,
        )
        if not isinstance(loader, DummySequenceLoader):
            raise SystemExit("--dummy requires DummySequenceLoader")
        return _backproject_static_pixels(loader)
    raise SystemExit("--init-pc is required when --kitti-odom-seq is used.")


# ---------------------------------------------------------------------------
# Optional subset wrapper for --max-frames
# ---------------------------------------------------------------------------

class _SubsetLoader(SequenceLoader):
    """First-N wrapper around any :class:`SequenceLoader`."""

    def __init__(self, parent: SequenceLoader, k: int) -> None:
        self._parent = parent
        self._k = min(int(k), len(parent))

    @property
    def name(self) -> str:
        return f"{self._parent.name}/first{self._k}"

    @property
    def id2label(self) -> Mapping[int, str]:
        return self._parent.id2label

    def __len__(self) -> int:
        return self._k

    def __getitem__(self, idx: int) -> Frame:
        if not 0 <= idx < self._k:
            raise IndexError(f"frame index {idx} out of range [0, {self._k})")
        return self._parent[idx]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args   = _parse_args()
    loader = _build_loader(args)
    if args.max_frames and args.max_frames > 0:
        loader = _SubsetLoader(loader, args.max_frames)

    init_pc = _build_init_pc(args, loader)
    print(f"[init] loader        : {loader.name}  ({len(loader)} frames)")
    print(
        f"[init] init cloud    : {init_pc.n_points:,} pts  "
        f"({len(init_pc.class_counts())} classes)"
    )

    cfg = TrainConfig(
        max_iters    = args.max_iters,
        eval_every   = args.eval_every,
        ckpt_every   = args.ckpt_every,
        eval_stride  = args.eval_stride,
        lambda_ssim  = args.lambda_ssim,
        near_plane   = args.near_plane,
        far_plane    = args.far_plane,
        seed         = args.seed,
        save_renders = (not args.no_renders),
    )

    try:
        device = torch.device("cuda")
    except Exception as e:                       # pragma: no cover
        print(f"ERROR: failed to create CUDA device: {e}", file=sys.stderr)
        return 2

    summary = train(loader, init_pc, args.out, cfg, device=device)

    print()
    print("=" * 70)
    print(" PHASE 3 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  3DGS PLY (open in SuperSplat) : {summary['ply_path']}")
    print(f"  Final eval PSNR (mean)        : {summary['final_eval_psnr_mean']:.2f} dB")
    print(f"  Final eval SSIM (mean)        : {summary['final_eval_ssim_mean']:.3f}")
    print(f"  Renders / metrics / summary   : {args.out}")
    print()
    print("To inspect the trained 3DGS:")
    print("  1) Open https://playcanvas.com/supersplat/editor in any browser")
    print(f"  2) Drag-and-drop the file:  {summary['ply_path']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



