"""Phase 2 CLI: load + summarise a teammate semantic point cloud.

Reads either the PLY or the NPZ output of
``scripts/lift_to_semantic_pointcloud.py`` and prints / visualises:

* total point count and world-space bounding box
* per-class histogram (with Cityscapes class names)
* three orthographic projections (XY top-down, XZ side, YZ front) colored
  by class — saved as a single PNG so you can sanity-check geometry
  without opening MeshLab / CloudCompare.

Examples
--------
    # Synthetic mock (no teammate run required):
    python -m semantic_gs.scripts.init_pointcloud --self-test --out smoke_test_out

    # Teammate output (after running scripts/lift_to_semantic_pointcloud.py):
    python -m semantic_gs.scripts.init_pointcloud \\
        --input /storage/user/<user>/semantic_clouds/seq04_static.ply \\
        --out   smoke_test_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from semantic_gs.data.adapters.semantic_pointcloud_npz import (
    load_semantic_pointcloud_npz,
)
from semantic_gs.data.adapters.semantic_pointcloud_ply import (
    load_semantic_pointcloud_ply,
)
from semantic_gs.data.cityscapes import (
    CITYSCAPES_ID_TO_COLOR_F32,
    CITYSCAPES_ID_TO_NAME,
)
from semantic_gs.data.pointcloud import SemanticPointCloud


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load and summarise a semantic point cloud produced by "
            "scripts/lift_to_semantic_pointcloud.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="PATH",
                     help="Path to a teammate .ply OR .npz semantic cloud.")
    src.add_argument("--self-test", action="store_true",
                     help="Run on a synthetic mock cloud (no teammate output needed).")
    p.add_argument("--out", type=Path, default=Path("smoke_test_out"),
                   help="Output directory for the 3-panel viz PNG.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip writing the viz PNG (only print stats).")
    p.add_argument("--max-viz-points", type=int, default=200_000,
                   help="Random-subsample to this many points before plotting "
                        "(matplotlib scatter is slow for >>1e6 pts).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_pointcloud(path: Path) -> SemanticPointCloud:
    ext = path.suffix.lower()
    if ext == ".ply":
        return load_semantic_pointcloud_ply(path)
    if ext == ".npz":
        return load_semantic_pointcloud_npz(path)
    raise ValueError(f"Unknown extension {ext!r}; expected .ply or .npz")


def _build_self_test_pointcloud() -> SemanticPointCloud:
    """Synthesize a small mock cloud via the dummy loader + mock writer."""
    from semantic_gs.data.adapters.dummy import DummySequenceLoader
    from semantic_gs.data.adapters.mock_teammate_outputs import (
        _backproject_static_pixels,
    )
    dummy = DummySequenceLoader(num_frames=5)
    return _backproject_static_pixels(dummy)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_stats(pc: SemanticPointCloud, source: str) -> None:
    print(f"=== Semantic point cloud: {source} ===")
    print(f"  N points         : {pc.n_points:,}")
    if pc.n_points == 0:
        print("  (empty)")
        return

    mn, mx = pc.bbox
    extent = mx - mn
    print(f"  bbox min (x,y,z) : ({mn[0]:.2f}, {mn[1]:.2f}, {mn[2]:.2f}) m")
    print(f"  bbox max (x,y,z) : ({mx[0]:.2f}, {mx[1]:.2f}, {mx[2]:.2f}) m")
    print(f"  bbox extent      : ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}) m")

    counts = pc.class_counts()
    print(f"  classes present  : {len(counts)}")
    total = sum(counts.values())
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    width = max((len(CITYSCAPES_ID_TO_NAME.get(cid, f'class_{cid}')) for cid in counts), default=8)
    for cid, n in rows:
        name = CITYSCAPES_ID_TO_NAME.get(cid, f"class_{cid}")
        frac = 100.0 * n / total
        print(f"    {name:<{width}}  id={cid:>2}  {n:>10,}  ({frac:5.1f} %)")
    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_viz(pc: SemanticPointCloud, out_path: Path, max_points: int) -> None:
    if pc.n_points == 0:
        print("[viz] empty cloud, nothing to plot.")
        return
    import matplotlib.pyplot as plt  # lazy

    rng = np.random.default_rng(seed=0)
    if pc.n_points > max_points:
        idx = rng.choice(pc.n_points, size=max_points, replace=False)
    else:
        idx = np.arange(pc.n_points)
    xyz    = pc.xyz[idx]
    labels = pc.labels[idx]

    # Cityscapes palette per point (fallback grey for unknown class IDs).
    pal = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    for cid in np.unique(labels):
        col = CITYSCAPES_ID_TO_COLOR_F32.get(int(cid), np.array([0.5, 0.5, 0.5], np.float32))
        pal[labels == int(cid)] = col

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Top-down (XY): driving plane.
    axes[0].scatter(xyz[:, 0], xyz[:, 2], c=pal, s=0.5, marker=".")
    axes[0].set_title("Top-down (X, Z)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].set_aspect("equal")

    # Side view (XZ in world is "X horizontal, Z forward"; here YZ is X vs Y).
    axes[1].scatter(xyz[:, 0], xyz[:, 1], c=pal, s=0.5, marker=".")
    axes[1].set_title("Front (X, Y)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m, down)")
    axes[1].invert_yaxis()   # so "up" in world is up in plot
    axes[1].set_aspect("equal")

    axes[2].scatter(xyz[:, 2], xyz[:, 1], c=pal, s=0.5, marker=".")
    axes[2].set_title("Side (Z, Y)")
    axes[2].set_xlabel("Z (m, forward)")
    axes[2].set_ylabel("Y (m, down)")
    axes[2].invert_yaxis()
    axes[2].set_aspect("equal")

    fig.suptitle(f"Semantic point cloud — {pc.n_points:,} pts "
                 f"(showing {len(idx):,})", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    if args.self_test:
        pc = _build_self_test_pointcloud()
        source = "self-test (dummy aggregation)"
        out_stem = "self_test"
    else:
        path = Path(args.input)
        if not path.is_file():
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            return 2
        pc = _load_pointcloud(path)
        source = str(path)
        out_stem = path.stem

    _print_stats(pc, source)
    if not args.no_viz:
        _save_viz(pc, args.out / f"pointcloud_{out_stem}.png", args.max_viz_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

