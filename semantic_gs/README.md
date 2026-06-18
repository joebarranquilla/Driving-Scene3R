# `semantic_gs` — Semantic Gaussian 3D Splatting module

This package is the author's responsibility inside the Driving-Scene3R
pipeline. It consumes calibrated RGB images, camera intrinsics, per-frame
camera poses, depth maps, and panoptic segmentation, and produces a
global **static** semantic 3D Gaussian Splatting representation of a
driving scene.

> **Hard rule:** this module never edits files outside `semantic_gs/` and
> `tests/`. Adapters wrap teammates' outputs without modifying them.

---

## Conventions (every submodule depends on these)

These are non-negotiable to keep geometry consistent across the pipeline.

### Camera frame: **OpenCV**
- `+X` right, `+Y` down, `+Z` forward (into the scene).
- Pixel `(u, v)` with `u = column index`, `v = row index`.
- Same convention used by KITTI, OpenCV, Mask2Former, and our existing
  `scripts/rgbd_to_pointcloud.py`.
- Back-projection: `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`.

### Pose: **camera-to-world (`c2w`)**
- Every `Frame.T_cam_to_world` is a `4 x 4 float64` matrix.
- A 3-D point `p_cam` in camera coords is mapped to world coords by
  `p_world = T_cam_to_world @ [p_cam; 1]`.
- World axis orientation is **dataset-defined**: for KITTI odometry we
  inherit cam-0's world (X right, Y down, Z forward at frame 0); for
  the dummy dataset we use the same convention so a forward-driving car
  has monotonically increasing `T[2, 3]`.

### Units
- Depth: **metres**, `float32`, `0` or `NaN` means "invalid".
- All translations: **metres**.

### Image tensors
- RGB: `uint8`, `(H, W, 3)`, channel order **RGB** (not BGR).
- Depth: `float32`, `(H, W)`.
- Panoptic segmentation: `int32`, `(H, W)`, value `0 = void`.
- Static mask: `bool`, `(H, W)`, `True = static pixel suitable for GS`.

---

## Implementation phases

| Phase | Status | Description |
|---|---|---|
| **0** | ✅ done | Skeleton, `Frame`/`PinholeCamera` contracts, dummy adapter, smoke test, tests. |
| **1** | ✅ done | KITTI-odometry adapter (RGB + calib + poses + teammate depth/panoptic NPZs), mock teammate writer for hermetic tests, `--kitti-odom-seq` smoke-test CLI. |
| **2** | ✅ done | Adapter for teammate's `lift_to_semantic_pointcloud.py` PLY+NPZ output (`SemanticPointCloud` dataclass), Cityscapes class table, `init_pointcloud.py` CLI for inspection + 3-panel viz, semantic-boundary-margin support in `build_static_mask`. |
| **3** | ✅ done | Vanilla `gsplat` training (RGB only) initialised from the semantic point cloud. `GaussianModel`, photometric loss (L1 + SSIM, masked), trainer with eval / checkpoint / Inria-format PLY export, `train_gs.py` CLI. |
| 4 | todo | Per-Gaussian semantic-logit head + cross-entropy supervision. |
| 5 | todo | Scale-up, densification schedule, W&B logging, SLURM (PRACT QoS). |
| 6 | todo | UE5 export. |

---

## Cross-team alignment notes

These reflect agreements with the `lift-semantic` branch owner and must
stay synchronised if their script's behaviour changes.

### Person & rider are STATIC (not dynamic)
`DEFAULT_DYNAMIC_CLASSES` only contains the vehicle classes (car, truck,
bus, train, motorcycle, bicycle). Person and rider are kept as static
because the teammate's lift script does so, and aligning the two
prevents Phase 3 GS training from fighting itself (init points existing
where the photometric loss says "remove these"). Flip via the
`dynamic_classes` argument if Phase 3 visuals show ghosting.

### ~6 cm cam-0/cam-2 baseline drift in incoming clouds
The teammate's `lift_to_semantic_pointcloud.py` uses cam-0 odometry
poses directly for cam-2 back-projection (i.e. the rectified ~6 cm baseline
along world-X is **not** applied). Our Phase-1 `KITTIOdomSequenceLoader`
*does* apply that shift. Net effect: every point in the loaded
`SemanticPointCloud` is shifted ~6 cm in world-X relative to the per-frame
camera poses used for Phase-3 photometric supervision. The error is
**systematic**, not smeared, so GS training will correct it within the
first few hundred iterations. No code change needed today; revisit if
visual artefacts appear.

### Cityscapes label table
`semantic_gs/data/cityscapes.py` mirrors
`scripts/utils.py::CITYSCAPES_LABELS` from the lift-semantic branch.
Kept in sync by hand until `utils.py` is promoted to a proper shared
package (open issue for the team).

---

## Quickstart

### Phase 0 — synthetic data (no GPU, no dataset, no teammate output)

```bash
pip install -e .                                 # optional but recommended
python -m semantic_gs.scripts.smoke_test_adapters --dummy --out smoke_test_out
pytest -q
```

### Phase 1 — real KITTI odometry (per-frame data)

Prerequisites on the I9 cluster (or any machine with the same files):

| Input | Path (example) |
|---|---|
| KITTI sequence dir (`image_2/` + `calib.txt`) | `/storage/group/dataset_mirrors/.../sequences/04` |
| KITTI GT poses (sequences 00–10) | `/storage/group/dataset_mirrors/.../poses/04.txt` |
| Teammate depth NPZs | `/storage/user/<user>/depth_predictions/04/*.npz` |
| Teammate panoptic NPZs + `id2label.json` | `/storage/user/<user>/panoptic_predictions/{04/*.npz, id2label.json}` |

```bash
python -m semantic_gs.scripts.smoke_test_adapters \
    --kitti-odom-seq /storage/.../sequences/04 \
    --depth-dir      /storage/user/<user>/depth_predictions/04 \
    --pano-dir       /storage/user/<user>/panoptic_predictions/04 \
    --pose-path      /storage/.../poses/04.txt \
    --frames 0 50 100 --out smoke_test_out
```

The same 4-panel viz as the dummy case, but on real KITTI frames.

### Phase 2 — semantic point cloud (sequence-aggregated)

After the teammate runs `scripts/lift_to_semantic_pointcloud.py`:

```bash
# Quick stats + 3-panel orthographic viz of a teammate PLY/NPZ:
python -m semantic_gs.scripts.init_pointcloud \
    --input /storage/user/<user>/semantic_clouds/seq04_static.ply \
    --out   smoke_test_out

# Same on synthetic data (no teammate run needed — proves the pipeline):
python -m semantic_gs.scripts.init_pointcloud --self-test --out smoke_test_out
```

### Phase 3 — Gaussian Splatting training (CUDA GPU required)

Trains a vanilla 3DGS model initialised from the Phase 2 point cloud
and supervised by the per-frame KITTI RGB (masked by `static_mask`).
Produces:

* `summary.json` — final PSNR/SSIM on held-out eval frames
* `metrics.csv` — per-iter training curves
* `renders/iter_*_eval_*.png` — 4-panel **predicted vs GT vs error vs mask**
  visualisations (the smoking-gun "is the reconstruction correct?" check)
* `exports/gaussians_final.ply` — **standard Inria 3DGS PLY** that you can
  drag-and-drop into <https://playcanvas.com/supersplat/editor> to fly
  around the reconstructed scene in 3D
* `checkpoints/iter_*.pt` — torch state for Phase 4 to extend

```bash
# Phase-3 deps (CUDA-only — install on the GPU box, NOT your Windows laptop):
pip install -e ".[phase3]"          # adds gsplat

# Train:
python -m semantic_gs.scripts.train_gs \
    --kitti-odom-seq /storage/.../sequences/04 \
    --depth-dir      /storage/user/<user>/depth_predictions/04 \
    --pano-dir       /storage/user/<user>/panoptic_predictions/04 \
    --pose-path      /storage/.../sequences/04/poses.txt \
    --init-pc        /storage/user/<user>/semantic_clouds/seq04_static.npz \
    --out            /storage/user/<user>/semantic_gs_runs/seq04 \
    --max-iters      10000 --sky-dome --sky-dome-ground

# CUDA smoke test (no KITTI / no teammate output needed, runs in ~30 s):
python -m semantic_gs.scripts.train_gs --dummy --max-iters 200 \
    --out runs/dummy
```

A POC run on KITTI seq 04 with the default 1500 iterations + a
~2-5 M-point lift cloud takes ~3-5 min on a single 24 GB GPU and
typically reaches ~18-25 dB PSNR on held-out frames.

### Testing without any teammate output

The mock writer in `semantic_gs/data/adapters/mock_teammate_outputs.py`
materialises:

1. The dummy adapter's frames in the **exact KITTI on-disk layout** the
   teammates produce (see `materialize_dummy_as_kitti_layout`).
2. The same dummy fused into the **exact PLY + NPZ format** of
   `lift_to_semantic_pointcloud.py` output (see
   `materialize_dummy_as_lift_output`).

This lets the entire module be exercised end-to-end without anyone
having run inference on the cluster. See `tests/` for canonical
recipes.

Expected console output of the Phase-0/1 smoke test:
- Five lines summarizing per-frame shapes/dtypes/value ranges.
- A 4-panel PNG written to `smoke_test_out/<loader>_frame_<stem>.png`
  showing RGB | depth | panoptic | static-mask.

Visual sanity check: dynamic vehicle classes (cars, trucks, ...) **and**
sky (KITTI loader default) must be **missing** from the static mask
while road / building / vegetation / person / rider are kept.



