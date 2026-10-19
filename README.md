# Driving-Scene3R
The goal of this project is to build a pipeline that leverages available foundation models in computer vision, and can output 3D reconstructed driving scenes from calibrated video sequences. 

## Conda environment (recommended)

Follow these steps to create a conda environment and install dependencies. There are two common ways to install PyTorch: a CPU-only build, or a CUDA-enabled build. Choose the command that matches your machine.

- Create and activate the environment (Python 3.10 recommended for compatibility):

```bash
conda create -n driving-scene3r python=3.10 -y
conda activate driving-scene3r
```

- Install the Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## MobileStereoNet depth inference

See `scripts/run_mobilestereonet_inference.py` for full documentation and all CLI options.

### Setup

```bash
# Clone MobileStereoNet (no pip install needed)
git clone https://github.com/cogsys-tuebingen/mobilestereonet /usr/prakt/<user>/mobilestereonet

# Download a pretrained checkpoint:
# go to https://github.com/cogsys-tuebingen/mobilestereonet,
# click the hyperlinked model name in the evaluation table (e.g. "SF + KITTI2015")
# and save the .ckpt to /usr/prakt/<user>/checkpoints/
```

### Run

```bash
python scripts/run_mobilestereonet_inference.py \
    --msnet_path   /usr/prakt/<user>/mobilestereonet \
    --dataset_root /storage/.../dataset/sequences \
    --sequences    00 01 02 \
    --checkpoint   /usr/prakt/<user>/checkpoints/MSNet2D_SF_KITTI2015.ckpt \
    --output_dir   /usr/prakt/<user>/depth_predictions \
    --batch_size   4
```

Output: `{output_dir}/{sequence}/{frame_stem}.npz` with key `"depth"` — float32 (H, W) in metres.  
Interrupted runs resume automatically (existing frames are skipped).

## Mask2Former panoptic segmentation

See `scripts/run_mask2former_inference.py` for full documentation and all CLI options.

### Run

No manual download needed — weights are fetched from the HuggingFace Hub on first use.

```bash
python scripts/run_mask2former_inference.py \
    --dataset_root /storage/.../dataset/sequences \
    --sequences    00 01 02 \
    --output_dir   /usr/prakt/<user>/panoptic_predictions \
    --batch_size   4
```

The default model is `facebook/mask2former-swin-large-cityscapes-panoptic` (Cityscapes label space: car, pedestrian, road, sky, …). A lighter variant can be selected with `--hf_model facebook/mask2former-swin-base-cityscapes-panoptic`.

Output: `{output_dir}/{sequence}/{frame_stem}.npz` with keys:
- `"panoptic_seg"` — int32 (H, W), segment ID per pixel (0 = void)
- `"segment_ids"` / `"label_ids"` / `"scores"` — 1-D arrays, one entry per segment

A `{output_dir}/id2label.json` mapping label IDs to class names is written once.

Interrupted runs resume automatically (existing frames are skipped).

## 3D scene pipeline

Once depth and panoptic predictions are ready, lift them into a static semantic
point cloud. Sky and dynamic vehicle classes are excluded from the scene cloud;
detected dynamic segments are saved separately under a sibling `*_dynamic/`
directory.

```bash
python scripts/lift_to_semantic_pointcloud.py \
    --dataset_root /storage/.../dataset/sequences \
    --sequence 00 \
    --depth_dir /usr/prakt/<user>/depth_predictions \
    --panoptic_dir /usr/prakt/<user>/panoptic_predictions \
    --output /usr/prakt/<user>/semantic_clouds/seq00_static.ply \
    --n_frames 150 --aggregation voxel --save_npz
```

Extract road geometry from the same sequence to create a road mesh, occupancy
grid, and centerline graph. The grid and centerline can then constrain a
feasible vehicle path for use when placing or reintroducing dynamic agents.
The planner consumes the road products rather than individual object tracks.

```bash
python scripts/extract_road_layout.py \
    --dataset_root /storage/.../dataset/sequences \
    --sequence 00 \
    --depth_dir /usr/prakt/<user>/depth_predictions \
    --panoptic_dir /usr/prakt/<user>/panoptic_predictions \
    --output_dir /usr/prakt/<user>/road_layout/seq00 \
    --n_frames 150

pip install casadi  # required by the trajectory optimizer
python scripts/plan_trajectory.py \
    --occupancy_grid /usr/prakt/<user>/road_layout/seq00/00_occupancy_grid.npz \
    --centerline /usr/prakt/<user>/road_layout/seq00/00_centerline_graph.npz \
    --output_dir /usr/prakt/<user>/trajectories/seq00 \
    --n_frames 150 --start_frame 0 --horizon_frames 100 --visualize
```

The planner writes `optimal_trajectory.npz` and `.json`, plus a top-down PNG
when `--visualize` is set. See each script's CLI help for additional options.

## Semantic Gaussian Splatting

The `semantic_gs/` package loads calibrated RGB frames, poses, depth, and
panoptic labels to initialize and train a static 3D Gaussian Splatting scene.
It includes KITTI and synthetic-data adapters, point-cloud inspection, and
CUDA-based photometric training with `gsplat`. Setup, smoke tests, and training
instructions are in [`semantic_gs/README.md`](semantic_gs/README.md).

## TripoSR trial 

### Setup

```bash
# Clone MobileStereoNet (no pip install needed)
git clone https://github.com/VAST-AI-Research/TripoSR /usr/prakt/<user>/
cd ../TripoSR
pip install -r requirements.txt
#follow troubleshooting in https://github.com/VAST-AI-Research/TripoSR
```

### Run

```bash
python run.py ../Driving-Scene3R/isolated_car.jpg --output-dir output/
```

after getting mesh of object, use scripts/pca.py for adding the mesh back into the point cloud of the world

```bash
#change back into root directory
python scripts/pca.py
```


