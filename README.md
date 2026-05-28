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

## Semantic 3D point cloud (static world)

See `scripts/lift_to_semantic_pointcloud.py` for full documentation and all CLI options.

Lifts MobileStereoNet depth maps and Mask2Former panoptic predictions into a 3D semantic point cloud that represents **only the static world**: sky and dynamic vehicle classes (car, truck, bus, train, motorcycle, bicycle) are removed; persons and riders are kept as static.

### Run

```bash
python scripts/lift_to_semantic_pointcloud.py \
    --dataset_root  /storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences \
    --sequence      00 \
    --depth_dir     /usr/prakt/<user>/depth_predictions \
    --panoptic_dir  /usr/prakt/<user>/panoptic_predictions \
    --poses_root    /storage/group/dataset_mirrors/kitti_odom_grey/sequences \
    --output        /usr/prakt/<user>/semantic_clouds/seq00_static.ply \
    --n_frames      10 \
    --aggregation   voxel \
    --voxel_size    0.1
```

Key arguments:
- `--n_frames` — frames to aggregate (default: 10, ≈1 s at 10 fps); `--start_frame` selects the starting index
- `--aggregation` — `concat` (stack as-is) | `voxel` (deduplication via voxel grid, default) | `icp` (ICP refinement on top of pose alignment, requires open3d)
- `--color_mode` — `semantic` (Cityscapes palette, default) | `rgb` (original image colours from `image_2/`)
- `--poses_root` — use when poses.txt lives under a different dataset mirror than the images (e.g. grey vs colour split)
- `--save_npz` — additionally save `xyz / colors / labels` arrays as a compressed NPZ

Output: a PLY file with XYZ + RGB + a `label` scalar property (semantic class ID) readable by CloudCompare, MeshLab, and Open3D.

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


