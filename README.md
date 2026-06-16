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

## SAM 3 instance segmentation

See `scripts/run_sam3_inference.py` for full documentation and all CLI options.

### Setup

No repo clone needed — SAM 3 is included in the `ultralytics` package. Install the required CLIP dependency:

```bash
pip install git+https://github.com/ultralytics/CLIP.git
```

Download `sam3.pt` (~3.3 GB) from the shared team cloud storage and place it at:

```
~/checkpoints/sam3/sam3.pt
```

That is, create the directory `checkpoints/sam3/` inside your own home folder and put the file there. The script finds it automatically — no extra flags needed.

```bash
mkdir -p ~/checkpoints/sam3
# then copy/download sam3.pt into that folder
```

### Run

```bash
python scripts/run_sam3_inference.py --sequences 00 01 02
```

All paths (dataset root, model weights, output directory) have sensible defaults. Override only when needed:

```bash
python scripts/run_sam3_inference.py --sequences 00 --output_dir /tmp/test
```

Concepts default to those listed in `concepts.txt` in the repo root (`car`, `pedestrian`, `cyclist`). Edit that file to change them, or pass `--concepts car truck pedestrian` to override on the fly.

Output: `{output_dir}/{sequence}/{frame_stem}.npz` with keys:
- `"instance_seg"` — int32 (H, W), `(track_id + 1)` per pixel (0 = background)
- `"track_ids"` / `"label_ids"` / `"scores"` — 1-D arrays, one entry per instance

`instance_seg` stores `track_id + 1` (not raw track IDs) to avoid ambiguity between background (0) and the first tracked object (also ID 0). To find track N's pixels: `instance_seg == N + 1`.

`label_ids` indexes into `{output_dir}/concepts.json` (written once), which maps each index to its concept name.

Unlike Mask2Former, the same object keeps the same `track_id` across all frames in a sequence, enabling downstream temporal accumulation. Interrupted runs resume automatically at the sequence level (tracking is stateful, so a partially-processed sequence reruns from frame 0).

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


