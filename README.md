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

### 1 — Get the MobileStereoNet source

The script imports MobileStereoNet directly from a local clone; no installation needed.

```bash
git clone https://github.com/cogsys-tuebingen/mobilestereonet \
    /usr/prakt/s0038/mobilestereonet
```

### 2 — Download a pretrained checkpoint

1. Open <https://github.com/cogsys-tuebingen/mobilestereonet>
2. In the evaluation tables, the model names are **hyperlinks** to Google Drive.  
   For KITTI odometry, use the **MSNet2D** checkpoint trained on **"SF + KITTI2015"**.
3. Place the downloaded `.ckpt` at, e.g., `/usr/prakt/s0038/checkpoints/MSNet2D_KITTI.ckpt`

### 3 — Run inference

```bash
python scripts/run_mobilestereonet_inference.py \
    --msnet_path   /usr/prakt/s0038/mobilestereonet \
    --dataset_root /storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences \
    --sequences    00 01 02 \
    --checkpoint   /usr/prakt/s0038/checkpoints/MSNet2D_KITTI.ckpt \
    --output_dir   /usr/prakt/s0038/depth_predictions \
    --batch_size   4
```

| Argument | Default | Description |
|---|---|---|
| `--msnet_path` | — | Path to the cloned MobileStereoNet repo |
| `--dataset_root` | — | KITTI sequences root (contains `00/`, `01/`, …) |
| `--sequences` | — | Space-separated sequence IDs to process |
| `--checkpoint` | — | Pretrained `.ckpt` file |
| `--output_dir` | `/usr/prakt/s0038/depth_predictions` | Where to write NPZ files |
| `--model` | `MSNet2D` | `MSNet2D` or `MSNet3D` |
| `--maxdisp` | `192` | Maximum disparity (must match checkpoint) |
| `--batch_size` | `4` | Stereo pairs per forward pass |
| `--num_workers` | `4` | Image-loading workers |
| `--no_cuda` | off | Force CPU inference |

### Output format

One compressed `.npz` file per frame:

```
{output_dir}/{sequence}/{frame_stem}.npz
    └── "depth"  →  float32 array (H, W),  values in **metres**
```

Example load:

```python
import numpy as np
data  = np.load("depth_predictions/00/000000.npz")
depth = data["depth"]   # shape (375, 1242), dtype float32
```

The script is **resume-friendly**: if a `.npz` already exists for a frame it is
skipped automatically, so interrupted runs can be restarted safely.

