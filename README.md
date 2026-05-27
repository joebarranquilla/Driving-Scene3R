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


