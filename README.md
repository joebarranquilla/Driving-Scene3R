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

