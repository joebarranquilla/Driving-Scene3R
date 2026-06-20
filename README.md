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

## Dynamic Objects

### Using Sam3d-Objects

#### Setup

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
conda env create -f environments/default.yml
conda activate sam3d-objects
# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
```

#### Run

(the models need around 32gb ram usage so run this inside the clusters)
first get the needed checkpoints from shared google drive folder and add them into checkpoints/
then add your images inside notebook/images. You need the original image, and the masks of the objects that you want to have an 3d version of. (You can first get the masks from sam3). Name the object masks {num}.png starting with 0.
replace the original sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py file in the repo with the gaussian_model.py that you can find in scripts/ for more efficient ram usage.
the original inference scripts in the repo are jupyter notebooks. A .py version of it can be found in scripts/demo_multi_object.py. 

```bash
python notebook/demo_multi_object.py
```

The .ply and gif output should then be saved in notebook/gaussians

### Adding the objects back into the gaussian world

Run the modified version of scripts/run_sam3_inference.py in order to get the masks of the singular objects. the masks are saved with the object id. find the object id that you want to use, e.g. 5, and then run by changing the track id accordingly:

```bash
python scripts/extract_mask_pixels.py   --npz /usr/prakt/s0043/sam3_predictions/04/000000.npz   --by track_id --value 5   --depth_npz /usr/prakt/s0043/depth_predictions/04/000000.npz   --save_uvz ./000000_obj5_xyz.csv
```

This saves a .csv file with the x,y and z coordinates of the object mask. 
Then, to merge the existing world ("../gaussians.ply") with the newly generated 3d object from sam3d-objects ("../sam-3d-objects/notebook/gaussians/new_mask_seq4_posed.ply"), change the necessary path variables accordignly in scripts/combine.py and run:

```bash
python scripts/combine.py
```

This creates a "merged_with_car.ply" that includes the newly generated dynamic object inside the static 3d world.

