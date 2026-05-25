#!/usr/bin/env python3
"""
MobileStereoNet (MSNet2D) inference script for KITTI odometry sequences.

Reads stereo pairs from image_2/ (left) and image_3/ (right), converts the
predicted disparity maps to metric depth using the camera calibration, and
saves one compressed NPZ file per frame:

    {output_dir}/{sequence}/{frame_stem}.npz   →  key "depth": float32 (H, W) in metres

Usage example
-------------
python scripts/run_mobilestereonet_inference.py \\
    --msnet_path /usr/prakt/s0038/mobilestereonet \\
    --dataset_root /storage/group/dataset_mirrors/kitti_odom_color/\\
data_odometry_color/dataset/sequences \\
    --sequences 00 01 02 \\
    --checkpoint /usr/prakt/s0038/checkpoints/MSNet2D_KITTI.ckpt \\
    --output_dir /usr/prakt/s0038/depth_predictions

Checkpoint download
-------------------
1. Go to https://github.com/cogsys-tuebingen/mobilestereonet
2. In the evaluation tables, the model names are hyperlinks to pretrained
   weights hosted on Google Drive.  For KITTI, prefer the model trained on
   "SF + KITTI2015" (SceneFlow pretraining + KITTI fine-tuning).
3. Download the .ckpt file and place it at the path you pass to --checkpoint.

MobileStereoNet source
----------------------
Clone with:
    git clone https://github.com/cogsys-tuebingen/mobilestereonet \\
        /usr/prakt/s0038/mobilestereonet
then pass that directory to --msnet_path.
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MobileStereoNet (MSNet2D) inference on KITTI odometry sequences "
            "and save per-frame depth maps as compressed NPZ files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    parser.add_argument(
        "--msnet_path", required=True,
        help=(
            "Path to the cloned MobileStereoNet repository. "
            "Clone: git clone https://github.com/cogsys-tuebingen/mobilestereonet"
        ),
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help=(
            "Root directory that contains numbered KITTI odometry sequence folders "
            "(e.g. /storage/.../dataset/sequences)."
        ),
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True,
        metavar="SEQ",
        help="One or more sequence IDs to process, e.g.: --sequences 00 01 02",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help=(
            "Path to the pretrained MSNet2D checkpoint (.ckpt). "
            "See module docstring for download instructions."
        ),
    )

    # --- optional ---
    parser.add_argument(
        "--output_dir", default="/usr/prakt/s0038/depth_predictions",
        help="Root directory for output NPZ files.",
    )
    parser.add_argument(
        "--model", default="MSNet2D", choices=["MSNet2D", "MSNet3D"],
        help="MobileStereoNet variant.",
    )
    parser.add_argument(
        "--maxdisp", type=int, default=192,
        help="Maximum disparity search range (must match the checkpoint).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of stereo pairs to process in one forward pass.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="CPU worker processes for image loading.",
    )
    parser.add_argument(
        "--pad_to", type=int, default=32,
        help=(
            "Pad image dimensions to the next multiple of this value before "
            "inference (MobileStereoNet requires divisibility by 32)."
        ),
    )
    parser.add_argument(
        "--no_cuda", action="store_true",
        help="Disable CUDA and run on CPU (very slow for long sequences).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# KITTI calibration parsing
# ---------------------------------------------------------------------------

def parse_kitti_calib(calib_path: str):
    """
    Parse a KITTI odometry calib.txt file and return the focal length (px)
    and stereo baseline (m) for the colour stereo pair (cameras 2 and 3).

    calib.txt format (space-separated 3×4 matrix rows):
        P0: <12 values>   ← left  greyscale
        P1: <12 values>   ← right greyscale
        P2: <12 values>   ← left  colour  (image_2)
        P3: <12 values>   ← right colour  (image_3)
        Tr: <12 values>   ← lidar-to-camera transform

    P2 = [f, 0, cx, 0,   0, f, cy, 0,   0, 0, 1, 0]
    P3 = [f, 0, cx, -f·B, 0, f, cy, 0,  0, 0, 1, 0]
    → baseline B = |P3[0,3]| / P3[0,0]
    """
    data = {}
    with open(calib_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = np.fromstring(value, sep=" ", dtype=np.float64)

    P2 = data["P2"].reshape(3, 4)
    P3 = data["P3"].reshape(3, 4)

    focal_length = float(P2[0, 0])                          # pixels
    baseline     = float(abs(P3[0, 3]) / P3[0, 0])         # metres

    return focal_length, baseline


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

# ImageNet statistics used throughout MobileStereoNet
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_normalised_tensor = transforms.Compose([
    transforms.ToTensor(),                            # uint8 HWC → float32 CHW in [0,1]
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def load_image_tensor(path: str) -> torch.Tensor:
    """Load a PNG image as a normalised float32 tensor of shape (3, H, W)."""
    img = Image.open(path).convert("RGB")
    return _to_normalised_tensor(img)   # (3, H, W)


def pad_tensor_to_multiple(tensor: torch.Tensor, multiple: int = 32):
    """
    Pad a (C, H, W) tensor so that H and W are each divisible by *multiple*.

    Padding is added to the **top** and **right** edges to match the
    convention used in the original MobileStereoNet prediction script.

    Returns
    -------
    padded   : torch.Tensor  (C, H', W')
    top_pad  : int
    right_pad: int
    """
    _, h, w = tensor.shape
    top_pad   = (multiple - h % multiple) % multiple
    right_pad = (multiple - w % multiple) % multiple
    # F.pad order: (left, right, top, bottom)
    padded = F.pad(tensor, (0, right_pad, top_pad, 0), mode="constant", value=0.0)
    return padded, top_pad, right_pad


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    msnet_path: str,
    model_name: str,
    maxdisp: int,
    checkpoint_path: str,
    use_cuda: bool,
) -> nn.Module:
    """
    Dynamically import MobileStereoNet from *msnet_path*, instantiate the
    requested model, and load the pretrained checkpoint.
    """
    if msnet_path not in sys.path:
        sys.path.insert(0, msnet_path)

    # Import after path is set up
    try:
        from models import __models__  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(
            f"ERROR: could not import MobileStereoNet models from '{msnet_path}'.\n"
            f"Make sure --msnet_path points to the repository root.\n"
            f"Detail: {exc}"
        )

    if model_name not in __models__:
        sys.exit(
            f"ERROR: model '{model_name}' not found in MobileStereoNet. "
            f"Available: {list(__models__.keys())}"
        )

    model = __models__[model_name](maxdisp)
    model = nn.DataParallel(model)

    if use_cuda:
        model.cuda()

    print(f"Loading checkpoint: {checkpoint_path}")
    map_location = None if use_cuda else torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict["model"])
    model.eval()
    print("Checkpoint loaded successfully.\n")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_batch(
    model: nn.Module,
    left_batch: torch.Tensor,    # (B, 3, H_pad, W_pad)
    right_batch: torch.Tensor,   # (B, 3, H_pad, W_pad)
    top_pads: list,
    right_pads: list,
    use_cuda: bool,
) -> list:
    """
    Forward pass on one batch of padded stereo pairs.

    Returns a list of (H_orig, W_orig) float32 numpy disparity arrays
    (in pixels) with padding removed.
    """
    if use_cuda:
        left_batch  = left_batch.cuda()
        right_batch = right_batch.cuda()

    disp_preds = model(left_batch, right_batch)
    disp_batch = disp_preds[-1]   # (B, H_pad, W_pad) — final prediction head

    result = []
    for i in range(disp_batch.shape[0]):
        disp = disp_batch[i].cpu().numpy().astype(np.float32)   # (H_pad, W_pad)
        tp = int(top_pads[i])
        rp = int(right_pads[i])
        # Crop out the padding
        h_crop = disp.shape[0] - tp if tp > 0 else disp.shape[0]
        w_crop = disp.shape[1] - rp if rp > 0 else disp.shape[1]
        disp = disp[tp : tp + h_crop, :w_crop]
        result.append(disp)

    return result


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def collect_frames(left_dir: str, right_dir: str):
    """
    Return sorted lists of (left_path, right_path) tuples for a sequence.
    Only .png files are considered.
    """
    left_files  = sorted(f for f in os.listdir(left_dir)  if f.endswith(".png"))
    right_files = sorted(f for f in os.listdir(right_dir) if f.endswith(".png"))

    if len(left_files) != len(right_files):
        raise ValueError(
            f"Frame count mismatch: {len(left_files)} left vs "
            f"{len(right_files)} right images."
        )

    return [
        (os.path.join(left_dir, lf), os.path.join(right_dir, rf))
        for lf, rf in zip(left_files, right_files)
    ]


def process_sequence(
    seq_id: str,
    dataset_root: str,
    output_dir: str,
    model: nn.Module,
    batch_size: int,
    pad_to: int,
    use_cuda: bool,
) -> None:
    seq_dir   = os.path.join(dataset_root, seq_id)
    calib_txt = os.path.join(seq_dir, "calib.txt")
    left_dir  = os.path.join(seq_dir, "image_2")
    right_dir = os.path.join(seq_dir, "image_3")

    for p in [calib_txt, left_dir, right_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    focal_length, baseline = parse_kitti_calib(calib_txt)
    print(f"  Calibration  →  f = {focal_length:.2f} px,  B = {baseline:.4f} m")

    frame_pairs = collect_frames(left_dir, right_dir)
    print(f"  Total frames →  {len(frame_pairs)}")

    out_seq_dir = os.path.join(output_dir, seq_id)
    os.makedirs(out_seq_dir, exist_ok=True)

    # --- Resume support: filter out already-saved frames ---
    pending = []
    for lp, rp in frame_pairs:
        stem     = os.path.splitext(os.path.basename(lp))[0]
        out_path = os.path.join(out_seq_dir, f"{stem}.npz")
        if not os.path.exists(out_path):
            pending.append((lp, rp, out_path))

    n_done = len(frame_pairs) - len(pending)
    if n_done:
        print(f"  Skipping     →  {n_done} already-saved frames (resume mode)")
    if not pending:
        print("  Nothing to do — all frames already saved.")
        return

    print(f"  To process   →  {len(pending)} frames  (batch_size={batch_size})\n")

    # --- Batched inference loop ---
    n_batches = (len(pending) + batch_size - 1) // batch_size
    for b_idx in tqdm(range(n_batches), desc=f"seq {seq_id}", unit="batch"):
        batch = pending[b_idx * batch_size : (b_idx + 1) * batch_size]

        left_tensors  = []
        right_tensors = []
        top_pads_list  = []
        right_pads_list = []

        for lp, rp, _ in batch:
            l_t = load_image_tensor(lp)
            r_t = load_image_tensor(rp)
            l_t, tp, rp_val = pad_tensor_to_multiple(l_t, multiple=pad_to)
            r_t, _,  _      = pad_tensor_to_multiple(r_t, multiple=pad_to)
            left_tensors.append(l_t)
            right_tensors.append(r_t)
            top_pads_list.append(tp)
            right_pads_list.append(rp_val)

        left_batch_t  = torch.stack(left_tensors,  dim=0)   # (B, 3, H_pad, W_pad)
        right_batch_t = torch.stack(right_tensors, dim=0)

        disp_maps = run_inference_batch(
            model, left_batch_t, right_batch_t,
            top_pads_list, right_pads_list, use_cuda,
        )

        # Convert disparity (px) → depth (m) and save
        for (_, _, out_path), disp in zip(batch, disp_maps):
            valid = disp > 0
            depth = np.zeros_like(disp)
            depth[valid] = (focal_length * baseline) / disp[valid]
            np.savez_compressed(out_path, depth=depth)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Validate critical paths before loading anything ---
    if not os.path.isdir(args.msnet_path):
        sys.exit(
            f"ERROR: --msnet_path '{args.msnet_path}' is not a directory.\n"
            "Clone MobileStereoNet with:\n"
            "  git clone https://github.com/cogsys-tuebingen/mobilestereonet "
            f"  {args.msnet_path}"
        )

    if not os.path.isfile(args.checkpoint):
        sys.exit(
            f"ERROR: checkpoint not found at '{args.checkpoint}'.\n\n"
            "Download a pretrained model:\n"
            "  1. Visit https://github.com/cogsys-tuebingen/mobilestereonet\n"
            "  2. In the evaluation tables, click the hyperlinked model name\n"
            "     (e.g. 'SF + KITTI2015') to download the .ckpt from Google Drive.\n"
            f"  3. Place the file at:  {args.checkpoint}"
        )

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        print(
            "WARNING: CUDA is not available or was disabled. "
            "CPU inference will be very slow for long KITTI sequences.\n"
        )

    model = load_model(
        args.msnet_path, args.model, args.maxdisp, args.checkpoint, use_cuda
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for seq_id in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"  Sequence: {seq_id}")
        print(f"{'=' * 60}")
        process_sequence(
            seq_id       = seq_id,
            dataset_root = args.dataset_root,
            output_dir   = args.output_dir,
            model        = model,
            batch_size   = args.batch_size,
            pad_to       = args.pad_to,
            use_cuda     = use_cuda,
        )

    print(f"\nAll sequences done. Depth maps written to: {args.output_dir}")


if __name__ == "__main__":
    main()
