#!/usr/bin/env python3
"""
Mask2Former panoptic segmentation inference script for KITTI odometry sequences.

Reads left colour frames from image_2/ and runs Mask2Former panoptic segmentation,
saving one compressed NPZ file per frame:

    {output_dir}/{sequence}/{frame_stem}.npz
        "panoptic_seg"  →  int32  (H, W)   segment_id per pixel (0 = void/background)
        "segment_ids"   →  int32  (N,)     unique segment IDs present in this frame
        "label_ids"     →  int32  (N,)     semantic class ID for each segment
        "scores"        →  float32 (N,)    prediction confidence for each segment

Usage example
-------------
python scripts/run_mask2former_inference.py \\
    --dataset_root /storage/group/dataset_mirrors/kitti_odom_color/\\
data_odometry_color/dataset/sequences \\
    --sequences 00 01 02 \\
    --output_dir /usr/prakt/<user>/panoptic_predictions

Model
-----
The model is loaded directly from the HuggingFace Hub (no manual download needed).
The default checkpoint is trained on Cityscapes panoptic segmentation, which covers
the same label space as KITTI driving scenes (car, pedestrian, road, sky, …).

Available checkpoints (pass to --hf_model):
  facebook/mask2former-swin-large-cityscapes-panoptic   ← default, most accurate
  facebook/mask2former-swin-base-cityscapes-panoptic    ← lighter, faster
  facebook/mask2former-swin-small-cityscapes-panoptic   ← smallest

The first run will download and cache the model weights (~600 MB for swin-large).
Set the HF_HOME or TRANSFORMERS_CACHE env variable to control the cache location.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Mask2Former panoptic segmentation on KITTI odometry sequences "
            "and save per-frame results as compressed NPZ files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
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

    # --- optional ---
    parser.add_argument(
        "--output_dir", default="/usr/prakt/<user>/panoptic_predictions",
        help="Root directory for output NPZ files.",
    )
    parser.add_argument(
        "--hf_model",
        default="facebook/mask2former-swin-large-cityscapes-panoptic",
        help=(
            "HuggingFace Hub model ID. Weights are downloaded and cached "
            "automatically on first use."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of frames to process in one forward pass.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum confidence score for a predicted segment to be kept.",
    )
    parser.add_argument(
        "--overlap_threshold", type=float, default=0.8,
        help="Overlap area threshold for merging small disconnected mask regions.",
    )
    parser.add_argument(
        "--image_subdir", default="image_2",
        help="Sub-directory inside each sequence folder containing the PNG frames.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true",
        help="Disable CUDA and run on CPU (very slow for long sequences).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(hf_model: str, use_cuda: bool):
    """
    Load the Mask2Former processor and model from the HuggingFace Hub.
    Weights are cached locally after the first download.
    """
    print(f"Loading processor and model: {hf_model}")
    print("(downloading weights on first use — this may take a moment)")

    processor = AutoImageProcessor.from_pretrained(hf_model)
    model     = Mask2FormerForUniversalSegmentation.from_pretrained(hf_model)

    if use_cuda:
        model = model.cuda()

    model.eval()
    print("Model loaded successfully.\n")
    return processor, model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_batch(
    processor,
    model: Mask2FormerForUniversalSegmentation,
    pil_images: list,
    original_sizes: list,
    threshold: float,
    overlap_threshold: float,
    use_cuda: bool,
) -> list:
    """
    Run panoptic segmentation on a batch of PIL images.

    Parameters
    ----------
    pil_images     : list of PIL.Image.Image (RGB)
    original_sizes : list of (H, W) tuples — target output resolution per image
    threshold      : minimum segment confidence to keep
    overlap_threshold : overlap threshold for mask merging

    Returns
    -------
    list of dicts, one per image:
        {
          "panoptic_seg" : np.ndarray int32 (H, W),
          "segment_ids"  : np.ndarray int32 (N,),
          "label_ids"    : np.ndarray int32 (N,),
          "scores"       : np.ndarray float32 (N,),
        }
    """
    inputs = processor(images=pil_images, return_tensors="pt")

    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model(**inputs)

    # Post-process: resize predictions back to original image resolution
    panoptic_results = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=threshold,
        overlap_mask_area_threshold=overlap_threshold,
        target_sizes=original_sizes,
    )

    batch_out = []
    for result in panoptic_results:
        seg_map  = result["segmentation"]          # torch.Tensor int64 (H, W)
        seg_info = result["segments_info"]         # list of dicts

        if seg_map is None:
            # No segment above threshold — return an empty void frame
            h, w = original_sizes[len(batch_out)]
            panoptic_seg = np.zeros((h, w), dtype=np.int32)
            segment_ids  = np.array([], dtype=np.int32)
            label_ids    = np.array([], dtype=np.int32)
            scores       = np.array([], dtype=np.float32)
        else:
            panoptic_seg = seg_map.cpu().numpy().astype(np.int32)
            segment_ids  = np.array([s["id"]       for s in seg_info], dtype=np.int32)
            label_ids    = np.array([s["label_id"] for s in seg_info], dtype=np.int32)
            scores       = np.array([s["score"]    for s in seg_info], dtype=np.float32)

        batch_out.append({
            "panoptic_seg": panoptic_seg,
            "segment_ids":  segment_ids,
            "label_ids":    label_ids,
            "scores":       scores,
        })

    return batch_out


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def collect_frames(image_dir: str) -> list:
    """Return sorted absolute paths to all PNG files in *image_dir*."""
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in: {image_dir}")
    return [os.path.join(image_dir, f) for f in files]


def process_sequence(
    seq_id: str,
    dataset_root: str,
    output_dir: str,
    processor,
    model,
    batch_size: int,
    threshold: float,
    overlap_threshold: float,
    image_subdir: str,
    use_cuda: bool,
) -> None:
    seq_dir   = os.path.join(dataset_root, seq_id)
    image_dir = os.path.join(seq_dir, image_subdir)

    for p in [seq_dir, image_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    frame_paths = collect_frames(image_dir)
    print(f"  Total frames →  {len(frame_paths)}")

    out_seq_dir = os.path.join(output_dir, seq_id)
    os.makedirs(out_seq_dir, exist_ok=True)

    # --- Resume support: filter out already-saved frames ---
    pending = []
    for fp in frame_paths:
        stem     = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_seq_dir, f"{stem}.npz")
        if not os.path.exists(out_path):
            pending.append((fp, out_path))

    n_done = len(frame_paths) - len(pending)
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

        pil_images     = []
        original_sizes = []

        for fp, _ in batch:
            img = Image.open(fp).convert("RGB")
            pil_images.append(img)
            original_sizes.append((img.height, img.width))

        results = run_inference_batch(
            processor, model,
            pil_images, original_sizes,
            threshold, overlap_threshold,
            use_cuda,
        )

        for (_, out_path), res in zip(batch, results):
            np.savez_compressed(
                out_path,
                panoptic_seg = res["panoptic_seg"],
                segment_ids  = res["segment_ids"],
                label_ids    = res["label_ids"],
                scores       = res["scores"],
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        print(
            "WARNING: CUDA is not available or was disabled. "
            "CPU inference will be very slow for long KITTI sequences.\n"
        )

    processor, model = load_model(args.hf_model, use_cuda)

    # Save the model's label mapping alongside predictions for easy lookup later
    id2label = model.config.id2label   # {label_id: class_name}

    os.makedirs(args.output_dir, exist_ok=True)
    label_map_path = os.path.join(args.output_dir, "id2label.json")
    if not os.path.exists(label_map_path):
        with open(label_map_path, "w") as fh:
            json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)
        print(f"Label map saved to: {label_map_path}\n")

    for seq_id in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"  Sequence: {seq_id}")
        print(f"{'=' * 60}")
        process_sequence(
            seq_id            = seq_id,
            dataset_root      = args.dataset_root,
            output_dir        = args.output_dir,
            processor         = processor,
            model             = model,
            batch_size        = args.batch_size,
            threshold         = args.threshold,
            overlap_threshold = args.overlap_threshold,
            image_subdir      = args.image_subdir,
            use_cuda          = use_cuda,
        )

    print(f"\nAll sequences done. Panoptic maps written to: {args.output_dir}")


if __name__ == "__main__":
    main()
