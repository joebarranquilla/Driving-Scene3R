#!/usr/bin/env python3
"""
SAM 3 instance segmentation and tracking for KITTI odometry sequences.

Uses SAM3VideoSemanticPredictor (text-prompted concept segmentation + video
tracking) to produce per-frame instance masks with persistent IDs across frames.
The same car / pedestrian / cyclist keeps the same track_id from frame to frame,
which is what the downstream temporal-accumulation stage requires.

Output per frame:
    {output_dir}/{sequence}/{frame_stem}.npz
        "instance_seg"  → int32   (H, W)   track_id per pixel (0 = background)
        "track_ids"     → int32   (N,)     persistent tracking IDs this frame
        "label_ids"     → int32   (N,)     class index matching --concepts order
        "scores"        → float32 (N,)     detection confidence per instance

A one-time {output_dir}/concepts.json mapping index → concept name is also written.

Resume behaviour: a sequence is skipped only when every expected output frame
already exists.  Because SAM 3 tracking is stateful, a partially-processed
sequence must be rerun from the beginning to keep ID continuity.

Usage
-----
python scripts/run_sam3_inference.py \\
    --dataset_root /storage/group/dataset_mirrors/kitti_odom_color/ \\
data_odometry_color/dataset/sequences \\
    --sequences    00 01 \\
    --checkpoint   /usr/prakt/s0044/checkpoints/sam3/sam3.pt \\
    --output_dir   /usr/prakt/s0044/sam3_predictions

Slurm (long sequences)
----------------------
sbatch --part=PRACT --qos=practical_course --gres=gpu:1 --time=8:00:00 \\
    --wrap="conda run -n driving-scene3r python scripts/run_sam3_inference.py ..."
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# SAM3 predictor adapted for image-directory sources
# ---------------------------------------------------------------------------

def _build_predictor(checkpoint: str, conf: float, device):
    """
    Return a SAM3VideoSemanticPredictor that accepts image directories.

    Ultralytics' SAM3VideoSemanticPredictor asserts dataset.mode == "video"
    and uses dataset.frame (which never increments for image sources) to index
    frames.  This subclass fixes both without touching any other behaviour.
    """
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    class _ImageSeqPredictor(SAM3VideoSemanticPredictor):

        @staticmethod
        def init_state(predictor):
            """Accept image-directory sources in addition to video files."""
            if len(predictor.inference_state) > 0:
                return
            assert predictor.dataset is not None
            # dataset.nf = total number of image files (image mode)
            # dataset.frames[0] = total frame count (video mode)
            if predictor.dataset.mode == "video":
                num_frames = predictor.dataset.frames[0]
            else:
                num_frames = predictor.dataset.nf
            predictor.inference_state = {
                "num_frames": num_frames,
                "tracker_inference_states": [],
                "tracker_metadata": {},
                "text_prompt": None,
                "per_frame_geometric_prompt": [None] * num_frames,
            }

        def inference(self, im, bboxes=None, labels=None, text=None, *args, **kwargs):
            """Use dataset.count for image sources (dataset.frame stays 0)."""
            if self.dataset.mode == "video":
                frame = self.dataset.frame - 1
            else:
                frame = self.dataset.count - 1   # count is 1-based at call time
            self.inference_state["im"] = im
            if "text_ids" not in self.inference_state:
                self.add_prompt(frame_idx=frame, text=text, bboxes=bboxes, labels=labels)
            return self._run_single_frame_inference(frame, reverse=False)

    predictor = _ImageSeqPredictor(overrides=dict(
        model   = checkpoint,
        task    = "segment",
        mode    = "predict",
        device  = device,
        conf    = conf,
        verbose = False,
    ))
    return predictor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM 3 video instance segmentation on KITTI odometry sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root directory that contains numbered KITTI sequence folders.",
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True, metavar="SEQ",
        help="One or more sequence IDs, e.g. --sequences 00 01 02",
    )
    parser.add_argument(
        "--checkpoint",
        default="/usr/prakt/s0044/checkpoints/sam3/sam3.pt",
        help="Path to SAM 3 checkpoint (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        default="/usr/prakt/s0044/sam3_predictions",
        help="Root directory for output NPZ files.",
    )
    parser.add_argument(
        "--concepts", nargs="+",
        default=["car", "pedestrian", "cyclist"],
        help="Text concept prompts — defines the label_id order (0, 1, 2, …).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--image_subdir", default="image_2",
        help="Sub-directory inside each sequence folder that holds PNG frames.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true",
        help="Disable CUDA and run on CPU (very slow for full sequences).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_frames(image_dir: str) -> list[str]:
    """Return sorted absolute paths to all PNG files in *image_dir*."""
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in: {image_dir}")
    return [os.path.join(image_dir, f) for f in files]


def build_instance_seg(
    masks: np.ndarray,      # bool (N, H, W)
    track_ids: np.ndarray,  # int32 (N,)
) -> np.ndarray:
    """Rasterise per-instance binary masks into a single int32 (H, W) map.

    Pixels belonging to multiple instances keep the last mask's ID (conflicts
    are rare — SAM 3 applies non-overlapping constraints internally).
    """
    seg = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int32)
    for mask, tid in zip(masks, track_ids):
        seg[mask] = int(tid)
    return seg


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_id: str,
    dataset_root: str,
    output_dir: str,
    predictor,
    concepts: list[str],
    conf: float,
    image_subdir: str,
) -> None:
    seq_dir   = os.path.join(dataset_root, seq_id)
    image_dir = os.path.join(seq_dir, image_subdir)

    for p in [seq_dir, image_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path not found: {p}")

    frame_paths = collect_frames(image_dir)
    out_seq_dir = os.path.join(output_dir, seq_id)
    os.makedirs(out_seq_dir, exist_ok=True)

    stems   = [os.path.splitext(os.path.basename(fp))[0] for fp in frame_paths]
    missing = [s for s in stems if not os.path.exists(os.path.join(out_seq_dir, f"{s}.npz"))]

    if not missing:
        tqdm.write(f"  All {len(frame_paths)} frames already saved — skipping.")
        return

    if len(missing) < len(frame_paths):
        tqdm.write(
            f"  {len(frame_paths) - len(missing)} / {len(frame_paths)} frames exist "
            f"but tracking requires processing from frame 0 — rerunning full sequence."
        )

    tqdm.write(f"  Frames: {len(frame_paths)}  |  concepts: {concepts}\n")

    # Reset tracking state between sequences
    predictor.inference_state = {}

    results_iter = predictor(
        source=image_dir,
        text=concepts,
        stream=True,
        conf=conf,
    )

    for fp, result in tqdm(
        zip(frame_paths, results_iter),
        total=len(frame_paths),
        desc=f"seq {seq_id}",
        unit="frame",
        dynamic_ncols=True,
    ):
        stem     = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_seq_dir, f"{stem}.npz")

        if result.masks is not None and len(result.masks) > 0:
            masks_np  = result.masks.data.cpu().numpy().astype(bool)   # (N, H, W)
            boxes_np  = result.boxes.data.cpu().numpy()                # (N, 7)
            # boxes columns: x1 y1 x2 y2 track_id score cls
            track_ids = boxes_np[:, 4].astype(np.int32)
            scores    = boxes_np[:, 5].astype(np.float32)
            label_ids = boxes_np[:, 6].astype(np.int32)
            inst_seg  = build_instance_seg(masks_np, track_ids)
        else:
            from PIL import Image
            with Image.open(fp) as img:
                h, w = img.height, img.width
            inst_seg  = np.zeros((h, w), dtype=np.int32)
            track_ids = np.array([], dtype=np.int32)
            label_ids = np.array([], dtype=np.int32)
            scores    = np.array([], dtype=np.float32)

        np.savez_compressed(
            out_path,
            instance_seg = inst_seg,
            track_ids    = track_ids,
            label_ids    = label_ids,
            scores       = scores,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = 0 if use_cuda else "cpu"
    if not use_cuda:
        print(
            "WARNING: CUDA not available or disabled. "
            "CPU inference will be very slow for long KITTI sequences.\n"
        )

    print("Loading SAM 3 model …")
    predictor = _build_predictor(args.checkpoint, args.conf, device)
    print("Model loaded.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    concept_map_path = os.path.join(args.output_dir, "concepts.json")
    if not os.path.exists(concept_map_path):
        with open(concept_map_path, "w") as fh:
            json.dump({str(i): c for i, c in enumerate(args.concepts)}, fh, indent=2)
        print(f"Concept map saved: {concept_map_path}\n")

    for seq_id in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"  Sequence: {seq_id}")
        print(f"{'=' * 60}")
        process_sequence(
            seq_id       = seq_id,
            dataset_root = args.dataset_root,
            output_dir   = args.output_dir,
            predictor    = predictor,
            concepts     = args.concepts,
            conf         = args.conf,
            image_subdir = args.image_subdir,
        )

    print(f"\nAll sequences done. Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
