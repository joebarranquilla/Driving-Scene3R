#!/usr/bin/env python3
"""
Correct depth predictions after recomputing the baseline.

Reads already-saved depth NPZ files, recalculates depths using the correct
baseline formula (accounting for P2[0,3]), and overwrites them.

Usage:
    python scripts/correct_depth_baseline.py \\
        --dataset_root /storage/.../dataset/sequences \\
        --depth_dir /usr/prakt/<user>/depth_predictions \\
        --sequences 00 01 02
"""

from __future__ import print_function

import argparse
import os
import numpy as np
from tqdm.auto import tqdm


def parse_kitti_calib_correct(calib_path: str):
    """
    Parse KITTI calib.txt and compute the correct stereo baseline.

    Correct formula: B = |P2[0,3] - P3[0,3]| / f
    (accounts for both camera centers, not just the baseline term in P3)
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

    focal_length = float(P2[0, 0])
    # Correct baseline formula
    baseline = float(abs(P2[0, 3] - P3[0, 3]) / focal_length)

    return focal_length, baseline


def correct_sequence(seq_id: str, dataset_root: str, depth_dir: str) -> None:
    seq_calib_path = os.path.join(dataset_root, seq_id, "calib.txt")
    seq_depth_dir  = os.path.join(depth_dir, seq_id)

    if not os.path.isfile(seq_calib_path):
        raise FileNotFoundError(f"Calibration not found: {seq_calib_path}")
    if not os.path.isdir(seq_depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {seq_depth_dir}")

    focal_length, baseline_correct = parse_kitti_calib_correct(seq_calib_path)
    tqdm.write(f"  Correct calib  →  f = {focal_length:.2f} px,  B = {baseline_correct:.4f} m")

    # List all NPZ files
    npz_files = sorted(f for f in os.listdir(seq_depth_dir) if f.endswith(".npz"))
    tqdm.write(f"  Total frames   →  {len(npz_files)}")

    # Recompute depths
    for npz_name in tqdm(npz_files, desc=f"seq {seq_id}", unit="frame", position=0, leave=True, dynamic_ncols=True):
        npz_path = os.path.join(seq_depth_dir, npz_name)
        data = np.load(npz_path)
        old_depth = data["depth"]  # (H, W)

        # The old depth was computed as: old_depth = (f * B_old) / disp
        # We need to find disp, then recompute: new_depth = (f * B_correct) / disp
        # So: disp = (f * B_old) / old_depth
        # But we don't have B_old stored. We need to read it from the calib again.

        # Actually, simpler: we can extract the old baseline from the calib
        # and just rescale: new_depth = old_depth * (B_correct / B_old)

        # But we don't have B_old. Let me use the old formula instead:
        # B_old = |P3[0,3]| / P3[0,0]
        # Then rescale by (B_correct / B_old)

        data_arrays = dict(data)
        P2 = np.fromstring(open(seq_calib_path).readlines()[2].split(":")[1], sep=" ", dtype=np.float64).reshape(3, 4)
        P3 = np.fromstring(open(seq_calib_path).readlines()[3].split(":")[1], sep=" ", dtype=np.float64).reshape(3, 4)
        baseline_old = float(abs(P3[0, 3]) / P3[0, 0])

        # Rescale depth
        corrected_depth = old_depth * (baseline_correct / baseline_old)

        # Overwrite NPZ
        np.savez_compressed(npz_path, depth=corrected_depth.astype(np.float32))

    tqdm.write(f"  Correction factor: {baseline_correct / baseline_old:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Correct depth predictions by recomputing with the correct baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root directory with KITTI sequences (contains calib.txt for each seq).",
    )
    parser.add_argument(
        "--depth_dir", required=True,
        help="Directory containing saved depth predictions (per-sequence subdirs).",
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True,
        metavar="SEQ",
        help="Sequence IDs to correct, e.g.: --sequences 00 01 02",
    )

    args = parser.parse_args()

    for seq_id in args.sequences:
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"  Sequence: {seq_id}")
        tqdm.write(f"{'=' * 60}")
        correct_sequence(seq_id, args.dataset_root, args.depth_dir)

    tqdm.write(f"\nAll sequences corrected.")


if __name__ == "__main__":
    main()
