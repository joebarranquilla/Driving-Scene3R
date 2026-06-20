#!/usr/bin/env python3
"""Extract mask pixels from a SAM3-generated NPZ.

Usage examples:
  python scripts/extract_mask_pixels.py --npz /path/to/frame.npz --by track_id --value 5 --print_coords
  python scripts/extract_mask_pixels.py --npz /path/to/frame.npz --by index --value 0 --out_mask /tmp/mask.png
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Extract mask pixels from SAM3 NPZ outputs")
    p.add_argument("--npz", required=True, help="Path to the .npz file")
    p.add_argument("--by", choices=["track_id", "index", "label"], default="track_id")
    p.add_argument("--value", required=True, help="Integer value for the chosen --by mode")
    p.add_argument("--out_mask", help="Path to save boolean mask as PNG (white mask) ")
    p.add_argument("--print_coords", action="store_true", help="Print (y,x) pixel coordinates to stdout")
    p.add_argument("--save_coords", help="Save pixel coords as a CSV (y,x) file")
    p.add_argument("--depth_npz", help="Optional path to a depth .npz file containing key 'depth' (H,W)")
    p.add_argument("--calib", help="Path to calibration file (KITTI-style). If omitted, repo root 'calib.txt' is used.")
    p.add_argument("--save_xyz", help="Save per-pixel XYZ (camera coordinates) as CSV with columns x,y,z")
    p.add_argument("--save_uvz", help="Save per-pixel U,V,Z (pixel coordinates and depth) as CSV with columns u,v,z (no calib needed)")
    return p.parse_args()


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    return data


def build_mask_from_npz(npz, by: str, value: int):
    if "instance_seg" not in npz:
        raise KeyError("NPZ does not contain 'instance_seg' key. Is this a SAM3 output NPZ?")
    inst = npz["instance_seg"]  # int32 (H, W) storing track_id+1 per pixel

    if by == "track_id":
        tid = int(value)
        mask = inst == (tid + 1)

    elif by == "index":
        if "track_ids" not in npz:
            raise KeyError("NPZ does not contain 'track_ids' array")
        track_ids = npz["track_ids"]
        idx = int(value)
        if idx < 0 or idx >= len(track_ids):
            raise IndexError("index out of range for track_ids")
        tid = int(track_ids[idx])
        mask = inst == (tid + 1)

    elif by == "label":
        if "label_ids" not in npz or "track_ids" not in npz:
            raise KeyError("NPZ missing 'label_ids' or 'track_ids' arrays")
        lbl = int(value)
        label_ids = npz["label_ids"]
        track_ids = npz["track_ids"]
        matching = track_ids[label_ids == lbl]
        mask = np.zeros_like(inst, dtype=bool)
        for t in matching:
            mask |= (inst == (int(t) + 1))

    else:
        raise ValueError("unknown --by mode")

    return mask


def save_mask_png(mask: np.ndarray, out_path: str):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(out_path)


def load_calib_kitti(path: str):
    """Parse KITTI-style calib.txt and return fx, fy, cx, cy from P0 line."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"calib file not found: {path}")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('P0:') or line.startswith('P0 '):
                parts = line.split()[1:]
                vals = [float(v) for v in parts]
                if len(vals) < 9:
                    raise ValueError(f"unexpected P0 line: {line}")
                fx = vals[0]
                fy = vals[4]
                cx = vals[2]
                cy = vals[5]
                return fx, fy, cx, cy
    raise ValueError(f"P0 entry not found in calib file: {path}")


def main():
    args = parse_args()
    npz = load_npz(args.npz)
    mask = build_mask_from_npz(npz, args.by, int(args.value))

    ys, xs = np.where(mask)
    print(f"Mask pixels: {len(ys)}")

    if args.print_coords:
        for y, x in zip(ys, xs):
            print(y, x)

    if args.save_coords:
        coords = np.stack([ys, xs], axis=1)
        np.savetxt(args.save_coords, coords, fmt="%d", delimiter=",")
        print(f"Saved coords to: {args.save_coords}")

    if args.out_mask:
        save_mask_png(mask, args.out_mask)
        print(f"Saved mask PNG to: {args.out_mask}")

    # If depth NPZ and save_uvz requested, save pixel coords + depth (u,v,z)
    if args.depth_npz and args.save_uvz:
        depth_npz = load_npz(args.depth_npz)
        if "depth" not in depth_npz:
            raise KeyError("depth NPZ does not contain 'depth' key")
        depth = depth_npz["depth"]
        if depth.shape != mask.shape:
            raise ValueError(f"depth shape {depth.shape} does not match mask shape {mask.shape}")

        zs = depth[ys, xs].astype(float)
        valid = np.isfinite(zs) & (zs > 0)
        valid_count = int(valid.sum())
        print(f"Valid depth pixels: {valid_count} / {len(zs)}")

        if valid_count == 0:
            print("No valid depth values found for masked pixels; no UVZ saved.")
        else:
            ys_v = ys[valid].astype(int)
            xs_v = xs[valid].astype(int)
            zs_v = zs[valid]

            uvz = np.stack([xs_v, ys_v, zs_v], axis=1)
            np.savetxt(args.save_uvz, uvz, fmt="%.6f", delimiter=",", header="u,v,z", comments='')
            print(f"Saved UVZ to: {args.save_uvz} ({uvz.shape[0]} points)")

    # If depth NPZ and save_xyz requested, compute camera-space XYZ per masked pixel (requires calib)
    if args.depth_npz and args.save_xyz:
        depth_npz = load_npz(args.depth_npz)
        if "depth" not in depth_npz:
            raise KeyError("depth NPZ does not contain 'depth' key")
        depth = depth_npz["depth"]
        if depth.shape != mask.shape:
            raise ValueError(f"depth shape {depth.shape} does not match mask shape {mask.shape}")

        # calibration
        calib_path = args.calib or os.path.join(os.path.dirname(__file__), '..', 'calib.txt')
        fx, fy, cx, cy = load_calib_kitti(calib_path)

        zs = depth[ys, xs].astype(float)
        valid = np.isfinite(zs) & (zs > 0)
        valid_count = int(valid.sum())
        print(f"Valid depth pixels: {valid_count} / {len(zs)}")

        if valid_count == 0:
            print("No valid depth values found for masked pixels; no XYZ saved.")
        else:
            ys_v = ys[valid].astype(float)
            xs_v = xs[valid].astype(float)
            zs_v = zs[valid]

            X = (xs_v - cx) * zs_v / fx
            Y = (ys_v - cy) * zs_v / fy
            Z = zs_v

            pts = np.stack([X, Y, Z], axis=1)
            np.savetxt(args.save_xyz, pts, fmt="%.6f", delimiter=",", header="x,y,z", comments='')
            print(f"Saved XYZ to: {args.save_xyz} ({pts.shape[0]} points)")


if __name__ == "__main__":
    main()
