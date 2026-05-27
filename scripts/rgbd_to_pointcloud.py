"""
RGB-D to 3D Point Cloud
=======================
Converts an RGB image + depth map into a colored 3D point cloud.

Requirements:
    pip install numpy open3d opencv-python

Usage:
    python rgbd_to_pointcloud.py                         # generates synthetic demo data
    python rgbd_to_pointcloud.py --rgb rgb.png --depth depth.png
"""
import numpy as np

# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def rgbd_to_pointcloud(
    rgb: np.ndarray,          # (H, W, 3) uint8
    depth: np.ndarray,        # (H, W)    float32, metres
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,   # multiply raw depth values to get metres
    depth_trunc: float = 10.0,  # discard points farther than this (metres)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    points : (N, 3) float32  – XYZ in camera space (metres)
    colors : (N, 3) float32  – RGB normalised to [0, 1]
    """
    H, W = depth.shape
    depth_m = depth.astype(np.float32) * depth_scale

    # Pixel grid
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))   # (H, W) each

    # Back-project
    Z = depth_m
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy

    # Stack and filter
    xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)           # (H*W, 3)
    rgb_flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0    # (H*W, 3)

    valid = (Z.reshape(-1) > 0) & (Z.reshape(-1) < depth_trunc)
    return xyz[valid], rgb_flat[valid]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: str, depth_scale: float = 1.0) -> np.ndarray:
    """
    Supports:
      • .npz  – auto-detects the right key and unit scale
      • 16-bit PNG (e.g. RealSense, Azure Kinect) – values in mm → depth_scale=0.001
      • 32-bit float EXR / TIFF
      • 8-bit PNG (normalised)
    """
    if path.endswith(".npz"):
        return _load_depth_npz(path)

    import cv2
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if depth is None:
        raise FileNotFoundError(f"Cannot open depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32) * depth_scale


def _load_depth_npz(path: str) -> np.ndarray:
    """
    Auto-detects the depth array key and converts values to metres.

    Key priority:
      1. Any key whose name contains 'depth' (case-insensitive)
      2. First 2-D array found
      3. First array overall

    Unit auto-detection:
      • max value > 100  → assumed millimetres → divide by 1000
      • max value ≤ 100  → assumed metres → keep as-is
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    print(f"[i] .npz keys found: {keys}")

    # 1. Prefer a key with 'depth' in the name
    depth_keys = [k for k in keys if "depth" in k.lower()]
    if depth_keys:
        chosen = depth_keys[0]
    else:
        # 2. First 2-D array
        two_d = [k for k in keys if npz[k].ndim == 2]
        chosen = two_d[0] if two_d else keys[0]

    arr = npz[chosen].astype(np.float32)
    print(f"[i] Using key '{chosen}'  |  shape={arr.shape}  dtype={npz[chosen].dtype}")

    if arr.ndim == 3:
        arr = arr[..., 0]   # drop channel dim if accidentally 3-D

    # Auto-detect units
    max_val = float(np.nanmax(arr))
    if max_val > 100:
        print(f"[i] max depth = {max_val:.1f} → looks like millimetres, converting to metres")
        arr = arr / 1000.0
    else:
        print(f"[i] max depth = {max_val:.3f} → looks like metres, keeping as-is")

    return arr


# ---------------------------------------------------------------------------
# Visualisation (Open3D)
# ---------------------------------------------------------------------------

def save_ply(points: np.ndarray, colors: np.ndarray, path: str = "output.ply") -> None:
        N = len(points)
        cols_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
        with open(path, "w") as f:
            f.write(
                f"ply\nformat ascii 1.0\nelement vertex {N}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            )
            for (x, y, z), (r, g, b) in zip(points, cols_u8):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import numpy as np

def parse_kitti_calib(path):
    calib = {}
    with open(path) as f:
        for line in f:
            key, *vals = line.split()
            calib[key.rstrip(':')] = np.array(vals, dtype=np.float32).reshape(3, 4)
    return calib

def main():
    calib = parse_kitti_calib("../calib.txt")

    P2 = calib["P2"]      
    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]

    rgb   = load_rgb("../000000.png")
    depth = load_depth("../000000.npz")
    H, W  = depth.shape
    depth_scale = 1

    points, colors = rgbd_to_pointcloud(
        rgb, depth, fx, fy, cx, cy,
        depth_scale=depth_scale,
        depth_trunc=80.0,
    )

    save_ply(points, colors, "../output.ply")

if __name__ == "__main__":
    main()
