import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
import argparse
import json


def load_P_from_calib(calib_path: str, key: str = "P2") -> np.ndarray:
    """Read KITTI-style calib.txt and return camera matrix P (3x4) for given key.

    Returns intrinsic K (3x3) extracted from P.
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(calib_path)
    with open(calib_path, 'r') as fh:
        for line in fh:
            if line.startswith(key + ':'):
                parts = line.split()[1:]
                vals = [float(p) for p in parts]
                P = np.array(vals).reshape(3, 4)
                K = P[:, :3]
                return K
    raise KeyError(f"Key {key} not found in {calib_path}")


def backproject_pixels_to_camera(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject pixel coordinates with depths into camera coordinates."""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    pts = np.stack([x_cam, y_cam, zs], axis=1)
    return pts


def transform_points_camera_to_world(points_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform Nx3 points from camera frame to world frame using 4x4 T_wc."""
    Rm = T_wc[:3, :3]
    t = T_wc[:3, 3]
    return (points_cam @ Rm.T) + t


def compute_centroid_of_ply(ply_path: str) -> np.ndarray:
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    data = {name: np.array(vertex[name]) for name in vertex.data.dtype.names}
    xyz = np.stack([data['x'], data['y'], data['z']], axis=-1)
    return xyz.mean(axis=0)


def transform_gaussians(ply_path, translation, rotation_degrees, scale=1.0):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    data = {name: np.array(vertex[name]) for name in vertex.data.dtype.names}
    
    xyz = np.stack([data['x'], data['y'], data['z']], axis=-1)
    
    # 1. Scale and Localize around its own center
    centroid = xyz.mean(axis=0)
    xyz_centered = (xyz - centroid) * scale
    
    # 2. AUTOMATIC CONVENTION ALIGNMENT (Fixes the upside-down issue)
    # Maps TriPoSR (OpenCV space) to the required camera tracking orientation
    # 2. AUTOMATIC CONVENTION ALIGNMENT (Fixes both upside-down and backward facing issues)
    # Maps TriPoSR (OpenCV space) to the required camera tracking orientation
    M_align = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])
    
    # Keep the SciPy object and the raw 3x3 matrix completely separate
    r_scipy = R.from_euler('xyz', rotation_degrees, degrees=True)
    r_custom_matrix = r_scipy.as_matrix()
    
    # Combine the convention alignment matrix with your custom euler rotation matrix
    combined_rotation = r_custom_matrix @ M_align
    
    # Apply the combined mathematically-grounded rotation matrix to the coordinates
    xyz_rotated = np.dot(xyz_centered, combined_rotation.T)
    
    # 3. Translate directly to the target world coordinates
    xyz_transformed = xyz_rotated + np.array(translation)
    
    data['x'] = xyz_transformed[:, 0]
    data['y'] = xyz_transformed[:, 1]
    data['z'] = xyz_transformed[:, 2]
    
    # 4. Scale individual Gaussian radii in log-space (if they exist)
    if 'scale_0' in data and scale != 1.0:
        log_scale_offset = np.log(scale)
        data['scale_0'] += log_scale_offset
        data['scale_1'] += log_scale_offset
        data['scale_2'] += log_scale_offset

    # 5. Update Quaternions (if they exist)
    if 'rot_0' in data:
        q_obj = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=-1)
        r_obj = R.from_quat(q_obj[:, [1, 2, 3, 0]]) # Scipy expects [x, y, z, w]
        
        # ALSO construct the alignment rotation as a SciPy object to match types
        r_align = R.from_matrix(M_align)
        r_global = r_scipy * r_align
        
        # Compose the combined global rotation array against the local Gaussian arrays safely
        r_new = r_global * r_obj
        q_new = r_new.as_quat()[:, [3, 0, 1, 2]] # Convert back to [w, x, y, z]
        
        data['rot_0'] = q_new[:, 0]
        data['rot_1'] = q_new[:, 1]
        data['rot_2'] = q_new[:, 2]
        data['rot_3'] = q_new[:, 3]
    else:
        print("Notice: No 'rot_0' field found in object. Treating as standard Point Cloud/Mesh structural source.")
        
    # 6. Update Spherical Harmonics (if they exist)
    if 'f_rest_0' in data:
        sh_features = np.stack([data[f'f_rest_{i}'] for i in range(45)], axis=-1).reshape(-1, 15, 3)
        
        # Rotates the first-order SH features directly using the raw 3x3 matrix
        sh_features[:, :3, :] = np.dot(sh_features[:, :3, :].transpose(0, 2, 1), combined_rotation.T).transpose(0, 2, 1)
        
        for i in range(45):
            sh_id = i // 3
            ch_id = i % 3
            data[f'f_rest_{i}'] = sh_features[:, sh_id, ch_id]

    return data

def merge_scenes(world_path, object_path, output_path, translation=[0, 0, 0], rotation=[0, 0, 0], scale=1.0):
    """Merges an object PLY into a world PLY, generating explicit visibility defaults."""
    print("Loading world scene...")
    world_ply = PlyData.read(world_path)
    world_vertex = world_ply['vertex']
    world_data = {name: np.array(world_vertex[name]) for name in world_vertex.data.dtype.names}
    
    print("Transforming object scene...")
    obj_data = transform_gaussians(object_path, translation, rotation, scale=scale)

    # Debug reporting
    try:
        obj_xyz = np.stack([obj_data['x'], obj_data['y'], obj_data['z']], axis=-1)
        print("Transformed object centroid:", obj_xyz.mean(axis=0))
        print("Transformed object bbox min,max:", obj_xyz.min(axis=0), obj_xyz.max(axis=0))
    except Exception as e:
        print("Warning: could not process object bounding boxes:", e)

    print("Merging data arrays...")
    merged_data = {}
    n_world = len(world_data['x'])
    n_obj = len(obj_data['x'])

    all_keys = list(dict.fromkeys(list(world_data.keys()) + list(obj_data.keys())))

    for name in all_keys:
        a = world_data.get(name)
        b = obj_data.get(name)

        if b is None:
            # The object is a mesh missing Gaussian properties; assign explicit visibility attributes
            if name == 'opacity':
                filler = np.full(n_obj, 15.0, dtype=a.dtype)         # Solid, non-transparent opacity logit
            elif name in ['scale_0', 'scale_1', 'scale_2']:
                filler = np.full(n_obj, np.log(0.03), dtype=a.dtype) # Explicit tiny dot radius (~3cm radius splat)
            elif name == 'rot_0':
                filler = np.full(n_obj, 1.0, dtype=a.dtype)         # Identity Quaternion W=1
            elif name in ['rot_1', 'rot_2', 'rot_3']:
                filler = np.zeros(n_obj, dtype=a.dtype)             # Identity Quaternion X,Y,Z=0
            else:
                filler = np.zeros(n_obj, dtype=a.dtype)
            merged = np.concatenate([a, filler])
        elif a is None:
            filler = np.zeros(n_world, dtype=b.dtype)
            merged = np.concatenate([filler, b])
        else:
            merged = np.concatenate([a, b])

        merged_data[name] = merged

    types = [(name, merged_data[name].dtype) for name in all_keys]
    merged_elements = np.empty(len(merged_data['x']), dtype=types)

    for name in all_keys:
        merged_elements[name] = merged_data[name]
        
    el = PlyElement.describe(merged_elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Successfully created: {output_path}")

def place_object_from_mask(
    world_path: str,
    object_path: str,
    output_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    rotation_deg=[0.0, 0.0, 0.0],
):
    """Calculates world target positioning, scales object dynamically, and merges."""
    pts_cam = backproject_pixels_to_camera(xs.astype(float), ys.astype(float), zs.astype(float), K)
    pts_world = transform_points_camera_to_world(pts_cam, T_wc)
    target_centroid = pts_world.mean(axis=0)

    # Calculate dynamic scaling based on target size (e.g., 4.5 meters for a standard car)
    raw_obj_centroid = compute_centroid_of_ply(object_path)
    plydata = PlyData.read(object_path)
    v = plydata['vertex']
    raw_xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
    extents = raw_xyz.max(axis=0) - raw_xyz.min(axis=0)
    longest_side = np.max(extents)
    
    target_car_size = 4.5 
    scale_factor = target_car_size / longest_side
    print(f"-> Target World Center: {target_centroid}")
    print(f"-> Calculated dynamic scale factor: {scale_factor:.4f} (Raw: {longest_side:.4f}m -> Target: {target_car_size}m)")

    # Execute merge with dynamic scaling configurations
    merge_scenes(
        world_path, 
        object_path, 
        output_path, 
        translation=target_centroid.tolist(), 
        rotation=rotation_deg, 
        scale=scale_factor
    )


def debug_and_visualize(xs, ys, zs, K, T_wc, object_ply, img_path=None, out_img="debug_reproj.png"):
    pts_cam = backproject_pixels_to_camera(xs.astype(float), ys.astype(float), zs.astype(float), K)
    pts_world = transform_points_camera_to_world(pts_cam, T_wc)
    target_centroid = pts_world.mean(axis=0)

    obj_centroid = compute_centroid_of_ply(object_ply)
    translation = target_centroid - obj_centroid

    print("--- DEBUG PLACEMENT ---")
    print("K:\n", K)
    print("Object centroid (obj frame):", obj_centroid)
    print("Target centroid (world):", target_centroid)
    print("Computed translation (world):", translation)

    Rm = T_wc[:3, :3]
    t = T_wc[:3, 3]
    Xc = (target_centroid - t) @ Rm
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_reproj = fx * (Xc[0] / Xc[2]) + cx
    v_reproj = fy * (Xc[1] / Xc[2]) + cy

    u_mean = xs.mean() if len(xs) > 0 else float('nan')
    v_mean = ys.mean() if len(ys) > 0 else float('nan')
    err = np.hypot(u_reproj - u_mean, v_reproj - v_mean)
    print(f"Mask centroid (u,v): ({u_mean:.1f}, {v_mean:.1f})")
    print(f"Reprojected centroid (u,v): ({u_reproj:.1f}, {v_reproj:.1f})")
    print(f"Reprojection error (pixels): {err:.2f}")

    if img_path and os.path.exists(img_path):
        from PIL import Image, ImageDraw
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = 6
        draw.ellipse((u_mean - r, v_mean - r, u_mean + r, v_mean + r), outline=(255, 0, 0), width=3)
        draw.ellipse((u_reproj - r, v_reproj - r, u_reproj + r, v_reproj + r), outline=(0, 255, 0), width=3)
        img.save(out_img)
        print(f"Saved debug image: {out_img}")

    # The script can optionally place the object from an optimal_trajectory.json
    # Use the --coords_source=json CLI flag to enable this behaviour.


def place_object_from_trajectory_json(world_path: str, object_path: str, output_path: str, json_path: str, rotation_deg=[0.0, 0.0, 0.0]):
    """Place the object at the chosen (x,y,z) step from an optimal_trajectory.json file

    and automatically apply the matching heading rotation (theta).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, 'r') as fh:
        jd = json.load(fh)

    traj = jd.get('trajectory')
    if traj is None:
        raise KeyError('trajectory field not found in JSON')

    xs = traj.get('x')
    ys = traj.get('y')
    zs = traj.get('z')
    thetas = traj.get('theta')  # <--- Extract the theta array
    
    if xs is None or ys is None or zs is None or thetas is None:
        raise KeyError('trajectory must contain x, y, z, and theta arrays')

    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        raise ValueError('Empty trajectory arrays in JSON')

    # Choose trajectory index (matching your original step selection)
    chosen_idx = 10 if len(xs) > 10 else 0
    target = np.array([float(xs[chosen_idx]), float(ys[chosen_idx]), float(zs[chosen_idx])], dtype=np.float32)

    # --- NEW: Extract theta for the chosen step and convert to degrees ---
    theta_rad = float(thetas[chosen_idx])
    theta_deg = np.degrees(theta_rad) 
    
    # Map to the Y axis slot for the 'xyz' euler sequence
    rotation_deg = [0.0, theta_deg, 0.0]
    print(f"-> Extracted heading rotation: {theta_rad:.4f} rad ({theta_deg:.2f} deg) at step {chosen_idx}")

    # Snap Y to the local road surface so the object is always on/above the street.
    try:
        world_ply = PlyData.read(world_path)
        wv = world_ply['vertex']
        world_xyz = np.stack([wv['x'], wv['y'], wv['z']], axis=-1)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(world_xyz[:, [0, 2]])
            k = min(8, len(world_xyz))
            _, idxs = tree.query(np.array([[target[0], target[2]]]), k=k, workers=-1)
            idxs = idxs[0] if k > 1 else np.atleast_1d(idxs[0])
        except Exception:
            dists = np.linalg.norm(world_xyz[:, [0, 2]] - np.array([target[0], target[2]]), axis=1)
            k = min(8, len(world_xyz))
            idxs = np.argsort(dists)[:k]

        local_world_ys = world_xyz[idxs, 1]
        local_world_y = float(np.mean(local_world_ys))
        if target[1] + 1.2 > local_world_y:
            print(f"    [info] Adjusting JSON target Y {target[1]:.4f} -> {local_world_y:.4f} to sit on/above road surface")
            target[1] = local_world_y - 1.2  
    except Exception as e:
        print(f"    [warn] Could not snap target to local road height: {e}")

    # Compute dynamic scaling similarly to mask-based placement
    raw_obj_centroid = compute_centroid_of_ply(object_path)
    plydata = PlyData.read(object_path)
    v = plydata['vertex']
    raw_xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
    extents = raw_xyz.max(axis=0) - raw_xyz.min(axis=0)
    longest_side = np.max(extents)

    target_car_size = 4.5
    scale_factor = target_car_size / longest_side if longest_side > 0 else 1.0

    print(f"-> Placing object at JSON point: {target}")
    print(f"-> Calculated dynamic scale factor: {scale_factor:.4f}")

    # Execute merge with dynamic scaling and automatic rotation configurations
    merge_scenes(
        world_path,
        object_path,
        output_path,
        translation=target.tolist(),
        rotation=rotation_deg,  # <--- Passes the auto-calculated yaw vector
        scale=scale_factor,
    )


def _parse_args_and_run():
    p = argparse.ArgumentParser(description='Merge object into world; choose coords source (csv or json).')
    p.add_argument('--coords_source', choices=['csv', 'json'], default='csv', help='Where to read target coordinates from')
    p.add_argument('--csv_path', default='000000_obj5_xyz.csv', help='CSV file path when using csv source')
    p.add_argument('--json_path', default='optimal_trajectory.json', help='JSON file path when using json source')
    p.add_argument('--world_ply', default='../gaussians.ply')
    p.add_argument('--object_ply', default='../sam-3d-objects/notebook/gaussians/new_mask_seq4_posed.ply')
    p.add_argument('--out_ply', default='merged_with_car.ply')
    p.add_argument('--calib_path', default='/storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences/04/calib.txt')
    args = p.parse_args()

    world_ply = args.world_ply
    object_ply = args.object_ply
    out_ply = args.out_ply

    K = load_P_from_calib(args.calib_path, key='P2')

    if args.coords_source == 'csv':
        data = pd.read_csv(args.csv_path)
        xs = data['u'].values; ys = data['v'].values; zs = data['z'].values
        print(f"Loaded {len(xs)} pixels from CSV: {args.csv_path}")

        # load first pose as before
        pose_file = "/storage/group/dataset_mirrors/kitti_odom_color/data_odometry_color/dataset/sequences/04/04.txt"
        vals = np.loadtxt(pose_file)
        if vals.ndim == 1:
            flat = vals
            if flat.size % 12 == 0:
                mats = flat.reshape(-1, 3, 4)
                T0 = np.eye(4)
                T0[:3, :4] = mats[0]
            elif flat.size % 16 == 0:
                mats = flat.reshape(-1, 4, 4)
                T0 = mats[0]
        else:
            if vals.shape[1] == 12:
                mats = vals.reshape(-1, 3, 4)
                T0 = np.eye(4)
                T0[:3, :4] = mats[0]
            elif vals.shape[1] == 16:
                mats = vals.reshape(-1, 4, 4)
                T0 = mats[0]

        pts_cam = backproject_pixels_to_camera(xs.astype(float), ys.astype(float), zs.astype(float), K)
        pts_w = transform_points_camera_to_world(pts_cam, T0)
        norm1 = np.linalg.norm(pts_w.mean(axis=0))
        try:
            T0_inv = np.linalg.inv(T0)
            pts_w_inv = transform_points_camera_to_world(pts_cam, T0_inv)
            norm2 = np.linalg.norm(pts_w_inv.mean(axis=0))
        except np.linalg.LinAlgError:
            norm2 = np.inf

        if norm2 < norm1:
            T_wc = T0_inv
            print("Inverted pose matrix based on spatial heuristic")
        else:
            T_wc = T0

        img_path = os.path.join(os.path.dirname(pose_file), "image_2", "000000.png")
        debug_and_visualize(xs, ys, zs, K, T_wc, object_ply, img_path=img_path, out_img="debug_reproj.png")
        place_object_from_mask(world_ply, object_ply, out_ply, xs, ys, zs, K, T_wc, rotation_deg=[0,0,0])

    else:
        # json source
        place_object_from_trajectory_json(world_ply, object_ply, out_ply, args.json_path, rotation_deg=[0,0,0])


if __name__ == "__main__":
    _parse_args_and_run()