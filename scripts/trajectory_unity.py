from plyfile import PlyData
import numpy as np
import json

def smooth_outliers(trajectory, threshold=1.0):
    trajectory = np.array(trajectory)
    mean = np.mean(trajectory)
    std = np.std(trajectory)

    mask = np.abs((trajectory - mean) / std) > 3

    trajectory[mask] = np.nan

    indices = np.arange(len(trajectory))
    trajectory = np.interp(
        indices,
        indices[~np.isnan(trajectory)],
        trajectory[~np.isnan(trajectory)]
    )
    window = 9

    kernel = np.ones(window) / window
    trajectory = np.convolve(trajectory, kernel, mode="same")
    return trajectory

def ema_filter(data, alpha=0.05):
    
    data = np.asarray(data, dtype=float)

    filtered = np.zeros_like(data)
    filtered[0] = data[0]

    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]

    return filtered

def deadband_filter(data, threshold=0.1):

    data = np.asarray(data, dtype=float)

    filtered = np.zeros_like(data)
    filtered[0] = data[0]

    for i in range(1, len(data)):
        if abs(data[i] - filtered[i-1]) > threshold:
            filtered[i] = data[i]
        else:
            filtered[i] = filtered[i-1]

    return filtered

world_ply = PlyData.read('../gaussians.ply')
wv = world_ply['vertex']
world_xyz = np.stack([wv['x'], wv['y'], wv['z']], axis=-1)
#read the target from the JSON file
new_targets = []
with open("optimal_trajectory.json", 'r') as f:
    json_data = json.load(f)
    traj = json_data.get('trajectory')
    if traj is None:
        raise KeyError('trajectory field not found in JSON')

    xs = traj.get('x')
    ys = traj.get('y')
    zs = traj.get('z')    

    for i in range(len(xs)):
        target = np.array([xs[i], ys[i], zs[i]])
        t2 = np.array([xs[i], ys[i], zs[i]])
        try:
            from scipy.spatial import cKDTree
            #print("scipy available, using cKDTree for nearest neighbor search")
            tree = cKDTree(world_xyz[:, [0, 2]])
            k = min(64, len(world_xyz))
            _, idxs = tree.query(np.array([[t2[0], t2[2]]]), k=k, workers=-1)
            idxs = idxs[0] if k > 1 else np.atleast_1d(idxs[0])
        except Exception:
            print("scipy not available, falling back to numpy")
            dists = np.linalg.norm(world_xyz[:, [0, 2]] - np.array([t2[0], t2[2]]), axis=1)
            k = min(8, len(world_xyz))
            idxs = np.argsort(dists)[:k]

        local_world_ys = world_xyz[idxs, 1]
        local_world_y = float(np.median(local_world_ys))
        target[1] = -(local_world_y)
        target[1] += 0.5
        new_targets.append(target)

xs = [t[0] for t in new_targets]
ys = [t[1] for t in new_targets]
zs = [t[2] for t in new_targets]
smoothed_y = smooth_outliers(ys, threshold=1.0)
smoothed_x = deadband_filter(xs, threshold=0.3)
smoothed_x = ema_filter(smoothed_x, alpha=0.02)

with open("smoothed_targets.json", "w") as f:
    json.dump({"trajectory": {"x": smoothed_x.tolist(), "y": smoothed_y.tolist(), "z": zs}}, f, indent=4)    


