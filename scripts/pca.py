import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from plyfile import PlyData, PlyElement

def fit_oriented_bbox(points_3d):
    # PCA to find principal axes (handles non-axis-aligned cars)
    pca = PCA(n_components=3)
    pca.fit(points_3d)
    centroid = points_3d.mean(axis=0)
    axes = pca.components_          # [3, 3] — rotation matrix
    projected = pca.transform(points_3d)
    extents = projected.max(axis=0) - projected.min(axis=0)  # [3] — real-world dims
    return centroid, axes, extents

def get_point_cloud():
    #example scene as .ply file
    ply_data = PlyData.read('output.ply')
    points = np.vstack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).T
    colors = np.vstack((ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue'])).T
    return points, colors


def select_subscene(points, pixel_coords):
    """Select part of the scene where the pixel coordinates of the 2d bounding box are located.
        Pixel coords in format [x_min, y_min, x_max, y_max]   
    """
    # project 3d points to 2d (assuming a simple pinhole camera model for demonstration)
    # In a real scenario, you would use the actual camera intrinsics and extrinsics
    x_min, y_min, x_max, y_max = pixel_coords
    
    # 1. Real camera intrinsics from P2
    f_x = 718.856000
    f_y = 718.856000
    c_x = 607.192800
    c_y = 185.215700
    
    # Extract 3D coordinates (assuming points is an Nx3 array of X, Y, Z)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    # Avoid division by zero for points right at or behind the camera
    safe_Z = np.where(Z == 0, 1e-6, Z)
    
    # 2. Correct 3D to 2D pinhole projection formulas
    projected_x = (X * f_x / safe_Z) + c_x
    projected_y = (Y * f_y / safe_Z) + c_y
    
    # 3. Create the mask based on bounding box constraints
    mask = (projected_x >= x_min) & (projected_x <= x_max) & \
           (projected_y >= y_min) & (projected_y <= y_max) & \
           (Z > 0)  # Ensure points are actually in front of the camera!
           
    return np.where(mask)[0], points[mask]

points, colors_orig = get_point_cloud()

# Select a small part of the scene (subscene) to place the mesh into
sub_indices, sub_points = select_subscene(points, np.array([429, 240, 672, 361]))

vertices = []
colors = []
faces = []
with open("../TripoSR/output/0/mesh.obj", "r") as f:
    for line in f:
        if line.startswith("v "):
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            color1, color2, color3 = map(float, parts[4:7])  # Assuming colors are in the OBJ file
            vertices.append([x, y, z])
            colors.append([color1, color2, color3])
    for line in open("../TripoSR/output/0/mesh.obj", "r"):
        if line.startswith("f "):
            parts = line.strip().split()
            face = [int(idx.split("/")[0]) - 1 for idx in parts[1:4]]  # OBJ is 1-indexed
            faces.append(face)        

mesh_points = np.array(vertices)  # Only take the position coordinates
color_data = np.array(colors)  # Color data if needed for later use

# ==============================================================================
# 1. Clean up Subscene Points (Filter out the far background street/walls)
# ==============================================================================
# Filter out background outliers: keep only points within a tight depth window
sub_depths = sub_points[:, 2]
median_depth = np.median(sub_depths)

# A car is usually about 4.5 meters long, so let's only keep points within 
# +/- 2.5 meters of the main point cluster
car_mask = (sub_depths >= median_depth - 2.5) & (sub_depths <= median_depth + 2.5)
clean_sub_points = sub_points[car_mask]

# Re-run bbox fit on the cleaned car points
centroid_sub, axes_sub, extents_sub = fit_oriented_bbox(clean_sub_points)

# ==============================================================================
# 2. Scale the Mesh Uniformly using Ground-Truth Car Dimensions
# ==============================================================================
mesh_min = mesh_points.min(axis=0)
mesh_max = mesh_points.max(axis=0)
mesh_size = mesh_max - mesh_min
mesh_center = (mesh_min + mesh_max) / 2

# TripoSR objects are scaled to ~1.0. A real car is ~4.5 meters long.
# If your subscene points are still too sparse or noisy, uncomment 'target_length' 
# to manually override and force a perfect real-world size:
#
# target_length = 4.5  # 4.5 meters long
# scale_factor = target_length / np.max(mesh_size)

# Otherwise, use the cleaned subscene maximum dimension:
scale_factor = np.max(extents_sub) / np.max(mesh_size)

# Optional fine-tuning dial if it still looks a fraction too big or small
manual_multiplier = 0.5  
final_scale = scale_factor * manual_multiplier

points_centered = mesh_points - mesh_center
points_scaled = points_centered * final_scale

# ==============================================================================
# 3. Orient and Position inside the Scene
# ==============================================================================
flip_upside_down = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# Apply the corrective flip FIRST, then rotate into the PCA world axes
points_flipped = points_scaled @ flip_upside_down
points_rotated = points_flipped @ axes_sub
# Start with a copy of the subscene centroid
translation = centroid_sub.copy()

# --- FIX HEIGHT LEVEL ---

# 1. Automatic Alignment: Match the bottom of the mesh to the street level
# (In KITTI, max Y coordinate represents the lowest point/ground level)
subscene_ground_level = np.max(clean_sub_points[:, 1])
mesh_bottom_level = np.max(points_rotated[:, 1])

# Set the Y translation so the car sits exactly on the detected street
translation[1] = subscene_ground_level - mesh_bottom_level

# Shift the flipped and rotated mesh points to the final world position
points_world_sub = points_rotated + translation

#add the transformed mesh points to the original point cloud and save as a new PLY file using original colors
combined_points = np.vstack((points, points_world_sub))
mesh_colors = np.array(colors)

# Normalize/convert original colors to uint8 (handle either 0-1 or 0-255 ranges)
def to_uint8(col_array):
    arr = np.asarray(col_array, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

colors_orig_u8 = to_uint8(colors_orig)
mesh_colors_u8 = to_uint8(mesh_colors)

combined_colors = np.vstack((colors_orig_u8, mesh_colors_u8))

# Create a single vertex element that includes color properties (red, green, blue)
num_vertices = combined_points.shape[0]
vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
vertex_all = np.empty(num_vertices, dtype=vertex_dtype)
vertex_all['x'] = combined_points[:, 0]
vertex_all['y'] = combined_points[:, 1]
vertex_all['z'] = combined_points[:, 2]
# If there are fewer colors than points, fill remaining with white
if combined_colors.shape[0] < num_vertices:
    padded_colors = np.ones((num_vertices, 3), dtype=np.uint8) * 255
    padded_colors[:combined_colors.shape[0], :] = combined_colors
    combined_colors = padded_colors

vertex_all['red'] = combined_colors[:, 0]
vertex_all['green'] = combined_colors[:, 1]
vertex_all['blue'] = combined_colors[:, 2]

ply_el = PlyElement.describe(vertex_all, 'vertex')
PlyData([ply_el], text=True).write('combined_scene.ply')


