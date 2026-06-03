import numpy as np
from plyfile import PlyData, PlyElement

def get_point_cloud():
    #example scene as .ply file
    ply_data = PlyData.read('../semantic_clouds/seq00_static3.ply')
    points = np.vstack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).T
    colors = np.vstack((ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue'])).T
    return points, colors

points, colors_orig = get_point_cloud()

vertices = []
colors = []
#read all the files in folder ~/semantic_clouds/seq00_static_dynamic
import os
for filename in os.listdir('../semantic_clouds/seq00_static_dynamic'):
    if filename.endswith('.ply'):
        ply_data = PlyData.read(os.path.join('../semantic_clouds/seq00_static_dynamic', filename))
        vertices.append(np.vstack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).T)
        colors.append(np.vstack((ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue'])).T)

# Combine per-file vertex arrays into a single [M,3] array (handle empty case)
if len(vertices) == 0:
    mesh_points = np.empty((0, 3), dtype=np.float32)
else:
    mesh_points = np.vstack(vertices)

# Combine per-file color arrays into a single [M,3] array
if len(colors) == 0:
    mesh_colors = np.empty((0, 3), dtype=np.float32)
else:
    mesh_colors = np.vstack(colors)

# Add the transformed mesh points to the original point cloud and save as a new PLY file using original colors
combined_points = np.vstack((points, mesh_points))  # mesh_points already [N,3]

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


