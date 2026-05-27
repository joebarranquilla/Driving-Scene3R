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
    #example scene
    scene = trimesh.load("glbscene_All_camTrue_meshTrue_blackFalse_whiteFalse.glb")
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.dump())
    else:
        mesh = scene
    vertices = mesh.vertices
    return vertices


def select_subscene(points, center=None, radius=None, fraction=0.2):
    """Select a small part of the scene.

    If `center` is None, use the global centroid. If `radius` is None,
    choose `fraction * max_extent` of the scene as radius.
    Returns indices and the selected points.
    """
    if center is None:
        center = points.mean(axis=0)
    if radius is None:
        extent = points.max(axis=0) - points.min(axis=0)
        radius = fraction * max(extent)
    # Euclidean distance selection
    dists = np.linalg.norm(points - center[None, :], axis=1)
    mask = dists <= radius
    indices = np.nonzero(mask)[0]
    return indices, points[indices]

points = get_point_cloud()

# Select a small part of the scene (subscene) to place the mesh into
sub_indices, sub_points = select_subscene(points, fraction=0.05)
centroid_sub, axes_sub, extents_sub = fit_oriented_bbox(sub_points)

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

mesh_min = mesh_points.min(axis=0)
mesh_max = mesh_points.max(axis=0)
mesh_size = mesh_max - mesh_min
mesh_center = (mesh_min + mesh_max) / 2

# Compute transform to fit the mesh into the selected subscene
scale_sub = extents_sub / mesh_size
points_centered = mesh_points - mesh_center
points_scaled = points_centered * scale_sub
points_rotated = points_scaled @ axes_sub
points_world_sub = points_rotated + centroid_sub 

#add the transformed mesh points with different color to the original point cloud and save as a new PLY file
combined_points = np.vstack((points, points_world_sub))
mesh_colors = np.array(color_data)
if mesh_colors.max() <= 1.0:
    mesh_colors = (mesh_colors * 255).astype(np.uint8)
else:
    mesh_colors = mesh_colors.astype(np.uint8)
combined_colors = np.vstack((np.full((points.shape[0], 3), [0, 0, 255], dtype=np.uint8), mesh_colors))  # Original points in blue, transformed mesh in mesh colors
# Create a structured array for PLY format
vertex_data = np.array([(combined_points[i, 0], combined_points[i, 1], combined_points[i, 2], combined_colors[i, 0], combined_colors[i, 1], combined_colors[i, 2]) for i in range(combined_points.shape[0])],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
ply_element = PlyElement.describe(vertex_data, 'vertex')
ply_data = PlyData([ply_element])
ply_data.write('combined_point_cloud.ply')
print("Combined point cloud saved to combined_point_cloud.ply")


