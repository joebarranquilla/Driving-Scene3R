import open3d as o3d
import numpy as np
from plyfile import PlyData


def extract_rgb(vertex):
    names = vertex.dtype.names  # ✅ FIX HERE

    # Case 1: direct RGB
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.vstack([
            vertex["red"],
            vertex["green"],
            vertex["blue"]
        ]).T

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        return rgb

    # Case 2: SH DC coefficients
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        C0 = 0.28209479177387814

        rgb = np.vstack([
            vertex["f_dc_0"],
            vertex["f_dc_1"],
            vertex["f_dc_2"]
        ]).T

        rgb = 0.5 + C0 * rgb
        rgb = np.clip(rgb, 0.0, 1.0)

        return rgb

    else:
        return None


def gaussian_ply_to_pointcloud(input_path, output_path):
    ply = PlyData.read(input_path)
    vertex = ply["vertex"].data

    # positions
    points = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).T

    #print the length of the longest side of the bounding box
    bbox = np.array([
        [points[:, 0].min(), points[:, 1].min(), points[:, 2].min()],
        [points[:, 0].max(), points[:, 1].max(), points[:, 2].max()]
    ])
    bbox_size = np.linalg.norm(bbox[1] - bbox[0])
    print(f"Bounding box size: {bbox_size:.2f} units")

    # colors
    colors = extract_rgb(vertex)

    # create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # save
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

    print(f"Saved point cloud to: {output_path}")
    print(f"Points: {len(points)}")
    print(f"Colors preserved: {'Yes' if colors is not None else 'No'}")


if __name__ == "__main__":
    input_ply = "/usr/prakt/s0043/sam-3d-objects/notebook/gaussians/multi/new_mask_seq4.ply"
    output_ply = "binary.ply"

    gaussian_ply_to_pointcloud(input_ply, output_ply)
