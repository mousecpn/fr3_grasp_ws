import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import numpy as np


def trimesh_to_open3d(trimesh_mesh):
    """
    Converts a Trimesh object to an Open3D object.

    Args:
        trimesh_mesh (trimesh.Trimesh): A Trimesh object.

    Returns:
        open3d.geometry.TriangleMesh: The corresponding Open3D mesh.
    """
    # Extract vertices and faces from Trimesh
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Create Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Optionally compute normals
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh 


import torch
def visualize_tsdf(tsdf, name=None, viz=True):
    x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), torch.linspace(start=-0.5, end=0.5 - 1.0 / 40, steps=40), indexing='ij')
    anchor_pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).numpy()[0]

    assert tsdf.shape == anchor_pos.shape[:-1]
    mask = (tsdf != 0.0)
    pcl = anchor_pos[mask]
    
    if viz==True:
        alpha = 1-np.abs(tsdf[mask])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pcl[...,0], pcl[...,1], pcl[...,2],alpha=alpha)
        ax.set_title(name)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
    return pcl

def visualize_point_cloud_with_normals(points, normals=None, grasps=None):
    """
    Visualizes a point cloud with normals using Open3D.

    Args:
        points (torch.Tensor): Tensor of shape (n, 3) representing point cloud.
        normals (torch.Tensor, optional): Tensor of shape (n, 3) representing normals. Defaults to None.
    """
    # Convert points to Open3D format
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcl.normals = o3d.utility.Vector3dVector(normals)

    # Visualize the point cloud
    if grasps is None:
        o3d.visualization.draw_geometries([pcl],
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([pcl, ]+ grasps,
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)
