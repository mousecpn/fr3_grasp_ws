"""Default grasp planner used from EdgeGraspNet.

https://haojhuang.github.io/edge_grasp_page/
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius

from icg_benchmark.simulator.io_smi import *
from icg_benchmark.simulator.utility import FarthestSamplerTorch, get_gripper_points_mask, orthognal_grasps
from icg_benchmark.utils.timing.timer import Timer

from .base import GraspPlannerModule
import time

def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(len(depth_imgs)):
        # extrinsic = Transform.from_list(extrinsics[i])
        extrinsic = extrinsics[i]
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


class EdgeGraspPlanner(GraspPlannerModule[o3d.geometry.PointCloud]):
    """Default grasp planner used from EdgeGraspNet.

    https://haojhuang.github.io/edge_grasp_page/
    """

    def __init__(self, model, device="cuda", sample_number=32, confidence_th=0.85, return_best_score=False) -> None:
        super().__init__()
        self.device = device
        self.sample_number = sample_number
        self.grasper = model
        self.confidence_th = confidence_th
        self.return_best_score = return_best_score

    def forward(self, pc):
        if pc is None:
            return None

        pos = np.asarray(pc.points)
        normals = np.asarray(pc.normals)
        pos = torch.from_numpy(pos).to(torch.float32).to(self.device)
        normals = torch.from_numpy(normals).to(torch.float32).to(self.device)

        fps_sample = FarthestSamplerTorch()
        _, sample = fps_sample(pos, self.sample_number)
        sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(self.device)
        sample = torch.unique(sample, sorted=True)
        sample_pos = pos[sample, :]
        radius_p_batch_index = radius(pos, sample_pos, r=0.05, max_num_neighbors=1024)
        radius_p_index = radius_p_batch_index[1, :]
        radius_p_batch = radius_p_batch_index[0, :]
        sample_pos = torch.cat(
            [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
            dim=0,
        )
        sample_copy = sample.clone().unsqueeze(dim=-1)
        sample_index = torch.cat(
            [sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
            dim=0,
        )
        edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
        all_edge_index = torch.arange(0, len(edges)).to(self.device)
        des_pos = pos[radius_p_index, :]
        des_normals = normals[radius_p_index, :]
        relative_pos = des_pos - sample_pos
        relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
        # only record approach vectors with a angle mask
        x_axis = torch.cross(des_normals, relative_pos_normalized)
        x_axis = F.normalize(x_axis, p=2, dim=1)
        valid_edge_approach = torch.cross(x_axis, des_normals)
        valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
        valid_edge_approach = -valid_edge_approach
        up_dot_mask = torch.einsum(
            "ik,k->i",
            valid_edge_approach,
            torch.tensor([0.0, 0.0, 1.0]).to(self.device),
        )
        relative_norm = torch.linalg.norm(relative_pos, dim=-1)
        depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
        geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
        geometry_mask = torch.logical_and(relative_norm < 0.038, geometry_mask)
        depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
        geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)

        if torch.sum(geometry_mask) < 10:
            return None

        pose_candidates = orthognal_grasps(
            geometry_mask,
            depth_proj,
            valid_edge_approach,
            des_normals,
            sample_pos,
        )
        table_grasp_mask = get_gripper_points_mask(pose_candidates, threshold=0.054)
        geometry_mask[geometry_mask.clone()] = table_grasp_mask
        # wether fps
        edge_sample_index = all_edge_index[geometry_mask]
        if len(edge_sample_index) > 0:
            if len(edge_sample_index) > 1500:
                edge_sample_index = edge_sample_index[torch.randperm(len(edge_sample_index))[:1500]]
            edge_sample_index, _ = torch.sort(edge_sample_index)

            data = Data(
                pos=pos,
                normals=normals,
                sample=sample,
                radius_p_index=radius_p_index,
                ball_batch=radius_p_batch,
                ball_edges=edges,
                approaches=valid_edge_approach[edge_sample_index, :],
                reindexes=edge_sample_index,
                relative_pos=relative_pos[edge_sample_index, :],
                depth_proj=depth_proj[edge_sample_index],
            )

            data = data.to(self.device)
            (
                score,
                depth_projection,
                approaches,
                sample_pos,
                des_normals,
            ) = self.grasper.model.act(data)

            n_grasps = 1 if self.return_best_score else min(64, len(score))
            k_score, max_indice = torch.topk(score, k=n_grasps)
            max_score = score[max_indice]
            max_score = F.sigmoid(max_score).cpu().numpy()
            # print('max score', max_score)
            if max_score.any() < self.confidence_th:
                print("no confident on this observation, skip")
                return None

            grasp_mask = torch.ones(len(depth_projection)) > 2.0
            grasp_mask[max_indice] = True
            trans_matrix = orthognal_grasps(
                grasp_mask.to(des_normals.device),
                depth_projection,
                approaches,
                des_normals,
                sample_pos,
            )
            trans_matrix = trans_matrix.cpu().numpy()

            return (trans_matrix, max_score, None)
        return None


class EdgeGraspObservation():
    def __init__(
        self,
        voxelization_size=0.0045,
        lower_bounds=np.array([0, 0, 0]),
        upper_bounds=np.array([0, 0, 1]),
        tsdf_size: float = 0.3,
        tsdf_resolution: int = 180,
        with_table: bool = False,
        # noise_level: float = 0.0008,
        noise_level: float = 0.0,
    ) -> None:
        self.tsdf_resolution = tsdf_resolution
        self.with_table = with_table
        self.noise_level = noise_level
        self.tsdf_size = tsdf_size
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.voxelization_size = voxelization_size


    def __call__(
        self,
        depth_imgs,#: list[np.ndarray],
        intrinsic,#: list[float],
        extrinsics,#: list[list[float]],
        eye,#: np.ndarray,
    ): #-> o3d.geometry.PointCloud | None:

        # Fuse image into tsdf
        # depth_imgs = np.stack([apply_noise(depth_img, 'dex') for depth_img in depth_imgs], axis=0)
        tsdf = create_tsdf(self.tsdf_size, 180, [depth_imgs], intrinsic, [extrinsics])
        pc = tsdf.get_cloud()
            

        # crop surface and borders from point cloud
        if self.with_table:
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                self.lower_bounds - np.array([0, 0, 0.05]), self.upper_bounds
            )
        else:
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower_bounds, self.upper_bounds)


        pc = pc.crop(bounding_box)
        if pc.is_empty():
            return None
        
        # plane_model, inliers = pc.segment_plane(
        #     distance_threshold=0.01,
        #     ransac_n=1,
        #     num_iterations=2
        # )
        # [a, b, c, d] = plane_model

        # pc  = pc.select_by_index(inliers, invert=True)  # 其他物体

        if self.noise_level > 0:
            vertices = np.asarray(pc.points)
            # add gaussian noise 95% confident interval (-1.96,1.96)
            vertices = vertices + np.random.normal(loc=0.0, scale=self.noise_level, size=(len(vertices), 3))
            pc.points = o3d.utility.Vector3dVector(vertices)
            # o3d.visualization.draw_geometries([pc])

        vertices = np.asarray(pc.points)

        if len(vertices) < 100:
            print("point cloud<100, skipping scene")
            return None

        time0 = time.time()
        pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pc.orient_normals_consistent_tangent_plane(30)
        # orient the normals direction
        # convert to o3d convention
        # eye[0] = -eye[0]
        # eye[1] = -eye[1]

        # eye_pc = o3d.geometry.PointCloud()
        # eye_pc.points = o3d.utility.Vector3dVector(eye + np.random.randn(1000,3)*0.001)

        pc.orient_normals_towards_camera_location(camera_location=eye)
        # o3d.visualization.draw_geometries([pc, eye_pc], point_show_normal=True)

        normals = np.asarray(pc.normals)
        vertices = np.asarray(pc.points)
        if len(vertices) < 100:
            print("point cloud<100, skipping scene")
            return None

        pc = pc.voxel_down_sample(voxel_size=self.voxelization_size)
        return pc