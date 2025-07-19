#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg

from utils_exp import vis
from utils_exp import vis
from utils_exp.perception import *
from utils_exp.transform import Rotation, Transform
from utils_exp.grasp import Grasp

import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3
import open3d as o3d
from grasp_planner.edgegrasp import EdgeGraspPlanner, EdgeGraspObservation
from grasp_planner.edgegraspnet import EdgeGrasper
from math import cos, sin

# tag lies on the table in the center of the workspace
# T_base_tag = Transform(Rotation.identity(), [0.49, 0.02, 0.21-0.2])
# T_base_tag = Transform(Rotation.from_quat([-0.014, 0.008, 0.007, 1.000]), [0.342, 0.197, -0.002])
T_base_tag = Transform(Rotation.from_quat([0.00, 0.000, 0.000, 1.000]), [0.342, 0.197, -0.032]) #-0.022

T_base_wall = Transform(Rotation.identity(), [1, 0, 0])
round_id = 0
from observer import PCLServer, TSDFServer

class CameraTest(object):
    def __init__(self, args):
        self.base_frame_id = "fr3_link0"
        # self.tool0_frame_id = "fr3_hand_tcp"
        # self.T_tool0_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, -0.045]})  # TODO
        # self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = 0.05
        self.size = 6.0 * self.finger_depth
        # self.drop_joints = [-0.008975558521233067, 0.37132317732482906, -0.10218697595800416, -1.8458744576171, -0.003763297210831997, 2.181597354133818, 0.4675279828184786]

        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        # self.obs_server = TSDFServer()
        self.obs_server = PCLServer(cam_topic_name='/camera/depth/image_rect_raw',)
        self.grasp_planner = EdgeGraspPlanner(
            EdgeGrasper(device="cuda", root_dir="/home/pinhao/icg_benchmark/data/edge_grasp_net_pretrained_para", load=180),
            confidence_th=0.8,
            return_best_score=False,)

        rospy.loginfo("Ready to take action")
        # self.fig, self.ax = plt.subplots()

    
    def check_achievable(self, grasp_pose):
        Tep = SE3(grasp_pose.as_matrix())
        success = self.panda.ik_LM(Tep)[1]
        return success


    def define_workspace(self):
        self.T_base_task = Transform(
            Rotation.from_quat([0, 0, 0, 1]), [0.311, -0.011, -0.005-0.05]
        )

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    
    def rotate_grasp(self, grasps):
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, - np.pi / 2.0])

        z_offset = 0
        t_augment = np.r_[0.0, 0.0, z_offset]
        T_augment = Transform(R_augment, t_augment)

        T_center = Transform(Rotation.identity(), np.r_[self.size/2, self.size/2, self.size/2])
        T = T_center * T_augment * T_center.inverse()
        for grasp in grasps:
            grasp.pose = T * grasp.pose
        return

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)

        pc = self.acquire_tsdf()
        if pc is None:
            return
        # vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))

        """
        poses = self.grasp_planner(pc)
        if len(poses) > 1:
            poses, scores, _ = poses
        grasps = []
        for i in range(len(poses)):
            pose = poses[i, :, :]
            
            dof_6 = Transform.from_matrix(pose)
            # decompose the quat
            quat = dof_6.rotation.as_quat()
            translation = dof_6.translation
            translation = translation - pose[:3,:3] @ np.array([0, 0, -0.022])
            dof_6.translation = translation
            width = 0.08

            candidate = Grasp(dof_6, width=width)
            grasps.append(candidate)
        grasps = np.asarray(grasps)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")
        """
        
        # tsdf = np.rot90(tsdf.get_grid(), k=-1, axes=(1,2))
        # tsdf_pcl = (visualize_tsdf(tsdf.squeeze(), viz=False)+0.5) * 0.3
        # visualize_point_cloud_with_normals(tsdf_pcl, None, None)
        time.sleep(5)
        # rospy.loginfo("Reconstructed scene")



    def acquire_tsdf(self):
        self.obs_server.reset()
        self.obs_server.integrate = True

        time.sleep(1)

        self.obs_server.integrate = False
        img = self.obs_server.img

        # tsdf = self.obs_server.low_res_tsdf
        pc = self.obs_server.pc

        # pc = self.tsdf_server.high_res_tsdf.get_cloud()
        # img[img>2.5] = 0
        # plt.imshow(img)
        # plt.show()

        # self.tsdf_server.color_img=None
        self.obs_server.img=None
        # self.tsdf_server.combine_depth = None

        # im = self.ax.imshow(img*1000, cmap='viridis', vmin=0, vmax=5.0) # Adjust vmax as needed
        # self.ax.set_title("Depth Image")
        # plt.colorbar(im, label='Depth (meters)') # Add a color bar
        # plt.show(block=False) # Show the plot without blocking the main thread


        return pc




def main(args):
    rospy.init_node("camera_test")
    panda_grasp = CameraTest(args)

    while True:
        panda_grasp.run()

if __name__=='__main__':
    main(None)