#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path
import cv2
import cv_bridge
import franka_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg
import sys
# sys.path.append("/home/pinhao/Desktop/DREDS/SwinDRNet")

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN, VGN_RVIZ
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.utils.panda_control import PandaCommander
from vgn.detection_implicit import VGNImplicit
from vgn.detection_diff import GIGADiff
import trimesh
from vgn.utils import visual
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3
from std_srvs.srv import Empty
from std_msgs.msg import Empty as Emptymsg
from observer import TSDFServer, PCLServer
from base_fr3_grasp import BasePandaGraspController

# tag lies on the table in the center of the workspace
# T_base_tag = Transform(Rotation.identity(), [0.49, 0.02, 0.21-0.2])
# T_base_tag = Transform(Rotation.from_quat([-0.014, 0.008, 0.007, 1.000]), [0.342, 0.197, -0.002])
T_base_tag = Transform(Rotation.from_quat([0.00, 0.000, 0.000, 1.000]), [0.342, 0.197, -0.032]) #-0.022

T_base_wall = Transform(Rotation.identity(), [1, 0, 0])
round_id = 0


class PandaGraspController(BasePandaGraspController):
    def __init__(self, args):
        super(PandaGraspController, self).__init__()
        self.tsdf_server = TSDFServer()
        if args.model_name == "vgn":
            self.plan_grasps = VGN_RVIZ(args.model, rviz=True, qual_th=0.7, best=True, force_detection=True)
        elif args.model_name == "giga":
            self.plan_grasps = VGNImplicit(args.model, "giga", best=True, force_detection=True, qual_th=0.8, out_th=0.1)
        elif args.model_name == "igd":
            self.plan_grasps = GIGADiff(args.model, "giga_diff", best=True, force_detection=True, qual_th=0.4, out_th=0.1)

    
    def grasp_detection(self, msg):
        if self.state is None:
            return
        grasps, scores, planning_time = self.plan_grasps(self.state)
        self.grasps = grasps
        self.scores = scores
        return 

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        self.pc.move_gripper(0.08)
        self.pc.home()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points)+np.array([self.cali_offset]))
        rospy.loginfo("Reconstructed scene")


        #### synchronous grasp planning#####
        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        #### synchronous grasp planning#####

        # while self.grasps is None and self.scores is None:
        #     continue
        # grasps, scores = self.grasps, self.scores
        # self.grasps, self.scores, self.state = None, None, None

        # for grasp in grasps:
        #     grasp.pose = self.T_realbase_base * grasp.pose
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")
    
        scores = scores.tolist()

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        # p_cloud_tri = trimesh.points.PointCloud(np.asarray(state.pc.points))
        # grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
        # composed_scene = trimesh.Scene(p_cloud_tri)
        # for i, g_mesh in enumerate(grasp_mesh_list):
        #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        # composed_scene.show()

        # grasp, score = grasps[0], scores[0]
        while True:
            flag = 1
            idx=0
            grasp, score = grasps[0], scores[0]
            # grasp, score, idx = self.select_grasp(grasps, scores)
            if self.check_achievable(self.T_base_task * grasp.pose):
                if self.pc.check_pose_achievable(self.T_base_task * grasp.pose * self.T_grasp_pregrasp):# and self.pc.check_pose_achievable(self.T_base_task * grasp.pose):                      
                    break
                else:
                    rot = grasp.pose.rotation
                    grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
                    if self.pc.check_pose_achievable(self.T_base_task * grasp.pose * self.T_grasp_pregrasp):# and self.pc.check_pose_achievable(self.T_base_task * grasp.pose):
                        break
                    else:
                        flag=0              
            else:
                flag = 0
            if flag == 0:
                grasps.pop(idx)
                scores.pop(idx)
                if len(grasps) == 0:
                    rospy.loginfo("No grasps detected")
                    return
        # grasp, score, idx = self.select_grasp(grasps, scores)
        # vis.draw_grasp(grasp, score, self.finger_depth)
        vis.draw_grasp(grasp, 0.99, self.finger_depth)
        rospy.loginfo("Selected grasp")

        # grasp.pose = self.T_realbase_base * grasp.pose

        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        if self.robot_error:
            self.recover_robot()
            return

        if label:
            self.drop()
        self.pc.home()

    def acquire_tsdf(self):
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        time.sleep(1)

        self.tsdf_server.integrate = False
        img = self.tsdf_server.img

        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        self.tsdf_server.img=None
        # self.tsdf_server.combine_depth = None

        # im = self.ax.imshow(img*1000, cmap='viridis', vmin=0, vmax=5.0) # Adjust vmax as needed
        # self.ax.set_title("Depth Image")
        # plt.colorbar(im, label='Depth (meters)') # Add a color bar
        # plt.show(block=False) # Show the plot without blocking the main thread


        return tsdf, pc

    def select_grasp(self, grasps, scores, topk=3):
        # select the highest grasp
        # grasps, scores = grasps[:topk], scores[:topk]

        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # y_coord = np.empty(len(grasps))
        # for i, grasp in enumerate(grasps):
        #     y_coord[i] = grasp.pose.translation[0]
        # idx = np.argmin(y_coord)
        # grasp, score = grasps[idx], scores[idx]

        # grasp, score = grasps[p[0]], scores[p[0]]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score, idx

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_base_pregrasp = T_base_grasp * self.T_grasp_pregrasp
        T_base_retreat = T_base_grasp * self.T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
        # self.pc.move_gripper(grasp.width)

        if self.robot_error:
            self.recover_robot()
            # self.pc.home()
            return False
        
        self.clear_octomap.call()
        rospy.sleep(0.05)
        
        self.approach_grasp(T_base_grasp)

        if self.robot_error:
            self.recover_robot()

        self.pc.grasp(width=0.0, force=20.0)

        if self.robot_error:
            self.recover_robot()
            return False

        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0)

        if self.robot_error:
            self.recover_robot()

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.13])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift * self.T_tcp_tool0)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0, velocity_scaling=0.05)

    def drop(self):
        # self.pc.goto_joints(
        #     [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        # )
        self.pc.goto_joints(self.drop_joints, 0.2, 0.2)
        self.pc.move_gripper(0.08)



def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    while True:
        panda_grasp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--model", type=Path, required=True)
    ### vgn 
    # parser.add_argument("--model_name", type=str, default="vgn")
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/vgn_packed.pt")
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/vgn_pile.pt")
    
    #### giga #####
    parser.add_argument("--model_name", type=str, default="giga")
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/runs/realexp_packed_giga/vgn_giga_53832.pt")
    parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/giga_random_side_view_packed.pt") # packed
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/giga_pile.pt") # pile

    #### igd ######
    # parser.add_argument("--model_name", type=str, default="igd")
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/realexp_packed_igd/vgn_giga_diff_53832.pt"
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/24-03-26-17-21_dataset=data_packed_train_processed4,augment=False,net=giga_diff,batch_size=128,lr=2e-04/vgn_giga_diff_53832.pt"
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/65_61|51_71.pt") # Pile
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/runs/packed/vgn_giga_diff_107664.pt") # packed
    # parser.add_argument("--model", type=Path, default=packed_path) # packed

    # parser.add_argument("--model_name", type=str, default="gpd")


    # parser.add_argument("--model", type=Path, default="/home/pinhao/vgn/data/models/vgn_conv.pth")
    args = parser.parse_args()
    main(args)
    # # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5