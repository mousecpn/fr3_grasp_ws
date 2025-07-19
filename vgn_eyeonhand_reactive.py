#!/usr/bin/env python


import argparse
from pathlib import Path
import numpy as np
import rospy
import sensor_msgs.msg
import sys
from dougsm_helpers.ros_control import ControlSwitcher
from geometry_msgs.msg import Twist
from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN, VGN_RVIZ
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from utils_exp.panda_control import PandaCommander
from vgn.detection_implicit import VGNImplicit
from vgn.detection_diff import GIGADiff

from threading import Thread, Lock
from fr3_grasp_eyeonbase import TSDFServer
from igd.utils.visual import grasp2mesh
import sys
sys.path.append('/home/pinhao/icg_benchmark')
sys.path.append('/home/pinhao/IGDv2')
from utils_exp.visualization import trimesh_to_open3d
from model.grasp_planner import VGNImplicit as EquiVGNImplicit

from base_fr3_grasp import BasePandaGraspController

class PandaGraspController(BasePandaGraspController):
    def __init__(self, args):
        super(PandaGraspController, self).__init__()
        self.tsdf_server = TSDFServer(cam_topic_name='/camera/depth/image_rect_raw',color_topic_name='/camera/color/image_rect_raw')
        if args.model_name == "vgn":
            self.plan_grasps = VGN_RVIZ(args.model, rviz=True, qual_th=0.7, best=False, force_detection=True)
        elif args.model_name == "giga":
            self.plan_grasps = VGNImplicit(args.model, "giga", best=True, force_detection=True, qual_th=0.3, out_th=0.1)
        elif args.model_name == "igd":
            self.plan_grasps = GIGADiff(args.model, "giga_diff", best=True, force_detection=True, qual_th=0.2, out_th=0.1)
        elif args.model_name == "equi_igd":
            self.plan_grasps = EquiVGNImplicit(args.model, args.model_name, best=True, force_detection=True, qual_th=0.1, out_th=0.1)
        elif args.model_name == "equi_giga":
            self.plan_grasps = EquiVGNImplicit(args.model, args.model_name, best=True, force_detection=True, qual_th=0.1, out_th=0.1)

        
        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                            'velocity': 'cartesian_velocity_node_controller',
                            'joint_velocity': 'joint_velocity_node_controller'})
        
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.joint_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', sensor_msgs.msg.JointState, queue_size=1)

    def run(self):
        if self.robot_error:
            self.recover_robot()

        if self.cs.current_controller != "moveit":
            self.cs.switch_controller("moveit")
        vis.clear()
        vis.draw_workspace(self.size)
        self.pc.move_gripper(0.08)
        self.pc.home()

        if self.robot_error:
            self.recover_robot()
            self.pc.home()

        tsdf, pc = self.acquire_tsdf()

        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(self.T_realbase_base.transform_point(np.asarray(pc.points)))
        rospy.loginfo("Reconstructed scene")
        self.pc.home(velocity_scaling=0.3, acceleration_scaling=0.2)
        # return

        ### debug ####
        # tsdf = np.rot90(tsdf.get_grid(), k=-1, axes=(1,2))
        ### debug ####
        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)

        # tsdf_pcl = (visualize_tsdf(tsdf.squeeze(), viz=False)+0.5) * 0.3
        # grasp_mesh_list = [trimesh_to_open3d(grasp2mesh(g, scores)) for g in grasps]
        # visualize_point_cloud_with_normals(tsdf_pcl, None, grasp_mesh_list)
        
        ### debug ####
        # grasps = self.rotate_grasp(grasps)
        ### debug ####

        for grasp in grasps:
            grasp.pose = self.T_realbase_base * grasp.pose
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        scores = scores.tolist()


        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return
        print("max score:", scores[0])
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
            rot = grasp.pose.rotation
            translation = grasp.pose.translation
            axis = rot.as_matrix()[:, 0]
            if axis[-1] > 0 and translation[-1] < 0.18:
                grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
            # grasp, score, idx = self.select_grasp(grasps, scores)
            if self.check_achievable(self.T_base_task * grasp.pose):
                if self.pc.check_pose_achievable(self.T_base_task * grasp.pose * self.T_grasp_pregrasp):# and self.pc.check_pose_achievable(self.T_base_task * grasp.pose):                      
                    break
                else:
                    flag = 0
                # else:
                #     rot = grasp.pose.rotation
                #     grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
                #     if self.pc.check_pose_achievable(self.T_base_task * grasp.pose * self.T_grasp_pregrasp):# and self.pc.check_pose_achievable(self.T_base_task * grasp.pose):
                #         break
                #     else:
                #         flag=0              
            else:
                flag = 0
            if flag == 0:
                grasps.pop(idx)
                scores.pop(idx)
                if len(grasps) == 0:
                    rospy.loginfo("No grasps detected")
                    return
        # grasp, score, idx = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        print("exe score:", score)
        # vis.draw_grasp(grasp, 0.99, self.finger_depth)
        rospy.loginfo("Selected grasp")


        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")
        self.set_table(0.12)

        if self.robot_error:
            self.recover_robot()

        if label:# or self.gripper_width > 0.001:
            self.drop()
        self.pc.home()
        self.set_table(0.045)

    def acquire_tsdf(self):
        self.tsdf_server.reset()
        T_task_cam = Transform(Rotation.from_quat([-0.866, 0, 0, -0.5]), [-0.15, 0.1616, 0.5200]).inverse()

        # r = 1.7 * self.size
        # theta = np.pi / 4
        # phi = -np.pi/2
        # origin = Transform(
        #     Rotation.identity(),
        #     np.r_[self.size / 2, self.size / 2, 0.0 + 0.25],
        # )
        # T_task_cam = camera_on_sphere(origin, r, theta, phi).inverse()


        T_depth_color = Transform(Rotation.from_quat([0.00, 0.000, 0.00, 1.000]), [-0.015, -0.000, -0.000])
        T_base_cam = self.T_base_task * T_task_cam * T_depth_color
        # T_tcp_cam = Transform(Rotation.from_quat([0.010, 0.003, 0.720, 0.694]), [0.055, -0.028, -0.058])
        # T_base_tcp = T_base_cam * T_tcp_cam.inverse()
        T_tcp_color = Transform(Rotation.from_quat([0.007, 0.000, 0.701, 0.713]), [0.031, -0.021, -0.037])
        T_base_tcp = T_base_cam * T_tcp_color.inverse()

        # self.tf_tree.broadcast_static(T_base_cam, self.base_frame_id, "T_base_cam")
        # self.tf_tree.broadcast_static(T_base_tcp, self.base_frame_id, "T_base_tcp")
        # print()
        self.pc.goto_pose(T_base_tcp, velocity_scaling=0.3, acceleration_scaling=0.2)
        time.sleep(1.5)
        self.tsdf_server.integrate = True

        while self.tsdf_server.img is None:
            time.sleep(2)
        # time.sleep(3)

        # for joint_target in self.scan_joints[1:]:
        #     self.pc.goto_joints(joint_target)

        self.tsdf_server.integrate = False
        img = self.tsdf_server.img
        # while img is None:
        #     img = self.tsdf_server.combine_depth
        
        # time.sleep(1)

        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()
        # img[img>2.5] = 0
        # plt.imshow(img)
        # plt.show()

        # self.tsdf_server.color_img=None
        self.tsdf_server.img=None
        # self.tsdf_server.combine_depth = None


        return tsdf, pc

    
    def joint_velo_publisher(self):
        while True:
            if (np.abs(np.array(self.joint_command_msg.velocity))).sum() > 0 and self.cs.current_controller == "joint_velocity":
                self.joint_command_msg.header.stamp = rospy.Time.now()
                self.joint_velo_pub.publish(self.joint_command_msg)
            time.sleep(0.02)

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


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)
    t2 = Thread(target=panda_grasp.joint_velo_publisher,name='velo_pubilisher')
    t2.start()

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
    # parser.add_argument("--model_name", type=str, default="giga")
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/giga_random_side_view_packed.pt") # packed
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/giga_packed.pt") # packed
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/models/giga_pile.pt") # pile

    #### igd ######
    # parser.add_argument("--model_name", type=str, default="igd")
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/realexp_packed_igd/vgn_giga_diff_53832.pt"
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/24-03-21-18-54_dataset=data_packed_train_processed4,augment=False,net=giga_diff,batch_size=128,lr=2e-04/vgn_giga_diff_53832.pt"
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/65_61|51_71.pt") # Pile
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/runs/packed/vgn_giga_diff_107664.pt") # packed
    # parser.add_argument("--model", type=Path, default=packed_path) # packed


    #### equigiga ######
    # parser.add_argument("--model_name", type=str, default="equi_giga")
    # parser.add_argument("--model", type=Path, default="equigiga_packed.pt") # Pile


    #### equiigd ######
    parser.add_argument("--model_name", type=str, default="equi_igd")
    parser.add_argument("--model", type=Path, default="equiigd_pile.pt") # Pile


    
    # parser.add_argument("--model", type=Path, default="/home/pinhao/vgn/data/models/vgn_conv.pth")
    args = parser.parse_args()
    main(args)

    # # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5