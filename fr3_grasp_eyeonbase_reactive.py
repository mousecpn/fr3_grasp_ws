#!/usr/bin/env python


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
from dougsm_helpers.ros_control import ControlSwitcher
from geometry_msgs.msg import Twist
from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN, VGN_RVIZ
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.utils.panda_control import PandaCommander
from vgn.detection_implicit import VGNImplicit
from vgn.detection_diff import GIGADiff

from fr3_grasp_eyeonbase import TSDFServer
import sys
sys.path.append('/home/pinhao/icg_benchmark')
from utils_exp.visualization import trimesh_to_open3d

from base_fr3_grasp import BasePandaGraspController
from utils_exp.control import calculate_velocity

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

        
        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                            'velocity': 'cartesian_velocity_node_controller',
                            'joint_velocity': 'joint_velocity_node_controller'})
        
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.joint_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', sensor_msgs.msg.JointState, queue_size=1)

    def run(self):
        if self.cs.current_controller != "moveit":
            self.cs.switch_controller("moveit")

        vis.clear()
        vis.draw_workspace(self.size)
        self.pc.move_gripper(0.08)
        self.pc.home()

        tsdf, pc = self.acquire_tsdf()

        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))#+np.array([self.cali_offset]))
        rospy.loginfo("Reconstructed scene")
        self.pc.home(velocity_scaling=0.8, acceleration_scaling=0.5)

        ### debug ####
        tsdf = np.rot90(tsdf.get_grid(), k=-1, axes=(1,2))
        ### debug ####
        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)

        # tsdf_pcl = (visualize_tsdf(tsdf.squeeze(), viz=False)+0.5) * 0.3
        # grasp_mesh_list = [trimesh_to_open3d(grasp2mesh(g, scores)) for g in grasps]
        # visualize_point_cloud_with_normals(tsdf_pcl, None, grasp_mesh_list)
        
        ### debug ####
        grasps = self.rotate_grasp(grasps)
        ### debug ####

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
        vis.draw_grasp(grasp, score, self.finger_depth)
        # vis.draw_grasp(grasp, 0.99, self.finger_depth)
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
    
    def goto_pose_reactive(self, target_pose, Gain=0.1, threshold=0.001):
        if self.cs.current_controller != "joint_velocity":
            self.cs.switch_controller("joint_velocity")
        arrived = False
        watch_dog = 0
        while True:
            
            robot_state_joint = self.joints_state
            joint_vel, arrived = calculate_velocity(self.panda, np.array(robot_state_joint), target_pose, 
                                                    obstacles=None, Gain=Gain, threshold=threshold)
            joint_vel_norm = np.linalg.norm(np.array(joint_vel))
            if joint_vel_norm < 0.2:
                watch_dog += 1
            if arrived is True or self.robot_error or watch_dog > 1000:
                break
            # time.sleep(0.04)
            self.joint_command_msg.velocity = joint_vel
        self.joint_command_msg.velocity = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
        return arrived
    
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

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_base_pregrasp = T_base_grasp * self.T_grasp_pregrasp
        T_base_retreat = T_base_grasp * self.T_grasp_retreat

        # self.goto_pose_reactive(T_base_pregrasp * self.T_tcp_tool0, threshold=0.01)
        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
        # self.pc.move_gripper(grasp.width)

        if self.robot_error:
            self.recover_robot()
            # self.pc.home()
            return False
        
        self.clear_octomap.call()
        rospy.sleep(0.05)
        
        # self.approach_grasp(T_base_grasp)
        self.pc.goto_pose(T_base_grasp* self.T_tcp_tool0, velocity_scaling=0.2)

        if self.robot_error:
            self.recover_robot()
            
        # self.cs.switch_controller('moveit')
        self.pc.grasp(width=0.0, force=30.0, speed=0.05)

        if self.robot_error:
            self.recover_robot()
            return False

        # self.goto_pose_reactive(T_base_retreat * self.T_tcp_tool0, threshold=0.05)
        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0, velocity_scaling=0.2)

        if self.robot_error:
            self.recover_robot()
    
        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.13])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        # self.goto_pose_reactive(T_base_lift * self.T_tcp_tool0, threshold=0.05)
        # self.cs.switch_controller('moveit')
        self.pc.goto_pose(T_base_lift * self.T_tcp_tool0, velocity_scaling=0.2)

        if self.gripper_width > 0.001:
            return True
        else:
            return False
        
    # def execute_grasp(self, grasp):
    #     T_task_grasp = grasp.pose
    #     T_base_grasp = self.T_base_task * T_task_grasp

    #     T_base_pregrasp = T_base_grasp * self.T_grasp_pregrasp
    #     T_base_retreat = T_base_grasp * self.T_grasp_retreat

    #     self.goto_pose_reactive(T_base_pregrasp * self.T_tcp_tool0, threshold=0.01)
    #     # self.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
    #     # self.pc.move_gripper(grasp.width)

    #     if self.robot_error:
    #         self.recover_robot()
    #         # self.pc.home()
    #         return False
        
    #     self.clear_octomap.call()
    #     rospy.sleep(0.05)
        
    #     self.approach_grasp(T_base_grasp)

    #     if self.robot_error:
    #         self.recover_robot()
            
    #     self.cs.switch_controller('moveit')
    #     self.pc.grasp(width=0.0, force=30.0)

    #     if self.robot_error:
    #         self.recover_robot()
    #         return False

    #     self.goto_pose_reactive(T_base_retreat * self.T_tcp_tool0, threshold=0.05)

    #     if self.robot_error:
    #         self.recover_robot()
    
    #     # lift hand
    #     T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.13])
    #     T_base_lift = T_retreat_lift_base * T_base_retreat
    #     self.goto_pose_reactive(T_base_lift * self.T_tcp_tool0, threshold=0.05)
    #     self.cs.switch_controller('moveit')

    #     if self.gripper_width > 0.001:
    #         return True
    #     else:
    #         return False

    def approach_grasp(self, T_base_grasp):
        self.goto_pose_reactive(T_base_grasp * self.T_tcp_tool0, Gain=0.05, threshold=0.005)

    def drop(self):
        # self.pc.goto_joints(
        #     [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        # )
        self.pc.goto_joints(self.drop_joints, 0.2, 0.2)
        self.pc.move_gripper(0.08)
        

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
    parser.add_argument("--model_name", type=str, default="igd")
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/realexp_packed_igd/vgn_giga_diff_53832.pt"
    # packed_path = "/home/pinhao/Desktop/GIGA/data/runs/24-03-21-18-54_dataset=data_packed_train_processed4,augment=False,net=giga_diff,batch_size=128,lr=2e-04/vgn_giga_diff_53832.pt"
    # parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/65_61|51_71.pt") # Pile
    parser.add_argument("--model", type=Path, default="/home/pinhao/Desktop/GIGA/data/runs/packed/vgn_giga_diff_107664.pt") # packed
    # parser.add_argument("--model", type=Path, default=packed_path) # packed


    
    # parser.add_argument("--model", type=Path, default="/home/pinhao/vgn/data/models/vgn_conv.pth")
    args = parser.parse_args()
    main(args)

    # # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5