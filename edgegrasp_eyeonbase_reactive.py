#!/usr/bin/env python

import numpy as np
import rospy
import sensor_msgs.msg
import sys
from dougsm_helpers.ros_control import ControlSwitcher
from geometry_msgs.msg import Twist
from utils_exp import vis
from utils_exp.perception import *
from utils_exp.transform import Rotation, Transform
from grasp_planner.edgegrasp import EdgeGraspPlanner, EdgeGraspObservation
from grasp_planner.edgegraspnet import EdgeGrasper
from utils_exp.grasp import Grasp

from threading import Thread, Lock
from observer import PCLServer
from base_fr3_grasp import BasePandaGraspController
import sys
sys.path.append('/home/pinhao/icg_benchmark')

from icg_benchmark.models.edge_grasp.edge_grasper import EdgeGrasper

T_base_wall = Transform(Rotation.identity(), [1, 0, 0])
round_id = 0
from utils_exp.control import calculate_velocity


class PandaGraspController(BasePandaGraspController):
    def __init__(self,):
        super(PandaGraspController, self).__init__()
        self.obs_server = PCLServer()
        self.grasp_planner = EdgeGraspPlanner(
            EdgeGrasper(device="cuda", root_dir="/home/pinhao/icg_benchmark/data/edge_grasp_net_pretrained_para", load=180),
            confidence_th=0.8,
            return_best_score=False,)
        
        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                            'velocity': 'cartesian_velocity_node_controller',
                            'joint_velocity': 'joint_velocity_node_controller'})
        
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.joint_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', sensor_msgs.msg.JointState, queue_size=1)
        
        
        self.obs_list = []


    def run(self):
        # self.cs.switch_controller('moveit')
        self.pc.home()
        self.pc.move_gripper(0.08)
        vis.clear()
        vis.draw_workspace(self.size)

        pc = self.acquire_tsdf()
        if pc is None:
            return
        # vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))

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
            translation = translation - pose[:3,:3] @ np.array([0, 0, -0.037])
            dof_6.translation = translation
            width = 0.08

            candidate = Grasp(dof_6, width=width)
            rot = candidate.pose.rotation
            candidate.pose.rotation = rot * Rotation.from_euler("z", np.pi)
            grasps.append(candidate)
        scores = scores.tolist()
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        while True:
            flag = 1
            idx = 0
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
                        flag = 0              
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
        self.obs_server.integrate = True

        time.sleep(2)
        self.obs_server.integrate = False

        pc = self.obs_server.pc

        return pc
    
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
        self.pc.grasp(width=0.0, force=30.0)

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

    def approach_grasp(self, T_base_grasp):
        self.goto_pose_reactive(T_base_grasp * self.T_tcp_tool0, Gain=0.05, threshold=0.02)

    def drop(self):
        # self.pc.goto_joints(
        #     [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        # )
        self.pc.goto_joints(self.drop_joints, 0.2, 0.2)
        self.pc.move_gripper(0.08)
        

def main():
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController()
    t2 = Thread(target=panda_grasp.joint_velo_publisher,name='velo_pubilisher')
    t2.start()

    while True:
        panda_grasp.run()


if __name__ == "__main__":
    main()

    # # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5