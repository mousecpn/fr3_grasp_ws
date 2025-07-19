#!/usr/bin/env python

import numpy as np
import rospy
import sensor_msgs.msg
import sys
from dougsm_helpers.ros_control import ControlSwitcher
from geometry_msgs.msg import Twist
from vgn import vis
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from grasp_planner.edgegrasp import EdgeGraspPlanner, EdgeGraspObservation
from grasp_planner.edgegraspnet import EdgeGrasper
from grasp_planner.vn_edge_grasper import EdgeGrasper as VNEdgeGrasper

from vgn.grasp import Grasp

from threading import Thread, Lock
from observer import PCLServer
from base_fr3_grasp import BasePandaGraspController
import sys
sys.path.append('/home/pinhao/icg_benchmark')
# from icg_benchmark.grasping.preprocessing import ICGNetObservation

# from icg_benchmark.models.edge_grasp.edge_grasper import EdgeGrasper

T_base_wall = Transform(Rotation.identity(), [1, 0, 0])
round_id = 0


class PandaGraspController(BasePandaGraspController):
    def __init__(self, vn=True):
        super(PandaGraspController, self).__init__()
        self.obs_server = PCLServer(cam_topic_name='/camera/depth/image_rect_raw', )
        if vn == False:
            self.grasp_planner = EdgeGraspPlanner(
                EdgeGrasper(device="cuda", root_dir="/home/pinhao/icg_benchmark/data/edge_grasp_net_pretrained_para", load=180),
                confidence_th=0.4,
                return_best_score=False,)
        else:
            self.grasp_planner = EdgeGraspPlanner(
                VNEdgeGrasper(device="cuda", root_dir="/home/pinhao/icg_benchmark/data/vn_edge_pretrained_para", load=105),
                confidence_th=0.1,
                return_best_score=False,
            )
        
        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                            'velocity': 'cartesian_velocity_node_controller',
                            'joint_velocity': 'joint_velocity_node_controller'})
        
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.joint_velo_pub = rospy.Publisher('/joint_velocity_node_controller/joint_velocity', sensor_msgs.msg.JointState, queue_size=1)
        
        
        self.obs_list = []


    def run(self):
        if self.robot_error:
            self.recover_robot()
        if self.cs.current_controller != "moveit":
            self.cs.switch_controller("moveit")
        self.pc.home()
        self.pc.move_gripper(0.08)
        vis.clear()
        vis.draw_workspace(self.size)

        pc = self.acquire_tsdf()
        if pc is None:
            return
        # vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(self.T_realbase_base.transform_point(np.asarray(pc.points)))
        self.pc.home()
        # return

        poses = self.grasp_planner(pc)
        if poses is None:
            return
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
            rot = candidate.pose.rotation
            
            axis = rot.as_matrix()[:, 0]
            if axis[-1] > 0 and translation[-1] < 0.12:
                candidate.pose.rotation = rot * Rotation.from_euler("z", np.pi)
            candidate.pose = self.T_realbase_base * candidate.pose
            grasps.append(candidate)
        scores = scores.tolist()
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        while True:
            flag = 1
            idx = 0
            # grasp, score = grasps[0], scores[0]
            grasp, score, idx = self.select_grasp(grasps, scores)
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
                #         flag = 0              
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
        self.set_table(0.15)

        if self.robot_error:
            self.recover_robot()

        if label or self.gripper_width > 0.001:
            self.drop()
        self.pc.home()
        self.set_table(0.035)

    def acquire_tsdf(self):
        self.obs_server.reset()
        T_task_cam = Transform(Rotation.from_quat([-0.866, 0, 0, -0.5]), [-0.15, 0.1616, 0.5200]).inverse()

        # r = 2 * self.size
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
        self.pc.goto_pose(T_base_tcp, velocity_scaling=0.4, acceleration_scaling=0.2)
        time.sleep(0.5)
        self.obs_server.integrate = True

        time.sleep(2)
        self.obs_server.integrate = False

        pc = self.obs_server.pc

        return pc


    
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
        # rot = grasp.pose.rotation
        # axis = rot.as_matrix()[:, 0]
        # if axis[0] < 0:
        #     grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score, idx


        

def main():
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController()
    t2 = Thread(target=panda_grasp.joint_velo_publisher,name='velo_pubilisher')
    t2.start()

    while True:
        panda_grasp.run()

def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()

if __name__ == "__main__":
    main()

    # # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5