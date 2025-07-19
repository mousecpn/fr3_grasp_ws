#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path
import franka_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

from utils_exp.perception import *
from utils_exp import ros_utils
from utils_exp.transform import Rotation, Transform
from utils_exp.panda_control import PandaCommander
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3
from std_srvs.srv import Empty
from std_msgs.msg import Empty as Emptymsg
from utils_exp import ros_utils
from franka_msgs.msg import FrankaState
from utils_exp.control import calculate_velocity

PANDA_JOINT_NAME = ["panda_joint{}".format(i+1) for i in range(7)]

class BasePandaGraspController(object):
    def __init__(self,):
        self.robot_error = False
        self.base_frame_id = "fr3_link0"
        self.tool0_frame_id = "fr3_hand_tcp"
        self.T_tool0_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, -0.05]})  # TODO
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = 0.05
        self.size = 6.0 * self.finger_depth
        # self.drop_joints = [1.0634903603929122, 0.5137441330101977, 0.01832026272857114, -1.6526025723500668, 0.010896748289250554, 2.1817810641417794, 1.0967807523648219]
        self.drop_joints = [1.0446276744688945, 0.30173620268867446, 0.08249576790484157, -1.8373576100084998, -0.007363607251237792, 2.1420626724004688, 0.06503973433038945]
        self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
    

        self.T_realbase_base = Transform(Rotation.identity(), [-0.001, -0.00, 0.022])
        # self.T_realbase_base = Transform(Rotation.identity(), [0.000, 0.000, 0.000])
        self.T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.08])
        self.T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.08])

        # Make a Panda robot
        self.panda = rtb.models.Panda()
        self.panda.qlim = np.array([[-2.7437,-1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
                                    [2.7437,1.7837,2.9007,-0.1518,2.8065,4.5169,3.0159]
                                    ])
        
        self.clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)

        self.joint_command_msg = sensor_msgs.msg.JointState()
        self.joint_command_msg.name = PANDA_JOINT_NAME

        self.cur_joints = np.zeros((7,))
        self.wrench = np.zeros((7,))

        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.force_callback)

        rospy.loginfo("Ready to take action")
    


    def force_callback(self, msg):
        self.wrench = np.array(msg.O_F_ext_hat_K)
        time.sleep(0.02)

    
    def check_achievable(self, grasp_pose):
        Tep = SE3(grasp_pose.as_matrix())
        success = self.panda.ik_LM(Tep)[1]
        return success

    def setup_panda_control(self):
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            franka_msgs.msg.FrankaState,
            self.robot_state_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.pc = PandaCommander()
        self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

    def define_workspace(self):
        self.T_base_task = Transform(
            Rotation.from_quat([0, 0, 0, 1]), [0.311, -0.011, 0.01-0.05] # -0.007-0.05
        )

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        self.set_table(0.045)


        # msg = geometry_msgs.msg.PoseStamped()
        # msg.header.frame_id = self.base_frame_id
        # msg.pose = ros_utils.to_pose_msg(self.T_base_task)
        # msg.pose.position.z += 0.01
        # msg.pose.position.y -= 0.4
        # msg.pose.position.x += 0.0
        # self.pc.scene.add_box("dropbox", msg, size=(0.3, 0.25, 0.25))

        rospy.sleep(1.0)  # wait for the scene to be updated

        # msg = geometry_msgs.msg.PoseStamped()
        # msg.header.frame_id = self.base_frame_id
        # msg.pose = ros_utils.to_pose_msg(T_base_wall)
        # msg.pose.position.x -= 0.01
        # self.pc.scene.add_box("wall", msg, size=(0.02, 0.6, 0.6))

        # rospy.sleep(1.0) 

    def robot_state_cb(self, msg):
        detected_error = False
        if np.any(msg.cartesian_collision):
            detected_error = True
        for s in franka_msgs.msg.Errors.__slots__:
            if getattr(msg.current_errors, s):
                detected_error = True
        if not self.robot_error and detected_error:
            self.robot_error = True
            rospy.logwarn("Detected robot error")

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]
        self.joints_state = msg.position[:7]
        self.cur_joints = np.array(msg.position)
        try:
            self.T_base_grasp_current = self.tf_tree.lookup(
                'fr3_link0', "fr3_hand_tcp", msg.header.stamp, rospy.Duration(0.1)
            )
        except:
            print("read error")
        time.sleep(0.01)

    def recover_robot(self):
        self.pc.recover()
        self.robot_error = False
        rospy.loginfo("Recovered from robot error")
    
    def rotate_grasp(self, grasps):
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, np.pi / 2.0])

        z_offset = 0
        t_augment = np.r_[0.0, 0.0, z_offset]
        T_augment = Transform(R_augment, t_augment)

        T_center = Transform(Rotation.identity(), np.r_[self.size/2, self.size/2, self.size/2])
        T = T_center * T_augment * T_center.inverse()
        for grasp in grasps:
            grasp.pose = T * grasp.pose
        return grasps
    

    def goto_pose_reactive(self, target_pose, Gain=0.1, threshold=0.001, detect_force=True, watch_dog_limit=70):
        if self.cs.current_controller != "joint_velocity":
            self.cs.switch_controller("joint_velocity")
        arrived = False
        watch_dog = 0
        last_joint = self.cur_joints
        while True:
            
            robot_state_joint = self.joints_state
            joint_vel, arrived = calculate_velocity(self.panda, np.array(robot_state_joint), target_pose, 
                                                    obstacles=None, Gain=Gain, threshold=threshold)
            joint_movement = np.linalg.norm(self.cur_joints-last_joint)
            last_joint = self.cur_joints
            if joint_movement < 0.002:
                watch_dog += 1
            else:
                watch_dog = 0
            if arrived is True or self.robot_error:
                break
            if watch_dog > watch_dog_limit:
                print('watch dog')
                break
            # print('force:',np.linalg.norm(self.wrench))
            if np.linalg.norm(self.wrench) > 15 and detect_force:
                # print('big force')
                print('force:',np.linalg.norm(self.wrench))
                break
            self.joint_command_msg.velocity = joint_vel # * (1/max((np.linalg.norm(self.wrench) -10),1))
            time.sleep(0.05)
        self.joint_command_msg.velocity = [0.0, 0.0, 0.0, 0., 0., 0., 0.]
        time.sleep(0.05)
        return arrived
    
    def set_table(self, height=0.035):
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(self.T_base_task)
        msg.pose.position.z += height
        msg.pose.position.y += 0.15
        msg.pose.position.x += 0.15
        self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))
    

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_base_pregrasp = T_base_grasp * self.T_grasp_pregrasp
        T_base_retreat = T_base_grasp * self.T_grasp_retreat

        # success = self.goto_pose_reactive(T_base_pregrasp * self.T_tcp_tool0, threshold=0.01)
        success = self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.1)
        # self.pc.move_gripper(grasp.width)
        if success is False:
            return False

        if self.robot_error:
            self.recover_robot()
            # self.pc.home()
            return False
        
        self.clear_octomap.call()
        rospy.sleep(0.05)
        
        robot_state_joint = self.joints_state
        self.panda.q = np.array(robot_state_joint)
        current_pose = np.array(self.panda.fkine(robot_state_joint).inv())
        current_pose = Transform.from_matrix(current_pose)
        error = np.sum(np.abs(np.r_[(current_pose * T_base_grasp).translation, (current_pose * T_base_grasp).rotation.as_euler('xyz') * np.pi / 180]))
        print(error)
        if error > 0.1:
            return False
        
        success = self.approach_grasp(T_base_grasp)
        # self.pc.goto_pose(T_base_grasp* self.T_tcp_tool0, velocity_scaling=0.2)

        if self.robot_error:
            self.recover_robot()
        
        # if success is False:
        #     return False
        
        # self.cs.switch_controller('moveit')
        self.pc.grasp(width=0.0, force=30.0, speed=0.05)

        if self.robot_error:
            self.recover_robot()
            # self.cs.switch_controller('moveit')
            # self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0, velocity_scaling=0.1)
            # return False

        self.pc.grasp(width=0.0, speed=1.3, force=30.0)

        self.goto_pose_reactive(T_base_retreat * self.T_tcp_tool0, Gain=0.1, threshold=0.01, detect_force=False, )
        # self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0, velocity_scaling=0.1)

        if self.robot_error:
            self.recover_robot()
    
        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.2])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.goto_pose_reactive(T_base_lift * self.T_tcp_tool0, threshold=0.05, detect_force=False, watch_dog_limit=20)
        self.cs.switch_controller('moveit')
        # self.pc.goto_pose(T_base_lift * self.T_tcp_tool0, velocity_scaling=0.2)

        
        # if self.robot_error:
        #     self.recover_robot()

        if self.gripper_width > 0.001:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        return self.goto_pose_reactive(T_base_grasp * self.T_tcp_tool0, Gain=0.4, threshold=0.01)

    def drop(self):
        # self.pc.goto_joints(
        #     [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        # )
        self.pc.goto_joints(self.drop_joints, 0.2, 0.2)
        self.pc.move_gripper(0.08)