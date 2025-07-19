import rospy
import actionlib

import moveit_commander
from moveit_commander.conversions import list_to_pose

import franka_gripper.msg
from franka_msgs.msg import ErrorRecoveryActionGoal
import numpy as np

class PandaCommander(object):
    """
    PandaCommander is a class which wraps some basic moveit functions for the Panda Robot,
    and some via the panda API
    """
    def __init__(self, group_name=None):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.groups = {}
        self.active_group = None
        self.set_group(group_name)

        self.reset_publisher = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=1)
        
    #     self.set_named_poses()
        
    # def set_named_poses(self):
    #     """
    #     Set poses used in the application via their name
    #     """
    #     this_pose = self.active_group.get_current_joint_values()
    #     print(this_pose)
    #     this_pose[0] += np.random.randn() * 0.05
    #     this_pose[1] += np.random.randn() * 0.05
        
    #     #   translation then rotation                                 x                    y                    z                     x                    y                   z                    w
    #     self.active_group.remember_joint_values('grip_ready', this_pose) #[-0.8680978268436771, -0.6478007817770287, -0.35404076143321106, -2.1504153873157840, -0.2511609577867720, 1.5078568387561373, -0.34946772768514023])
    #     self.active_group.remember_joint_values('drop_box', this_pose)  #[ 0.8263036876751351,  1.0896425800490797,  0.82033907273777810, -0.8825330551727532, -0.7649754029929701, 1.7213938517864830,  2.28231075602280060])
    

    def set_named_poses(self,name,pose):
        self.active_group.remember_joint_values(name, pose) 
        return


    def print_debug_info(self):
        if self.active_group:
            planning_frame = self.active_group.get_planning_frame()
            print("============ Reference frame: %s" % planning_frame)
            eef_link = self.active_group.get_end_effector_link()
            print("============ End effector: %s" % eef_link)
        else:
            print("============ No active planning group.")
        print("============ Robot Groups:", self.robot.get_group_names())
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

    def set_group(self, group_name):
        """
        Set the active move group
        :param group_name: move group name
        """
        self.active_group = group_name
        if group_name is None:
            self.active_group = None
            return
        else:
            if group_name not in self.groups:
                if group_name not in self.robot.get_group_names():
                    raise ValueError('Group name %s is not valid. Options are %s' % (group_name, self.robot.get_group_names()))
                self.groups[group_name] = moveit_commander.MoveGroupCommander(group_name)
            self.active_group = self.groups[group_name]

    def goto_joints(self, joint_values, group_name=None, wait=True):
        """
        Move to joint positions.
        :param joint_values:  Array of joint positions
        :param group_name:  Move group (use current if None)
        :param wait:  Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        joint_goal = self.active_group.get_current_joint_values()
        if len(joint_goal) != len(joint_values):
            raise IndexError('Expected %d Joint Values, got %d' % (len(joint_goal), len(joint_values)))
        for i, v in enumerate(joint_values):
            joint_goal[i] = v

        success = self.active_group.go(joint_goal, wait)
        self.active_group.stop()
        return success

    def goto_pose(self, pose, velocity=1.0, group_name=None, wait=True):
        """
        Move to pose
        :param pose: Array position & orientation [x, y, z, qx, qy, qz, qw]
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)
        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_pose_target(pose)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success

    def goto_pose_cartesian(self, pose, velocity=1.0, group_name=None, wait=True):
        """
        Move to pose following a cartesian trajectory.
        :param pose: Array position & orientation [x, y, z, qx, qy, qz, qw]
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)

        self.active_group.set_max_velocity_scaling_factor(velocity)
        (plan, fraction) = self.active_group.compute_cartesian_path(
                                           [pose],   # waypoints to follow
                                           0.005,    # eef_step
                                           0.0)      # jump_threshold
        if fraction != 1.0:
            raise ValueError('Unable to plan entire path!')

        success = self.active_group.execute(plan, wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success

    def goto_named_pose(self, pose_name, velocity=1.0, group_name=None, wait=True):
        """
        Move to named pos
        :param pose: Name of named pose
        :param velocity: Velocity (fraction of max) [0.0, 1.0]
        :param group_name: Move group (use current if None)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_named_target(pose_name)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        return success

    def home_gripper(self):
        """
        Home and initialise the gripper
        :return: Bool success
        """
        client = actionlib.SimpleActionClient('franka_gripper/homing', franka_gripper.msg.HomingAction)
        client.wait_for_server()
        client.send_goal(franka_gripper.msg.HomingGoal())
        return client.wait_for_result()

    def set_gripper(self, width, speed=0.1, wait=True):
        """
        Set gripper with.
        :param width: Width in metres
        :param speed: Move velocity (m/s)
        :param wait: Wait for completion if True
        :return: Bool success
        """
        client = actionlib.SimpleActionClient('franka_gripper/move', franka_gripper.msg.MoveAction)
        client.wait_for_server()
        client.send_goal(franka_gripper.msg.MoveGoal(width, speed))
        if wait:
            return client.wait_for_result()
        else:
            return True

    def grasp(self, width=0, e_inner=0.1, e_outer=0.1, speed=0.1, force=1):
        """
        Wrapper around the franka_gripper/grasp action.
        http://docs.ros.org/kinetic/api/franka_gripper/html/action/Grasp.html
        :param width: Width (m) to grip
        :param e_inner: epsilon inner
        :param e_outer: epsilon outer
        :param speed: Move velocity (m/s)
        :param force: Force to apply (N)
        :return: Bool success
        """
        client = actionlib.SimpleActionClient('franka_gripper/grasp', franka_gripper.msg.GraspAction)
        client.wait_for_server()
        client.send_goal(
            franka_gripper.msg.GraspGoal(
                width,
                franka_gripper.msg.GraspEpsilon(e_inner, e_outer),
                speed,
                force
            )
        )
        return client.wait_for_result()

    def stop(self):
        """
        Stop the current movement.
        """
        if self.active_group:
            self.active_group.stop()

    def recover(self):
        """
        Call the error reset action server.
        """
        self.reset_publisher.publish(ErrorRecoveryActionGoal())
        rospy.sleep(3.0)
