# The Real World Experiment of Equivariant Volumetric Grasping
The code is based on ROS1. Since our system is fully upgraded to ROS2, it is impossible for us to test and maintain the repo anymore. The code is just for a reference and study.

## 🔧 Installation
- Install ROS1
- catkin_make [dougsm_helpers](dougsm_helpers) and [franka_control_wrappers](franka_control_wrappers)
- You might need to install [VGN](https://github.com/ethz-asl/vgn) and clone [Equivariant-Volumetric-Grasping](https://github.com/mousecpn/Equivariant-Volumetric-Grasping).
- If you need to test ICGNet and EdgeGraspNet, plase clone [icgnet's repo](https://github.com/renezurbruegg/icg_benchmark).
- Calibration is required and publish your extrinsics in robot_bringup.launch or robot_bringup_realsense.launch
- [RoboticsToolbox](https://github.com/petercorke/robotics-toolbox-python)

