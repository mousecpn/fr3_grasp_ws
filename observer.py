from utils_exp.transform import Rotation, Transform
from utils_exp.perception import *
import cv_bridge
import rospy
from utils_exp import ros_utils
import sensor_msgs.msg
from grasp_planner.edgegrasp import EdgeGraspPlanner, EdgeGraspObservation
import cv2
from scipy import ndimage


class TSDFServer(object):
    def __init__(self,cam_topic_name="/stereo/depth", color_topic_name="/stereo/left/image_rect_color"):
        # self.cam_frame_id = rospy.get_param("~cam/frame_id")
        self.cam_frame_id = "camera_color_optical_frame"
        self.cam_frame_id_color = "camera_color_optical_frame"
        self.base_frame_id = 'fr3_link0'
        # self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.cam_topic_name = cam_topic_name
        self.color_topic_name = color_topic_name
        self.intrinsic = CameraIntrinsic.from_dict({"K": [386.4255676269531, 0.0, 322.94573974609375, 0.0, 386.4255676269531, 234.94528198242188, 0.0, 0.0, 1.0],
                                                    "height": 480,
                                                    "width": 640})
        # self.intrinsic = CameraIntrinsic.from_dict({"K": [426.67822265625, 0.0, 427.2525634765625, 0.0, 426.67822265625, 234.44296264648438, 0.0, 0.0, 1.0],
        #                                             "height": 480,
        #                                             "width": 848})
        self.size = 6.0 * 0.05 # rospy.get_param("~finger_depth")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)
        # rospy.Subscriber(self.color_topic_name, sensor_msgs.msg.Image, self.sensor_cb_color)
        self.img = None
        self.color_img = None
        self.T_depth_color = Transform(Rotation.from_quat([0.00, 0.000, 0.00, 1.000]), [-0.015, -0.000, -0.000])


    def reset(self):
        # self.low_res_tsdf = None
        # self.high_res_tsdf = None
        self.img = None
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)
        self.processed_tsdf = TSDFVolume(self.size, 180)

    def sensor_cb_color(self, msg):
        if not self.integrate:
            return

        color_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.uint8)

        self.color_img = color_img

    def sensor_cb(self, msg):
        if not self.integrate:
            return
        imgs = []
        # for i in range(3):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)  * 0.001
        img[np.isnan(img)] = 0.0
        # imgs.append(img)

        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )

        self.processed_tsdf.integrate(img, self.intrinsic, T_cam_task)

        pcd = self.processed_tsdf.get_cloud()
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

        img = pointcloud_to_depth_image(
            pts_world=np.asarray(pcd.points),
            R=T_cam_task.rotation.as_matrix(),
            t=T_cam_task.translation,
            K=self.intrinsic.K,
            image_size=(self.intrinsic.height, self.intrinsic.width)
        )
        # img = depth_dilation(img)

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        

        self.img = img
    

    def pcl2tsdf(self, pcd, voxel_size=0.3/40):
        # 1) 获取点云边界，定义一个三维体素网格大小 (Nx, Ny, Nz)
        min_bound = np.array([0,0,0])
        max_bound = np.array([0.3,0.3,0.3])
        voxel_size = voxel_size
        dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

        # 2) 把点云放到体素索引里
        coords = ((np.asarray(pcd.points) - min_bound) / voxel_size).astype(int)
        occ_grid = np.zeros(dims, dtype=bool)
        occ_grid[coords[:,0], coords[:,1], coords[:,2]] = True
        # 3) 先算“空域”距离
        dist_out = ndimage.distance_transform_edt(~occ_grid) * voxel_size
        # 4) 再算“实域”距离
        dist_in  = ndimage.distance_transform_edt( occ_grid) * voxel_size
        # 5) 带符号：在内部取负
        tsdf = dist_out
        tsdf[occ_grid] = -dist_in[occ_grid]
        # 6) 截断到 ±trunc
        trunc = 4 * voxel_size
        tsdf = np.clip(tsdf, 0, trunc) / trunc  # 归一化到 [-1,1]

        tsdf[tsdf==1] = 0.
        tsdf[tsdf==-1] = 0.
        tsdf = tsdf.astype(np.float32)
        return tsdf


def pointcloud_to_depth_image(
    pts_world: np.ndarray,      # (N,3) 世界坐标下的点
    R: np.ndarray,              # (3,3) 旋转
    t: np.ndarray,              # (3,) 平移
    K: np.ndarray,              # (3,3) 相机内参
    image_size: tuple           # (H, W)
) -> np.ndarray:
    """
    将世界坐标系点云投影为深度图。
    返回 depth_map，shape=(H,W)，无点的位置深度设为 0。
    """
    H, W = image_size
    # 1) 世界→相机
    # 批量变换： X_c = R @ X_w.T + t[:,None]
    Xc = (R @ pts_world.T) + t.reshape(3,1)  # (3, N)
    x, y, z = Xc[0], Xc[1], Xc[2]

    # 只保留 Z>0 的点（在相机前方）
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # 2) 相机→像素
    u = (K[0,0] * x) / z + K[0,2]
    v = (K[1,1] * y) / z + K[1,2]

    # 四舍五入并转为整数像素坐标
    u_pix = np.round(u).astype(int)
    v_pix = np.round(v).astype(int)

    # 3) 初始化深度图，并填充最小深度（z-buffer）
    depth_map = np.zeros((H, W), dtype=np.float32)
    # 记录已有像素的最小深度
    z_buffer = np.full((H, W), np.inf, dtype=np.float32)

    # 遍历所有投影点
    for ui, vi, zi in zip(u_pix, v_pix, z):
        # 确保像素坐标在图像范围内
        if 0 <= vi < H and 0 <= ui < W:
            # 如果该像素还没写，或当前点更近，则更新
            if zi < z_buffer[vi, ui]:
                z_buffer[vi, ui] = zi
                depth_map[vi, ui] = zi

    return depth_map


class PCLServer(object):
    def __init__(self, cam_topic_name="/stereo/depth", color_topic_name="/stereo/left/image_rect_color"):
        # self.cam_frame_id = rospy.get_param("~cam/frame_id")
        self.cam_frame_id = "camera_color_optical_frame"
        # self.cam_frame_id_color = "camera_color_optical_frame"
        self.base_frame_id = 'fr3_link0'
        # self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.cam_topic_name = cam_topic_name
        self.color_topic_name = color_topic_name
        self.intrinsic = CameraIntrinsic.from_dict({"K": [386.4255676269531, 0.0, 322.94573974609375, 0.0, 386.4255676269531, 234.94528198242188, 0.0, 0.0, 1.0],
                                                    "height": 480,
                                                    "width": 640})
        # self.intrinsic = CameraIntrinsic.from_dict({"K": [426.67822265625, 0.0, 427.2525634765625, 0.0, 426.67822265625, 234.44296264648438, 0.0, 0.0, 1.0],
        #                                             "height": 480,
        #                                             "width": 848})
        self.size = 6.0 * 0.05 # rospy.get_param("~finger_depth")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)
        # rospy.Subscriber(self.color_topic_name, sensor_msgs.msg.Image, self.sensor_cb_color)
        self.img = None
        self.color_img = None
        self.pc = None
        self.T_depth_color = Transform(Rotation.from_quat([0.00, 0.000, 0.00, 1.000]), [-0.015, -0.000, -0.000])
        self.lower = np.r_[0.02, 0.02, 0.055]
        self.upper = np.r_[0.28, 0.28, 0.28]
        self.preproc = EdgeGraspObservation(lower_bounds=self.lower, upper_bounds=self.upper)


    def reset(self):
        # self.low_res_tsdf = TSDFVolume(self.size, 40)
        # self.high_res_tsdf = TSDFVolume(self.size, 120)
        self.pc = None

    def sensor_cb_color(self, msg):
        if not self.integrate:
            return

        color_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.uint8)

        self.color_img = color_img

    def sensor_cb(self, msg):
        if not self.integrate:
            return

        
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32) * 0.001
        img[np.isnan(img)] = 0.0

        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )


        theta = np.pi / 6.0
        phi = - np.pi / 2.0
        r = 2.0 * 0.3
        

        # theta = np.pi / 4
        # phi = -np.pi/2
        eye = np.r_[
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta),
            ]


        self.pc = self.preproc(img, self.intrinsic, T_cam_task, eye)

        # self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        # self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)

        self.img = img
    
class Processed_tsdf(object):
    def __init__(self, pcd, tsdf, voxel_size):
        self.pcd = pcd
        self.tsdf = tsdf
        self.voxel_size = voxel_size
        self.size = 0.3
    
    def get_grid(self,):
        return self.tsdf

    def get_cloud(self,):
        return self.pcd


def depth_dilation(depth):
    depth_in = depth.copy()
    nan_mask = np.isnan(depth_in)
    depth_in[nan_mask] = -np.inf  # 或者一个比所有深度都小的数

    # 2) 定义结构元（kernel），例如 3x3
    kernel = np.ones((3, 3), dtype=np.float32)

    # 3) 做膨胀
    depth_dilated = cv2.dilate(depth_in, kernel, iterations=1)

    # 4) 把原本 NaN 的地方再标回 NaN（如果需要）
    depth_dilated[depth_dilated==-np.inf] = np.nan
    return depth_dilated