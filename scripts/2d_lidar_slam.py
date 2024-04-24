#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2,LaserScan
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import PoseStamped
import pcl
import pcl_helper
from utils.ScanContextManager import *
from utils.PoseGraphManager import * 
from utils.UtilsMisc import *
import utils.UtilsPointcloud as PointcloudUtils
import utils.ICP as ICP
import pygicp
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import argparse
from icecream import ic, argumentToString
import small_gicp
from visualization_msgs.msg import Marker
import visualization_msgs
import threading

@argumentToString.register(np.ndarray)
def _(obj):
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=10000) # 5000 is enough for real time
parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=20) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper 10
parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)
parser.add_argument('--data_base_dir', type=str, 
                    default='/your/path/.../data_odometry_velodyne/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default = '00')
parser.add_argument('--save_gap', type=int, default = 300)
parser.add_argument('--use_open3d', action='store_true')

REGISTRATION_TYPE = 'fast_gicp' # {small_gicp , fast_gicp, open3d , icp}

args = parser.parse_args()

class IcpTest:
    def __init__(self):
        rospy.init_node('pcl_to_numpy_node', anonymous=True)
        rospy.Subscriber("front/scan", LaserScan, self.callback,queue_size=1)
        rospy.Subscriber('rf2o_laser_odometry/odom_rf2o', Odometry , self.odom_callback)
        rospy.Subscriber('odometry/filtered', Odometry , self.gt_odom_callback)

        self.optimized_path_pub = rospy.Publisher("path_optimized",Path,queue_size=1)
        self.groud_truth_path_pub = rospy.Publisher("path_ground_truth",Path,queue_size=1)
        self.icp_in_callback_odob_pub = rospy.Publisher("odometry_in_callback",Odometry,queue_size=1)

        self.print_initial_settings()
        self.current_time = None
        self.last_time = None
        self.scan_is_ready = False
        self.is_first_node = True

        self.icp_initial = np.eye(4)
        self.curr_se3 = np.eye(4)

        self.fast_gicp = pygicp.FastGICP()
        self.fast_gicp.set_num_threads(24)
        self.fast_gicp.set_max_correspondence_distance(30.0)
        self.fast_gicp.get_final_transformation()
        self.fast_gicp.get_final_hessian()

        self.odom_x = 0.0
        self.pcd = o3d.geometry.PointCloud()
        self.odom_y = 0.0

        self.ground_truth_x = 0.0
        self.ground_truth_y = 0.0
        self.ground_truth_list = []
        
        self.poseGraphManager_ = PoseGraphManager()
        self.poseGraphManager_.addPriorFactor()
        self.scanContextManager_ = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                        num_candidates = args.num_candidates, 
                                        threshold = args.loop_threshold)
        self.poseGraphResultSaver_ = PoseGraphResultSaver(init_pose=self.poseGraphManager_.curr_se3, 
                             save_gap = args.save_gap,
                             num_frames = 1,
                             seq_idx = args.sequence_idx,
                             save_dir = "result/" + args.sequence_idx)
        
        self.is_first_node = True
        
        self.current_test_points = None
        self.previous_test_points = None
        self.test_initial = np.eye(4)
        self.test_curr_se3 = np.eye(4)

        self.current_scan_points = None
        self.current_downsampled_points = None
        self.previous_scan_points = None
        self.previous_downsampled_points = None 
        
        self.cur_node_idx = 0

        self.cur_odom_trans_mat = np.eye(4)
        self.old_odom_trans_mat = np.eye(4)


        rate = rospy.Rate(hz=1.0)
        thread = threading.Thread()
        self.scan_loop_process_time = 0.0
        while not rospy.is_shutdown() :

            if(not self.scan_is_ready): continue

            self.odometry_pub()
            self.path_pub()

            start_time = time.time()


            # loop detection and optimize the graph 
            if(self.poseGraphManager_.curr_node_idx > 1 and self.poseGraphManager_.curr_node_idx % args.try_gap_loop_detection == 0): 
                # 1/ loop detection 
                s = time.time()

                loop_idx, loop_dist, yaw_diff_deg = self.scanContextManager_.detectLoop()
                print(f'loop detection takes {time.time() -s}sec')
                print(f"scan loop process time : {self.scan_loop_process_time}")
                if(loop_idx == None): # NOT FOUND
                    pass
                else:

                    # loop_position = getGraphNodePose(self.poseGraphManager_.graph_optimized,loop_idx)
                    # self.detected_node_viz(loop_position)
                    print(f"Loop event detected  current_node_idx : {self.poseGraphManager_.curr_node_idx}, loop_idx : {loop_idx}, dist : {loop_dist}")
                    # 2-1/ add the loop factor 
                    loop_scan_down_pts = self.scanContextManager_.getPtcloud(loop_idx)
                    # loop_transform, _, _ = ICP.icp(self.current_downsampled_points, loop_scan_down_pts, yawdeg2se3(yaw_diff_deg), max_iterations=20)
                    self.fast_gicp.set_input_target(loop_scan_down_pts)
                    self.fast_gicp.set_input_source(self.current_scan_points)
                    loop_transform = self.fast_gicp.align()

                    self.poseGraphManager_.addLoopFactor(loop_transform, loop_idx)
                    # 2-2/ graph optimization 
                    self.poseGraphManager_.optimizePoseGraph()

                    
                    self.detected_node_viz(self.poseGraphManager_.graph_optimized.atPose3(gtsam.symbol('x', loop_idx)).x(),self.poseGraphManager_.graph_optimized.atPose3(gtsam.symbol('x', loop_idx)).y())
                    self.poseGraphResultSaver_.OptimizedPoseGraphUpdate(self.poseGraphManager_.curr_node_idx, self.poseGraphManager_.graph_optimized)


            self.poseGraphResultSaver_.OptimizedPoseUpdate(self.poseGraphManager_.curr_se3)
            # self.poseGraphResultSaver_.vizCurrentTrajectory(fig_idx=1)


            # print(f"loop takes {time.time() - start_time} sec times")
            rate.sleep()

    def detected_node_viz(self,loop_position_x ,loop_position_y):
        vis_pub = rospy.Publisher("node_marker",Marker,queue_size = 1)
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "Node Index"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = loop_position_x
        marker.pose.position.y = loop_position_y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        # marker.lifetime = 5.0
        vis_pub.publish( marker )

    def path_pub(self):

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'odom'
        poses_list = []
        ground_truth_poses_list = []

        self.ground_truth_list.append((self.ground_truth_x,self.ground_truth_y))
        ##
        for pose_array in self.poseGraphResultSaver_.pose_list:
            x = pose_array[3]
            y = pose_array[7]
            pose_stamp = PoseStamped()
            pose_stamp.header.frame_id = 'odom'
            pose_stamp.header.stamp = rospy.Time.now()
            pose_stamp.pose.position.x = x
            pose_stamp.pose.position.y = y
            poses_list.append(pose_stamp)
        path_msg.poses = poses_list
        self.optimized_path_pub.publish(path_msg)

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'odom'

        for (x,y) in self.ground_truth_list:
            pose_stamp = PoseStamped()
            pose_stamp.header.frame_id = 'odom'
            pose_stamp.header.stamp = rospy.Time.now()
            pose_stamp.pose.position.x = x
            pose_stamp.pose.position.y = y
            ground_truth_poses_list.append(pose_stamp)

        path_msg.poses = ground_truth_poses_list
        self.groud_truth_path_pub.publish(path_msg)


    def gt_odom_callback(self,msg):

        self.ground_truth_x = msg.pose.pose.position.x
        self.ground_truth_y = msg.pose.pose.position.y

        x,y,z,w = msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w
        rot_mat = self.quaternion_to_rotation_matrix(w,x,y,z)

        trans_mat = np.eye(4)
        trans_mat[:3,:3] = rot_mat
        trans_mat[0,-1] = self.ground_truth_x
        trans_mat[1,-1] = self.ground_truth_y

        self.cur_odom_trans_mat = copy.deepcopy(trans_mat)


    def odometry_pub(self):

        self.odom_pub = rospy.Publisher("odom_graph_slam",Odometry,queue_size=1)
        msg = Odometry()
        cur_se3 = self.poseGraphManager_.curr_se3
        r = R.from_matrix(cur_se3[:3,:3])
        x,y,z,w=r.as_quat()
        odom_x = cur_se3[0,-1]
        odom_y = cur_se3[1,-1]
        msg.child_frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = odom_x
        msg.pose.pose.position.y = odom_y
        msg.pose.pose.orientation.x = x
        msg.pose.pose.orientation.y = y
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w
        self.odom_pub.publish(msg)



    def initialize_first_node(self,first_idx , cur_pts):

        # self.poseGraphManager_.curr_node_idx = first_idx # make start with 0
        # self.scanContextManager_.addNode(node_idx=self.poseGraphManager_.curr_node_idx, ptcloud=cur_pts)
    
        # self.poseGraphManager_.prev_node_idx = first_idx
        self.previous_scan_points = copy.deepcopy(cur_pts)
        self.icp_initial = np.eye(4)

        self.is_first_node = False
        self.scan_is_ready = True

        
    def callback(self,  scan_points):
        s = time.time()
        points = []
        for idx, distance in enumerate(scan_points.ranges):
            if (distance > 30.0):
                distance = 30.0
            x = distance * np.cos(scan_points.angle_min + idx * scan_points.angle_increment)
            y = distance * np.sin(scan_points.angle_min + idx * scan_points.angle_increment)
            points.append([x,y,0])

        self.current_scan_points = np.array(points)
        if(self.is_first_node):
            self.initialize_first_node(0,self.current_scan_points)
            return
        # self.cur_node_idx += 1 
        
        result = small_gicp.align(target_points = self.previous_scan_points, source_points = self.current_scan_points, downsampling_resolution=0.01,init_T_target_source = self.icp_initial,\
                                          registration_type = 'GICP',voxel_resolution=0.01,max_corresponding_distance = 3.0,num_threads = 30)
        relative_pose = result.T_target_source
        self.poseGraphManager_.curr_se3 = np.matmul(self.poseGraphManager_.curr_se3, relative_pose)
        # self.scanContextManager_.addNode(node_idx = self.poseGraphManager_.curr_node_idx, ptcloud=self.current_scan_points)
        # self.poseGraphManager_.curr_node_idx = self.cur_node_idx 
        # self.icp_initial = relative_pose # assumption: constant velocity model (for better next ICP converges)
        # self.poseGraphManager_.addOdometryFactor(relative_pose)
        # self.poseGraphManager_.prev_node_idx = self.poseGraphManager_.curr_node_idx
        self.previous_scan_points = copy.deepcopy(self.current_scan_points)
        # self.scan_loop_process_time = time.time() - s
        print(*self.poseGraphManager_.curr_se3[:2,-1])

        # msg = Odometry()
        # cur_se3 = self.test_curr_se3
        # r = R.from_matrix(cur_se3[:3,:3])
        # x,y,z,w=r.as_quat()
        # odom_x = cur_se3[0,-1]
        # odom_y = cur_se3[1,-1]
        # msg.child_frame_id = "base_link"
        # msg.header.stamp = rospy.Time.now()
        # msg.header.frame_id = 'odom'
        # msg.pose.pose.position.x = odom_x
        # msg.pose.pose.position.y = odom_y
        # msg.pose.pose.orientation.x = x
        # msg.pose.pose.orientation.y = y
        # msg.pose.pose.orientation.z = z
        # msg.pose.pose.orientation.w = w
        # self.icp_in_callback_odob_pub.publish(msg)


    def odom_callback(self,msg):

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        x,y,z,w = msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w
        rot_mat = self.quaternion_to_rotation_matrix(w,x,y,z)

        trans_mat = np.eye(4)
        trans_mat[:3,:3] = rot_mat
        trans_mat[0,-1] = self.odom_x
        trans_mat[1,-1] = self.odom_y

        self.cur_odom_trans_mat = np.copy(trans_mat)

    def quaternion_to_rotation_matrix(self,w,x,y,z):
        # Quaternion은 (w, x, y, z) 형태로 입력받음
        # 회전 행렬 계산
        R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        return R
    

    def print_initial_settings(self):
        
        print(f"Number of ICP Points: {args.num_icp_points}")
        print(f"Number of Rings: {args.num_rings}")
        print(f"Number of Sectors: {args.num_sectors}")
        print(f"Number of Candidates: {args.num_candidates}")
        print(f"Gap for Loop Detection Attempts: {args.try_gap_loop_detection}")
        print(f"Loop Closure Threshold: {args.loop_threshold}")
        print(f"Sequence Index: {args.sequence_idx}")
        print(f"Data Saving Interval: {args.save_gap}")
        print(f"Use Open3D: {args.use_open3d}")
        print(f"REGISTRATION_TYPE : {REGISTRATION_TYPE}")
        
if __name__ == "__main__":
    start = IcpTest()
