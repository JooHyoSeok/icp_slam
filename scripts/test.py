#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
import pcl
import pcl_helper
from utils.ScanContextManager import *
from utils.PoseGraphManager import * 
from utils.UtilsMisc import *
import utils.UtilsPointcloud as PointcloudUtils
import utils.ICP as ICP
import time
import open3d as o3d

import argparse
from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(obj):
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time
parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=50) # same as the original paper
parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)
parser.add_argument('--data_base_dir', type=str, 
                    default='/your/path/.../data_odometry_velodyne/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default='00')
parser.add_argument('--save_gap', type=int, default=300)
parser.add_argument('--use_open3d', action='store_true')

args = parser.parse_args()

class IcpTest:
    def __init__(self):
        rospy.init_node('pcl_to_numpy_node', anonymous=True)
        rospy.Subscriber("points", PointCloud2, self.callback)
        rospy.Subscriber('odometry/filtered', Odometry , self.odom_callback)

        self.current_time = None
        self.last_time = None
        self.scan_is_ready = False
        self.is_first_node = True

        self.icp_initial = np.eye(4)
        self.curr_se3 = np.eye(4)
        
        self.odom_x = 0.0
        self.pcd = o3d.geometry.PointCloud()
        self.odom_y = 0.0
        
        self.poseGraphManager_ = PoseGraphManager()
        self.poseGraphManager_.addPriorFactor()
        self.scanContextManager_ = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                        num_candidates=args.num_candidates, 
                                        threshold=args.loop_threshold)
        self.poseGraphResultSaver_ = PoseGraphResultSaver(init_pose=self.poseGraphManager_.curr_se3, 
                             save_gap=args.save_gap,
                             num_frames=1,
                             seq_idx=args.sequence_idx,
                             save_dir="result/" + args.sequence_idx)
        
        self.is_first_node = True

        self.current_scan_points = None
        self.current_downsampled_points = None
        self.previous_scan_points = None
        self.previous_downsampled_points = None 
        
        self.icp_initial = None
        self.cur_node_idx = 0


        rate = rospy.Rate(30)
        while not rospy.is_shutdown() :

            if(not self.scan_is_ready): continue


            try: 
            # self.current_downsampled_points = PointcloudUtils.random_sampling(self.current_scan_points, num_points=args.num_icp_points)
            # self.previous_downsampled_points = PointcloudUtils.random_sampling(self.previous_scan_points, num_points=args.num_icp_points)

                self.current_downsampled_points = self.voxel_downsampling(self.current_scan_points)
                self.previous_downsampled_points = self.voxel_downsampling(self.previous_scan_points)
                # self.current_downsampled_points = PointcloudUtils.random_sampling(self.current_downsampled_points, num_points=6000)
                # self.previous_downsampled_points = PointcloudUtils.random_sampling(self.previous_downsampled_points, num_points=6000)

                source = o3d.geometry.PointCloud()
                target = o3d.geometry.PointCloud()

                source.points = o3d.utility.Vector3dVector(self.current_downsampled_points)
                target.points = o3d.utility.Vector3dVector(self.previous_downsampled_points)
                
                ic()
                
                source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                reg_p2l = o3d.pipelines.registration.registration_icp(
                    source, target, 1.0, self.icp_initial,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
                
                # ic()
                # reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
                #     source, target, 10, self.icp_initial, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())              
                # ic()
                relative_pose = reg_p2l.transformation 


                # relative_pose, _, _ = ICP.icp(self.current_downsampled_points, self.previous_downsampled_points, prev_relative_pose = self.icp_initial, max_iterations=20)
                self.icp_initial = relative_pose # assumption: constant velocity model (for better next ICP converges)
                self.curr_se3 = np.matmul(self.curr_se3, relative_pose).round(4)
                x,y  = self.curr_se3[0,-1], self.curr_se3[1,-1]
                
                print(f"odom x : {round(self.odom_x, 4)} y : {round(self.odom_y,4)}")
                print(f"icp  x : {x} y : {y}")

                self.previous_scan_points = copy.deepcopy(self.current_scan_points)
            except Exception as e:
                ic(e)
            rate.sleep()

    def voxel_downsampling(self, pts):
            
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        pts_downed = self.pcd.voxel_down_sample(voxel_size=0.2)
        pcd_down_np = np.asarray(pts_downed.points)

        return pcd_down_np

    def callback(self,  point_cloud_ros):

        self.current_scan_points = np.array(list(pc2.read_points(point_cloud_ros, field_names=("x", "y", "z"), skip_nans=True)))
        self.scan_is_ready = True
        if(self.is_first_node):
            self.initialize_first_node(self.current_scan_points)

    def initialize_first_node(self, cur_pts):

        self.previous_scan_points = copy.deepcopy(cur_pts)
        self.is_first_node = False

    def do_passthrough(self,pcl_data,filter_axis,axis_min,axis_max):
        
        passthrough = pcl_data.make_passthrough_filter()
        passthrough.set_filter_field_name(filter_axis)
        passthrough.set_filter_limits(axis_min, axis_max)
        return passthrough.filter()

    def do_ransac_plane_normal_segmentation(self,point_cloud, input_max_distance):

        segmenter = point_cloud.make_segmenter_normals(ksearch=50)
        segmenter.set_optimize_coefficients(True)
        segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  #pcl_sac_model_plane
        segmenter.set_normal_distance_weight(0.1)
        segmenter.set_method_type(pcl.SAC_RANSAC) #pcl_sac_ransac
        segmenter.set_max_iterations(1000)
        segmenter.set_distance_threshold(input_max_distance) #0.03)  #max_distance
        indices, coefficients = segmenter.segment()

        inliers = point_cloud.extract(indices, negative=False)
        outliers = point_cloud.extract(indices, negative=True)

        return indices, inliers, outliers
    
    def odom_callback(self,msg):

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def do_voxel_grid_downssampling(pcl_data,leaf_size):

        '''

        Create a VoxelGrid filter object for a input point cloud
        :param pcl_data: point cloud data subscriber
        :param leaf_size: voxel(or leaf) size
        :return: Voxel grid voxel_downsampling on point cloud
        :https://github.com/fouliex/RoboticPerception

        '''
        vox = pcl_data.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size) # The bigger the leaf size the less information retained
        return  vox.filter()
    

if __name__ == "__main__":
    start = IcpTest()
