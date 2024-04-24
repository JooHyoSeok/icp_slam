import os
import sys
import csv
import copy
import time
import random
import argparse

import numpy as np
np.set_printoptions(precision=4)
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

from utils.ScanContextManager import *
from utils.PoseGraphManager import * 
from utils.UtilsMisc import *
import utils.UtilsPointcloud as PointcloudUtils
import utils.ICP as ICP
import open3d as o3d

'''debugging print matrix shape'''
from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(obj):
    return f"ndarray, shape={obj.shape}, dtype={obj.dtype}"


parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=7500) # 5000 is enough for real time
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




sequence_dir = '/Users/hyoseokju/Desktop/slam/PyICP-SLAM/data_odometry_velodyne/00/velodyne'

sequence_manager = PointcloudUtils.KittiScanDirManager(sequence_dir)
scan_paths = sequence_manager.scan_fullpaths
fig_idx = 1
fig = plt.figure(fig_idx)

class ICPSlamSystem:
    def __init__(self):

        self.print_initial_settings()
        self.poseGraphManager_ = PoseGraphManager()
        self.poseGraphManager_.addPriorFactor()
        self.scanContextManager_ = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                        num_candidates=args.num_candidates, 
                                        threshold=args.loop_threshold)
        self.poseGraphResultSaver_ = PoseGraphResultSaver(init_pose=self.poseGraphManager_.curr_se3, 
                             save_gap=args.save_gap,
                             num_frames=len(scan_paths),
                             seq_idx=args.sequence_idx,
                             save_dir="result/" + args.sequence_idx)
        
        self.is_first_node = True

        self.current_scan_points = None
        self.current_downsampled_points = None
        self.previous_scan_points = None
        self.previous_downsampled_points = None 
        
        self.icp_initial = None
        
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
        
    def initialize_first_node(self,first_idx , cur_pts):

        self.poseGraphManager_.curr_node_idx = first_idx # make start with 0
        self.scanContextManager_.addNode(node_idx=self.poseGraphManager_.curr_node_idx, ptcloud=PointcloudUtils.random_sampling(cur_pts, num_points=args.num_icp_points))
    
        self.poseGraphManager_.prev_node_idx = first_idx
        self.previous_scan_points = copy.deepcopy(cur_pts)
        self.icp_initial = np.eye(4)
        self.is_first_node = False
    def use_open3d_icp(self):
        
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(self.current_downsampled_points)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(self.previous_downsampled_points)

        reg_p2p = o3d.pipelines.registration.registration_icp(
                                                    source = source, 
                                                    target = target, 
                                                    max_correspondence_distance = 10, 
                                                    init = self.icp_initial, 
                                                    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
                                                    )
        return reg_p2p.transformation 
    
    def execute_slam_process(self):

        for cur_node_idx, scan_path in tqdm(enumerate(scan_paths), total=len(scan_paths), mininterval= 0.0001):
            # get current information     
            if self.is_first_node:
                self.initialize_first_node(cur_node_idx ,PointcloudUtils.readScan(scan_path))
                continue

            '''
            ic| current_scan_points: ndarray, shape=(123940, 3), dtype=float32
            current_downsampled_points: ndarray, shape=(5000, 3), dtype=float32

            '''

            self.current_scan_points = PointcloudUtils.readScan(scan_path) 
            self.current_downsampled_points = PointcloudUtils.random_sampling(self.current_scan_points, num_points=args.num_icp_points)
            self.previous_downsampled_points = PointcloudUtils.random_sampling(self.previous_scan_points, num_points=args.num_icp_points)
            
            self.poseGraphManager_.curr_node_idx = cur_node_idx 
            self.scanContextManager_.addNode(node_idx = self.poseGraphManager_.curr_node_idx, ptcloud=self.current_downsampled_points)


            if args.use_open3d: # calc odometry using custom ICP
                relative_pose = self.use_open3d_icp() 
            else:   # calc odometry using open3d
                relative_pose, _, _ = ICP.icp(self.current_downsampled_points, self.previous_downsampled_points, prev_relative_pose = self.icp_initial, max_iterations=20)

            # update the current (moved) pose 
            self.poseGraphManager_.curr_se3 = np.matmul(self.poseGraphManager_.curr_se3, relative_pose)
            self.icp_initial = relative_pose # assumption: constant velocity model (for better next ICP converges)
            # icp_initial = SE(3)
            
            # add the odometry factor to the graph 
            self.poseGraphManager_.addOdometryFactor(relative_pose)

            # renewal the prev information 
            self.poseGraphManager_.prev_node_idx = self.poseGraphManager_.curr_node_idx
            self.previous_scan_points = copy.deepcopy(self.current_scan_points)

            # loop detection and optimize the graph 
            if(self.poseGraphManager_.curr_node_idx > 1 and self.poseGraphManager_.curr_node_idx % args.try_gap_loop_detection == 0): 
                # 1/ loop detection 
                loop_idx, loop_dist, yaw_diff_deg = self.scanContextManager_.detectLoop()
                if(loop_idx == None): # NOT FOUND
                    pass
                else:
                    print("Loop event detected: ", self.poseGraphManager_.curr_node_idx, loop_idx, loop_dist)
                    # 2-1/ add the loop factor 
                    loop_scan_down_pts = self.scanContextManager_.getPtcloud(loop_idx)
                    loop_transform, _, _ = ICP.icp(self.current_downsampled_points, loop_scan_down_pts, yawdeg2se3(yaw_diff_deg), max_iterations=20)
                    self.poseGraphManager_.addLoopFactor(loop_transform, loop_idx)

                    # 2-2/ graph optimization 
                    self.poseGraphManager_.optimizePoseGraph()
                    self.poseGraphResultSaver_.OptimizedPoseGraphUpdate(self.poseGraphManager_.curr_node_idx, self.poseGraphManager_.graph_optimized)
            self.poseGraphResultSaver_.OptimizedPoseUpdate(self.poseGraphManager_.curr_se3)
            self.poseGraphResultSaver_.vizCurrentTrajectory(fig_idx=fig_idx)

if __name__ == "__main__":
    start = ICPSlamSystem()
    start.execute_slam_process()