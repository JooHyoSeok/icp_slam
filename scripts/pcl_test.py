#!/usr/bin/env python3
# coding: utf-8

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import pcl
import pcl_helper

def do_passthrough(pcl_data,filter_axis,axis_min,axis_max):
    
    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

def do_ransac_plane_normal_segmentation(point_cloud, input_max_distance):
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

def callback(input_ros_msg):

    cloud = pcl_helper.ros_to_pcl(input_ros_msg)
    print("Input :", cloud, type(cloud))

    # 실행 코드 부분 
    cloud = do_passthrough(cloud, 'x', 1.0, 10.0)
    cloud = do_passthrough(cloud, 'y', -5.0, 5.0)
    _, _, cloud = do_ransac_plane_normal_segmentation(cloud, 0.05)

    cloud_new = pcl_helper.pcl_to_ros(cloud) #PCL을 ROS 메시지로 변경     
    pub.publish(cloud_new)

if __name__ == "__main__":
    rospy.init_node('tutorial', anonymous=True)
    rospy.Subscriber('/points', PointCloud2, callback)

    pub = rospy.Publisher("/velodyne_points_new", PointCloud2, queue_size=1)

    rospy.spin()