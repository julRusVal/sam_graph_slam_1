#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation. This uses the process_online_2d_data class.
This class can expects to be updated by a ROS node controlled by a sam_slam_listener instance. This script

The purpose of this script is to help with testing and reproduce the output from saved data.
"""
import numpy as np

from sam_slam_utils.sam_slam_helper_funcs import read_csv_to_array
from sam_slam_utils.sam_slam_proc_classes import online_slam_2d
from sam_slam_utils.sam_slam_helper_funcs import show_simple_graph_2d


# %% Load data
path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data'
dr_poses_graph = read_csv_to_array(path_name + '/dr_poses_graph.csv')
gt_poses_graph = read_csv_to_array(path_name + '/gt_poses_graph.csv')
detections_graph = read_csv_to_array(path_name + '/detections_graph.csv')
buoys = read_csv_to_array(path_name + '/buoys.csv')

online_graph = online_slam_2d()

# %% buoys
online_graph.buoy_setup(buoys)

# %% First
online_graph.add_first_pose(dr_poses_graph[0, :], gt_poses_graph[0, :])

for i in range(1, len(dr_poses_graph)):
    # Determine if a detection occurred
    if i in detections_graph[:, 6]:
        print(f"Detection update: {i}")
        id_detection = int(np.where(detections_graph[:, 6] == i)[0][0])
        detection = np.array((detections_graph[id_detection, 3], detections_graph[id_detection, 4]), dtype=np.float64)

        online_graph.online_update(dr_poses_graph[i, :],
                                   gt_poses_graph[i, :],
                                   detection)

    else:
        print(f"Odometry update: {i}")
        online_graph.online_update(dr_poses_graph[i, :],
                                   gt_poses_graph[i, :])

# %% Basic plot
show_simple_graph_2d(graph=online_graph.graph,
                     x_keys=online_graph.x,
                     b_keys=online_graph.b,
                     values=online_graph.current_estimate,
                     label="Online SLAM")
