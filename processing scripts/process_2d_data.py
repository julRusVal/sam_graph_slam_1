#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation
"""

# %% Imports
from sam_slam_utils.sam_slam_proc_classes import process_2d_data

# %% Main
if __name__ == "__main__":
    path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data'
    # path_name = '/Users/julian/Library/CloudStorage/Dropbox/Degree coding/simulated_slam/data'
    process = process_2d_data(path_name)

    process.correct_coord_problem()
    process.cluster_data()
    process.cluster_to_landmark()
    process.convert_poses_to_Pose2()
    process.Bearing_range_from_detection_2d()
    process.construct_graph_2d()
    process.optimize_graph()

    if True:
        process.visualize_clustering()
        process.visualize_raw()
        process.visualize_posterior()
    if True:
        process.show_graph_2d('Initial', False)
        process.show_graph_2d('Final', True)
    if True:
        process.show_error()
