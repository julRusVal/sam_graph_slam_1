#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation
"""

# %% Imports
from sam_slam_utils.sam_slam_proc_classes import process_2d_data

# %% Main
if __name__ == "__main__":

    path_name = '/Users/julian/Library/CloudStorage/Dropbox/Degree coding/simulated_slam/data'
    process = process_2d_data(path_name)

    process.correct_coord_problem()
    process.cluster_data()
    process.cluster_to_landmark()
    process.convert_poses_to_Pose2()
    process.Bearing_range_from_detection_2D()
    process.construct_graph_2D()
    process.optimize_graph()

    if False:
        process.visualize_clustering()
        process.visualize_raw()
        process.visualize_posterior()
    if False:
        process.show_graph_2D('Initial', False)
        process.show_graph_2D('Final', True)
    if True:
        process.show_error()
