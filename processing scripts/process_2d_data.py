#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation. This uses the offline_slam_2d class.
This class can except a path to the needed data saved in separate CSVs, or it can be passed a sam_slam_listener instance.

The purpose of this script is to help with testing and reproduce the output from saved data.
"""

# %% Imports
from sam_slam_utils.sam_slam_proc_classes import offline_slam_2d
from sam_slam_utils.sam_slam_proc_classes import analyze_slam

# %% Main
if __name__ == "__main__":
    # ===== Pick data source =====
    # linux path
    # path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data'
    # Mac path
    path_name = '/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/clean run'

    # ===== Perform offline slam ====
    process = offline_slam_2d(path_name)
    process.perform_offline_slam()

    if True:
        process.visualize_clustering()
        process.visualize_raw()
        process.visualize_posterior()
    if True:
        process.show_graph_2d('Initial', False)
        process.show_graph_2d('Final', True)
    if True:
        process.show_error()

    # %%
    # ===== Analysis class =====
    analysis = analyze_slam(process)
    # analysis.visualize_raw()
    # analysis.visualize_posterior()