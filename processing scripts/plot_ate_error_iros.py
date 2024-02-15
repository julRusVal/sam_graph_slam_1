import trajectory_analysis
import numpy as np
import os

# Plotting Settings
dr_color = 'r'
gt_color = 'b'
post_color = 'g'
online_color = 'm'
rope_color = 'b'
buoy_color = 'k'
title_size = 16
legend_size = 12
label_size = 14

plot = True

# ATE setting
offset = 0.0
scale = 1.0
max_difference = 0.0001
verbose = True

# Data settings
iros_data = True

# Paths and data loading
if iros_data:
    # This will load the IROS data
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # method 1
    method_1_path = script_directory + "/data/iros_method_1"
    method_1_dr_path = method_1_path + "/analysis_dr_3d.csv"
    method_1_final_path = method_1_path + "/analysis_final_3d.csv"
    method_1_online_path = method_1_path + "/analysis_online_3d.csv"

    method_1_error = np.genfromtxt(method_1_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_1_detections = np.genfromtxt(method_1_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 2
    method_2_path = script_directory + "/data/iros_method_2"
    method_2_dr_path = method_2_path + "/analysis_dr_3d.csv"
    method_2_final_path = method_2_path + "/analysis_final_3d.csv"
    method_2_online_path = method_2_path + "/analysis_online_3d.csv"

    method_2_error = np.genfromtxt(method_2_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_2_detections = np.genfromtxt(method_2_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 1
    method_3_path = script_directory + "/data/iros_method_3"
    method_3_dr_path = method_3_path + "/analysis_dr_3d.csv"
    method_3_final_path = method_3_path + "/analysis_final_3d.csv"
    method_3_online_path = method_3_path + "/analysis_online_3d.csv"

    method_3_error = np.genfromtxt(method_3_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_3_detections = np.genfromtxt(method_3_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)
else:
    raise ValueError("Please specify a directory")

method_1_final_dr_ate = trajectory_analysis.TrajectoryAnalysis(method_1_final_path,
                                                               method_1_dr_path,
                                                               output_directory_path=method_1_path,
                                                               verbose=verbose)

method_2_final_dr_ate = trajectory_analysis.TrajectoryAnalysis(method_2_final_path,
                                                               method_2_dr_path,
                                                               output_directory_path=method_2_path,
                                                               verbose=verbose)

method_3_final_dr_ate = trajectory_analysis.TrajectoryAnalysis(method_3_final_path,
                                                               method_3_dr_path,
                                                               output_directory_path=method_3_path,
                                                               verbose=verbose)

if plot:
    print("plotting")
    method_1_final_dr_ate.plot_trajectories()
