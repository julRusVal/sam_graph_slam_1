#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation. This uses the process_online_2d_data class.
This class expects to be updated by a ROS node controlled by a sam_slam_listener instance.

The purpose of this script is to help with testing and reproduce the output from saved data.
This is copied from process_online_2d_data.py, and modified to work with real data. Ideally these two would be unified.

Originally I had intended this to be separate from ROS as much as possible...but as the processing moved toward ROS so
too did this script.


"""
import rospy
import numpy as np
import matplotlib.pyplot as plt
import gtsam

from sam_slam_utils.sam_slam_proc_classes import online_slam_2d, analyze_slam
from sam_slam_utils.sam_slam_helpers import read_csv_to_array
from sam_slam_utils.sam_slam_helpers import show_simple_graph_2d

import gtsam.utils.plot as gtsam_plot


def report_on_progress(graph: gtsam.NonlinearFactorGraph,
                       current_estimate: gtsam.Values,
                       x_keys: dict, b_keys: dict,
                       cur_dr_pose2: gtsam.Pose2,
                       cur_gt_pose2: gtsam.Pose2,
                       step_num: int,
                       est_detect_loc: np.ndarray = None,
                       true_detect_loc: np.ndarray = None,
                       detection_state: bool = False):
    """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""

    # Print the current estimates computed using iSAM2.
    # print("*" * 50 + f"\nInference after State {key_num + 1}:\n")
    # print(current_estimate)

    # Compute the marginals for all states in the graph.
    marginals = gtsam.Marginals(graph, current_estimate)

    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    axes = fig.gca()
    plt.cla()

    # Plot estimated poses, skip some for clarity
    i = 0
    while i in x_keys.keys() and current_estimate.exists(x_keys[i]):
        if i == step_num or i % 10 == 0:
            gtsam_plot.plot_pose2(0, current_estimate.atPose2(x_keys[i]), 0.75, marginals.marginalCovariance(x_keys[i]))
        i += 1

    # Plot estimated positions of landmarks
    for i_b in range(len(b_keys.keys())):
        gtsam_plot.plot_point2(fignum=0,
                               point=current_estimate.atPoint2(b_keys[i_b]),
                               linespec='-',
                               P=marginals.marginalCovariance(b_keys[i_b]))

    # Plot ground truth
    gtsam_plot.plot_pose2(0, cur_gt_pose2, 0.5)
    plt.scatter(cur_gt_pose2.x(), cur_gt_pose2.y(), c='b', linewidths=2)

    # Plot dead reckoning
    gtsam_plot.plot_pose2(0, cur_dr_pose2, 0.5)
    plt.scatter(cur_dr_pose2.x(), cur_dr_pose2.y(), c='r', linewidths=2)

    # Plot the detections w.r.t. estimate and ground truth
    if detection_state is True:
        plt.scatter(est_detect_loc[0], est_detect_loc[1], c='g', marker='+', linewidths=2)

        if true_detect_loc is not None:
            plt.scatter(true_detect_loc[0], true_detect_loc[1], c='b', marker='x', linewidths=2)

    axes.set_xlim(-5, 5)
    axes.set_ylim(-10, 10)
    plt.axis('equal')
    if detection_state:
        plt.title(f"Step: {step_num} - Detection: ")
    else:
        plt.title(f"Step: {step_num} - Odometry")
    plt.pause(.5)

# %% ROS node start
rospy.init_node('slam_listener', anonymous=False)

# %% Parameters

# === General parameters
# Linux path
path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing'
# Mac path
# path_name = '/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/clean run'

removal_rate = 0  # keep every nth element, set to zero to not reduce
last_index = np.inf  # Only the first n dr updates will be used

manual_associations = False

# Map parameters
# TODO the map/rope info needs to be add

# ROS parameters
# TODO update parameters, changes have added new params that are not being set
rospy.set_param("prior_ang_sig_deg", 10.0)
rospy.set_param("prior_dist_sig", 5.0)
rospy.set_param("buoy_dist_sig", 0.5)
rospy.set_param("odo_ang_sig_deg", 2.5)
rospy.set_param("odo_dist_sig", 1)
rospy.set_param("detect_ang_sig_deg", 0.5)
rospy.set_param("detect_dist_sig", 0.25)
rospy.set_param("manual_associations", manual_associations)

# %% Load data
# TODO load rope detection data
dr_poses_graph = read_csv_to_array(path_name + '/dr_poses_graph.csv')
gt_poses_graph = read_csv_to_array(path_name + '/gt_poses_graph.csv')
detections_graph = read_csv_to_array(path_name + '/detections_graph.csv')
associations_graph = read_csv_to_array(path_name + '/associations_graph.csv')
buoys = read_csv_to_array(path_name + '/buoys.csv')

# %% Preprocess data
"""
Remove some portion of the detections as well as their associated dr and gt elements
"""
if removal_rate > 0:
    removal_mask = [True if i % removal_rate != 0 else False for i in range(len(detections_graph))]
    dr_rem_inds = detections_graph[removal_mask, 6].astype(np.int16)

    # Remove detections with mask
    detections_original = detections_graph
    detections_graph = np.delete(detections_graph, removal_mask, axis=0)

    # Remove dr and gt associated with the removed detections
    inds = np.arange(len(dr_poses_graph)).reshape((len(dr_poses_graph), 1))
    dr_poses_graph = np.hstack((dr_poses_graph, inds))
    dr_poses_graph = np.delete(dr_poses_graph, dr_rem_inds, axis=0)
    gt_poses_graph = np.delete(gt_poses_graph, dr_rem_inds, axis=0)

# %% Initialize the online graph and send the buoy information
online_graph = online_slam_2d()
online_graph.buoy_setup(buoys)

# %% First
online_graph.add_first_pose(dr_poses_graph[0, :], gt_poses_graph[0, :])

detection_state = False

for i in range(1, min(last_index + 1, len(dr_poses_graph))):
    # Determine if a detection occurred
    """
    If the data has been preprocessed its will have a width of 8 from the added original index
    """
    if dr_poses_graph.shape[1] == 8:
        dr_id = dr_poses_graph[i, 7]
    else:
        dr_id = i

    if dr_id in detections_graph[:, 6]:
        print(f"Detection update: {i}")
        detection_state = True
        detection_id = int(np.where(detections_graph[:, 6] == dr_id)[0][0])
        # Use relative detection
        detection = np.array((detections_graph[detection_id, 3], detections_graph[detection_id, 4]), dtype=np.float64)

        da_id = int(associations_graph[detection_id][0])

        if da_id == -2:  # -2 corresponds to the ID of a buoy detection
            online_graph.online_update(dr_poses_graph[i, :],
                                       gt_poses_graph[i, :],
                                       detection,
                                       da_id=da_id)

        elif manual_associations and da_id in range(len(buoys)):
            online_graph.online_update(dr_poses_graph[i, :],
                                       gt_poses_graph[i, :],
                                       detection,
                                       da_id=da_id)
        else:
            online_graph.online_update(dr_poses_graph[i, :],
                                       gt_poses_graph[i, :],
                                       detection)

    else:
        print(f"Odometry update: {i}")
        online_graph.online_update(dr_poses_graph[i, :],
                                   gt_poses_graph[i, :])

    # TODO: Removed true detect location but can use manual associations
    report_on_progress(graph=online_graph.graph,
                       current_estimate=online_graph.current_estimate,
                       x_keys=online_graph.x,
                       b_keys=online_graph.b,
                       cur_dr_pose2=online_graph.dr_Pose2s[-1],
                       cur_gt_pose2=online_graph.gt_Pose2s[-1],
                       step_num=i,
                       est_detect_loc=online_graph.est_detect_loc,
                       true_detect_loc=None,  # online_graph.true_detect_loc
                       detection_state=detection_state)

    detection_state = False

# %% Analysis
analysis = analyze_slam(online_graph)
analysis.visualize_posterior()
analysis.show_error()
analysis.show_graph_2d("Final Graph")

# %% Basic plot
# show_simple_graph_2d(graph=online_graph.graph,
#                      x_keys=online_graph.x,
#                      b_keys=online_graph.b,
#                      values=online_graph.current_estimate,
#                      label="Online SLAM")
