#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:48:36 2023

@author: julian

Angle Stuff:
    https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
    https://stackoverflow.com/questions/61479191/convert-any-angle-to-the-interval-pi-pi
Plotting:
    https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib
Permutations:
    https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
"""
# %% Imports
import math
import itertools
import numpy as np
from numpy.random import default_rng
from sklearn.mixture import GaussianMixture
import gtsam
import matplotlib.pyplot as plt
import csv
import networkx as nx
from sam_slam_utils.sam_slam_helpers import angle_between_rads


# %% Functions

def read_csv_to_array(file_path):
    """
    Reads a CSV file and returns the contents as a 2D Numpy array.

    Parameters:
        file_path (str): The path to the CSV file to be read.

    Returns:
        numpy.ndarray: The contents of the CSV file as a 2D Numpy array.
    """
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        return np.array(data, dtype=np.float64)
    except:
        return -1


def show_graph(graph, result, poses, landmarks):
    G = nx.DiGraph()
    for i in range(graph.size()):
        factor = graph.at(i)
        for variable in factor.keys():
            if variable in x.values():
                pos = (result.atPose2(variable).x(), result.atPose2(variable).y())
                G.add_node(variable, pos=pos)
            else:
                pos = (result.atPoint2(variable)[0], result.atPoint2(variable)[1])
                G.add_node(variable, pos=pos)
            for variable2 in factor.keys():
                if variable != variable2:
                    G.add_edge(variable, variable2)

    # Plot the graph using matplotlib
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, )
    plt.show()


# %% Parameters
rng = default_rng()

# Data source
import_data = False

# Initial position noise
dist_sigma_initial = 0.01
ang_sigma_initial = np.pi / 200

# Motion noise
dist_sigma = 0.01
ang_sigma = np.pi / 200

# Detection settings
detect_range = 75
detect_half_window = np.pi / 32
detect_dist_sigma = 1
# This is meant to account for the detection window and also always using +/- pi in the bearing constraint
measurement_noise_factor = 1

# Data association setting
# GMM
# 

# Plotting options
plot_measured = True
plot_truth = True
plot_measured_detections = False
plot_measured_detections_connected = False
plot_true_detections = True  # TODO
plot_detection_associated = True
plot_show_slam = True
plot_graph = True

colors = ['orange', 'green', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']

# %% Data
if import_data:
    # Import the data
    poses_list = read_csv_to_array("../testing scripts/trajectory.csv")
    landmarks = read_csv_to_array("../testing scripts/landmarks.csv")
    N = len(poses_list)
    N_landmarks = len(landmarks)
else:
    # Generate the data 
    landmarks = np.array([[20, 15],
                          [20, 40],
                          [40, 15],
                          [40, 40]])

    N_landmarks = landmarks.shape[0]

    # Define trajectory 
    start = (5.0, 5.0)  # (x, y)
    right_steps = 50  # 50
    up_steps = 50
    step_size = 1.0

    right = [(start[0] + i * step_size, start[1], 0.0) for i in range(right_steps + 1)]

    # Up motion
    last_right = start[0] + right_steps * step_size
    up = [(last_right, start[1] + i * step_size, np.pi / 2) for i in range(1, up_steps + 1)]

    # Left motion
    last_up = start[1] + up_steps * step_size
    left = [(last_right - i * step_size, last_up, np.pi) for i in range(right_steps + 1)]

    poses_list = right + up + left
    N = len(poses_list)

# %% Process measured data, no noise
# Store ground truth in np array
measured_poses = np.asarray(poses_list)

# Calculate ground truth odometry
measured_poses_diff = measured_poses[1:, :] - measured_poses[:-1, :]

# Odometry,
# every step is just forward take the turn info from the pose difference
measured_odometry = np.zeros(measured_poses_diff.shape)

measured_odometry[:, 0] = (measured_poses_diff[:, 0] ** 2 + measured_poses_diff[:, 1] ** 2) ** (1 / 2)
measured_odometry[:, 2] = measured_poses_diff[:, 2]

# %% Generate noisy data

# Corrupt initial position
noise_pos_initial = rng.normal(0.0, dist_sigma_initial, (1, 2))
noise_ang_initial = rng.normal(0.0, ang_sigma_initial, (1, 1))
noise_initial = np.hstack((noise_pos_initial, noise_ang_initial))

initial_position = measured_poses[0, :] + noise_initial

# Generate noise to add to the odometry
noise_dist = rng.normal(0.0, dist_sigma, (measured_odometry.shape[0], 2))
noise_ang = rng.normal(0.0, ang_sigma, (measured_odometry.shape[0], 1))
# Combine distance and angular noise in to a single array
noise = np.hstack((noise_dist, noise_ang))
# TODO Account for sideways drift, for now I'm going to zero out the sideways
# motion of the odometry: [forward motion, sideways motion, angular motion]
noise[1, :] = 0

odometry = measured_odometry[:, :] + noise[:, :]

# Calculate position based on DR
true_poses = np.zeros(measured_poses.shape)
true_poses[0, :] = initial_position

for ind in range(odometry.shape[0]):
    init_x = true_poses[ind, 0]
    init_y = true_poses[ind, 1]
    init_theta = true_poses[ind, 2]

    odo_x = odometry[ind, 0]
    odo_y = odometry[ind, 1]
    odo_theta = odometry[ind, 2]

    travel_dist = (odo_x ** 2 + odo_y ** 2) ** (1 / 2)
    # slip_angle = math.atan2(odo_y, odo_x)

    # Calc heading and bearing
    # the remainder method will keep theta bound [-pi,pi]  
    heading = math.remainder(init_theta + odo_theta, 2 * np.pi)
    # bearing = math.remainder(init_theta + odo_theta + slip_angle, np.pi)

    next_x = init_x + travel_dist * math.cos(heading)  # math.cos(bearing)
    next_y = init_y + travel_dist * math.sin(heading)  # math.sin(bearing)
    next_theta = heading

    # Update true_poses 
    true_poses[ind + 1, 0] = next_x
    true_poses[ind + 1, 1] = next_y
    true_poses[ind + 1, 2] = next_theta

true_poses_xy = true_poses[:, 0:2]

# %% Generate Simulated Detections
# Between each true pose location, calculate the relative bearing between the agent and the landmarks(s)

# Detections are stored in a list o lists, [array index at which detection occurred, distance, and relative bearing,
# landmark id] The landmark shouldn't be used except for checking!!!
detections = []
for ind_gt in range(measured_poses.shape[0]):
    gt_state = true_poses[ind_gt, :]

    for ind_lm in range(landmarks.shape[0]):
        lm_state = landmarks[ind_lm, :]
        # Position rel to ground truth
        rel_x = lm_state[0] - gt_state[0]
        rel_y = lm_state[1] - gt_state[1]
        # bearing of lm w.r.t. the agent
        bearing = math.atan2(rel_y, rel_x)
        # relative bearing is the smallest angle between the bearing of the landmark and the heading of the agent
        rel_bearing = angle_between_rads(bearing, gt_state[2])
        # Check if the target is within the detection window on either side of the agent
        # currently this does not check for min/max range or anything else
        detect_dist = (rel_x ** 2 + rel_y ** 2) ** (1 / 2)
        if detect_dist > detect_range:
            continue

        if abs(rel_bearing) <= (np.pi / 2 + detect_half_window) and abs(rel_bearing) >= (
                np.pi / 2 - detect_half_window):
            detections.append([ind_gt, detect_dist, rel_bearing, ind_lm])

# %% Process Simulated Detection
# =============================================================================
# The list detections does not contain any noise and also stores the true 
# associations with the landmarks. Detections are detected using the true
# poses but are mapped relative to measured poses and noise is add.
# Detections format: [[array index at which detection occurred,
#                      distance, and 
#                      relative bearing,
#                      landmark id]]
#
# measured_detections format, (len(detections),4) sized np,array():
#   [measured pose ind, noisy range, +/- np.pi, landmark ind]
# =============================================================================

# Define some storage
detection_start_locations = np.zeros((len(detections), 2))
detection_end_locations = np.zeros((len(detections), 2))
measured_detections = np.zeros((len(detections), 4))

for ind_detect in range(len(detections)):
    ind_pose = detections[ind_detect][0]
    start_x = measured_poses[ind_pose, 0]
    start_y = measured_poses[ind_pose, 1]
    start_theta = measured_poses[ind_pose, 2]

    # Find the point in the map that corresponds to the detection
    # For this operation we are using out measured pose as the source
    # In ros this would be handled by tf2?

    # TODO I think we want the bearing to be assumed to be +/-pi
    if detections[ind_detect][2] > 0:
        detect_bearing_sign = 1
    else:
        detect_bearing_sign = -1

    # Detect_bearing is [-pi,pi] in relation to the map frame
    detect_bearing = math.remainder(start_theta + detect_bearing_sign * np.pi / 2, 2 * np.pi)

    Detection_range_noisy = detections[ind_detect][1] + rng.normal(0.0, detect_dist_sigma)

    detection_x = start_x + Detection_range_noisy * math.cos(detect_bearing)
    detection_y = start_y + Detection_range_noisy * math.sin(detect_bearing)

    # Populate detection_end_locations[] and detection_start_locations[]
    detection_start_locations[ind_detect, :] = start_x, start_y
    detection_end_locations[ind_detect, :] = detection_x, detection_y

    # Record info to measured_detections
    measured_detections[ind_detect, 0] = ind_pose
    measured_detections[ind_detect, 1] = Detection_range_noisy
    measured_detections[ind_detect, 2] = detect_bearing_sign * np.pi / 2

# %% Data Association Start
# =============================================================================
# Multiple methods are available maybe some combination could be used
# 1. GMM (clustering) - offline
# 2. Max likelihood - online/offline
# =============================================================================

# GMM method
# ==========
# Init the model
gmm = GaussianMixture(n_components=N_landmarks)

# fit and predict w.r.t. the detection data
data_associations = gmm.fit_predict(detection_end_locations)

# Record the data gmm means
data_association_means = gmm.means_
landmark_associations = gmm.predict(landmarks)

# %% Relate the landmarks with the association labels
# =============================================================================
# This is kind of done above by using the gmm to predict the landmark's
# DA category, landmark_associations. This can fail when the ground truth is
# scewed enough. A slightly more complicated approach is used below. What 
# configuration between gmm means and land mark priors minimized distance^2.
# =============================================================================


# Construct all permutations between landmark index and data association index
inds = [ind for ind in range(N_landmarks)]
permutations = [list(zip(x, inds)) for x in itertools.permutations(inds, len(inds))]

#
best_perm_score = np.inf
best_perm_ind = -1

for perm_ind, perm in enumerate(permutations):
    perm_score = 0
    for lm_ind, da_ind in perm:
        perm_score += (landmarks[lm_ind, 0] - data_association_means[da_ind, 0]) ** 2
        perm_score += (landmarks[lm_ind, 1] - data_association_means[da_ind, 1]) ** 2

    if perm_score < best_perm_score:
        best_perm_score = perm_score
        best_perm_ind = perm_ind

# Populate mappings between landmark ids and category ids

landmark2DA = -1 * np.ones((N_landmarks), dtype=np.int8)
DA2landmark = -1 * np.ones((N_landmarks), dtype=np.int8)

for lm_ind, da_ind in permutations[best_perm_ind]:
    landmark2DA[lm_ind] = da_ind
    DA2landmark[da_ind] = lm_ind

# %% Complete Data Association Process
# =============================================================================
# data_associations is (len(detections),) np.array. It's indexed with respect
# to the detection index and it contains the DA category of each detection.
# These categories need to be converted to the landmark ids.
# =============================================================================

for ind_detect in range(len(measured_detections)):
    measured_detections[ind_detect, 3] = DA2landmark[data_associations[ind_detect]]

# %% Factor graph
graph = gtsam.NonlinearFactorGraph()
l = {k: gtsam.symbol('l', k) for k in range(N_landmarks)}
x = {k: gtsam.symbol('x', k) for k in range(measured_poses.shape[0])}

# Add prior and prior's noise model
# prior noise model is defined by: dist_sigma_initial and ang_sigma_initial
# The initial pose is given by initial measured pose
prior_model = gtsam.noiseModel.Diagonal.Sigmas((dist_sigma_initial,
                                                dist_sigma_initial,
                                                ang_sigma_initial))

graph.add(gtsam.PriorFactorPose2(x[0], gtsam.Pose2(measured_poses[0, 0],
                                                   measured_poses[0, 1],
                                                   measured_poses[0, 2]),
                                 prior_model))

prior_model_lm = gtsam.noiseModel.Diagonal.Sigmas((dist_sigma_initial,
                                                   dist_sigma_initial))
for ind_landmark in range(N_landmarks):
    graph.add(gtsam.PriorFactorPoint2(l[ind_landmark],
                                      gtsam.Point2(landmarks[ind_landmark, 0],
                                                   landmarks[ind_landmark, 1]),
                                      prior_model_lm))

# Add between factors
# This is constructed using the measured odometry
# TODO add transverse noise 
odometry_model = gtsam.noiseModel.Diagonal.Sigmas((dist_sigma, 0, ang_sigma))

Between = gtsam.BetweenFactorPose2
for ind_odo in range(measured_odometry.shape[0]):
    graph.add(Between(x[ind_odo], x[ind_odo + 1], gtsam.Pose2(measured_odometry[ind_odo, 0],
                                                              measured_odometry[ind_odo, 1],
                                                              measured_odometry[ind_odo, 2]),
                      odometry_model))

# Add Range-Bearing measurements to two different landmarks L1 and L2
measurement_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([detect_dist_sigma,
                                                               detect_half_window / measurement_noise_factor]))
BR = gtsam.BearingRangeFactor2D

for i in range(measured_detections.shape[0]):
    pose_id, detection_range, detection_bearing, landmark_id = measured_detections[i, :]

    graph.add(BR(x[pose_id],
                 l[landmark_id],
                 gtsam.Rot2(detection_bearing),
                 detection_range,
                 measurement_model))

# Create the initial estimate, using measured poses
initial_estimate = gtsam.Values()
for ind_poses in range(measured_poses.shape[0]):
    initial_estimate.insert(x[ind_poses], gtsam.Pose2(measured_poses[ind_poses, 0],
                                                      measured_poses[ind_poses, 1],
                                                      measured_poses[ind_poses, 2]))

for ind_landmark in range(N_landmarks):
    initial_estimate.insert(l[ind_landmark], gtsam.Point2(landmarks[ind_landmark, 0],
                                                          landmarks[ind_landmark, 1]))
if plot_graph:
    show_graph(graph, initial_estimate, x, l)
    # show(graph, initial_estimate, binary_edges=True)

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
slam_result = optimizer.optimize()

if plot_graph:
    show_graph(graph, slam_result, x, l)

# Extract Pose2 from current_estimate
slam_out_Pose2 = np.zeros((len(x), 3))
for i in range(len(x)):
    # TODO there has to be a better way to do this!!
    slam_out_Pose2[i, 0] = slam_result.atPose2(x[i]).x()
    slam_out_Pose2[i, 1] = slam_result.atPose2(x[i]).y()
    slam_out_Pose2[i, 2] = slam_result.atPose2(x[i]).theta()

# %% Plot
fig, ax = plt.subplots()
ax.axis('equal')

# Landmarks and maybe some other fixed things
ax.scatter(landmarks[:, 0], landmarks[:, 1], color='k')

# Poses, measured and/or the ground truth
if plot_measured:
    ax.scatter(measured_poses[:, 0], measured_poses[:, 1], color='r')
    ax.axis('equal')
if plot_truth:
    ax.scatter(true_poses[:, 0], true_poses[:, 1], color='b')
    ax.axis('equal')

if plot_measured_detections_connected:
    ab_pairs = np.c_[detection_start_locations, detection_end_locations]
    ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
    ax.plot(*ab_args, c='k')

if plot_measured_detections:
    # mcolors.TABLEAU_COLORS
    ax.scatter(detection_end_locations[:, 0], detection_end_locations[:, 1], marker='x')

if plot_detection_associated:

    for ind_detect in range(detection_end_locations.shape[0]):
        component_num = data_associations[ind_detect]
        ax.scatter(detection_end_locations[ind_detect, 0],
                   detection_end_locations[ind_detect, 1],
                   color=colors[component_num % len(colors)],
                   marker='+')

    for ind_landmark in range(landmarks.shape[0]):
        component_num = landmark2DA[ind_landmark]  # landmark_associations[ind_landmark]
        ax.scatter(data_association_means[ind_landmark, 0],
                   data_association_means[ind_landmark, 1],
                   color=colors[ind_landmark % len(colors)],
                   marker='4')

        ax.scatter(landmarks[ind_landmark, 0],
                   landmarks[ind_landmark, 1],
                   color=colors[component_num % len(colors)])

if plot_show_slam:
    ax.scatter(slam_out_Pose2[:, 0], slam_out_Pose2[:, 1], color='k')

plt.show()
