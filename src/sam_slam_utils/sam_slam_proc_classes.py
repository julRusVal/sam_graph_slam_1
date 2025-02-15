#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation
"""

# %% Imports
from __future__ import annotations
import itertools
import queue
import os
import ast

# Maths
import numpy as np
import scipy
from scipy import stats
import math
import matplotlib.pyplot as plt

# Clustering
# from sklearn.mixture import GaussianMixture  # Remove for jetson compatibility

# Graphing graphs
import networkx as nx

# Slam
import gtsam
import gtsam.utils.plot as gtsam_plot

# ROS
import rospy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# SMaRC
from sss_object_detection.consts import ObjectID

# sam_slam
from sam_slam_utils.sam_slam_helpers import calc_pose_error
from sam_slam_utils.sam_slam_helpers import create_Pose2, pose2_list_to_nparray
from sam_slam_utils.sam_slam_helpers import create_Pose3, merge_into_Pose3
from sam_slam_utils.sam_slam_helpers import read_csv_to_array, write_array_to_csv


# %% Functions

def correct_dr(uncorrected_dr: gtsam.Pose2):
    """
    This is part of the gt/dr mismatch. Its appears that there is something off with converting from
    sam/base_link from to the map frame, the results are mirrored about the y-axis.
    This function will mirror the input pose about the y-axis.
    """
    # return gtsam.Pose2(x=-uncorrected_dr.x(),
    #                    y=uncorrected_dr.y(),
    #                    theta=np.pi - uncorrected_dr.theta())

    return uncorrected_dr


def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    return angle


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_center(x1, y1, x2, y2):
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    return np.array((x_center, y_center), dtype=np.float64)


# %% Classes

class offline_slam_2d:
    def __init__(self, input_data=None):
        """
        Input can either be read from a file, if the path to a folder is provided as a string
        Alternatively, data can be extracted from an instance of sam_slam_listener.
        The data in sam_slam_listener is stored in lists

        Input formats
        dr_poses_graph: [[x, y, z, q_w, q_x, q_y, q_z]]
        gt_poses_graph: [[x, y, z, q_w, q_x, q_y, q_z]] {the sign of y needs to be flipped and theta adjusted by pi}
        detections_graph: [[x_map, y_map, z_map, x_rel, y_rel, z_rel, index of dr]]
        buoys:[[x, y, z]]
        """
        # Load data from a files
        if input_data is None:
            self.dr_poses_graph = read_csv_to_array('dr_poses_graph.csv')
            self.gt_poses_graph = read_csv_to_array('gt_poses_graph.csv')
            self.detections_graph = read_csv_to_array('detections_graph.csv')
            self.buoy_priors = read_csv_to_array('buoys.csv')

        elif isinstance(input_data, str):
            self.dr_poses_graph = read_csv_to_array(input_data + '/dr_poses_graph.csv')
            self.gt_poses_graph = read_csv_to_array(input_data + '/gt_poses_graph.csv')
            self.detections_graph = read_csv_to_array(input_data + '/detections_graph.csv')
            self.buoy_priors = read_csv_to_array(input_data + '/buoys.csv')

        # Extract data from an instance of sam_slam_listener
        else:
            self.dr_poses_graph = np.array(input_data.dr_poses_graph)
            self.gt_poses_graph = np.array(input_data.gt_poses_graph)
            self.detections_graph = np.array(input_data.detections_graph)
            self.buoy_priors = np.array(input_data.buoys)

        # ===== Clustering and data association =====
        self.n_buoys = len(self.buoy_priors)
        self.n_detections = self.detections_graph.shape[0]
        self.cluster_model = None
        self.cluster_mean_threshold = 2.0  # means within this threshold will cause fewer clusters to be used
        self.n_clusters = -1
        self.detection_clusterings = None
        self.buoy2cluster = None
        self.cluster2buoy = None

        # ===== Graph parameters =====
        self.graph = None
        self.x = None
        self.b = None
        self.dr_Pose2s = None
        self.gt_Pose2s = None
        self.between_Pose2s = None
        self.post_Pose2s = None
        self.post_Point2s = None
        self.bearings_ranges = []
        # TODO this will need more processing to make into da_check
        self.da_check_proto = []
        self.detect_locs = None

        # ===== Agent prior sigmas =====
        self.ang_sig_init = 5 * np.pi / 180
        self.dist_sig_init = 1
        # buoy prior sigmas
        self.buoy_dist_sig_init = 2.5
        # agent odometry sigmas
        self.ang_sig = 5 * np.pi / 180
        self.dist_sig = .25
        # detection sigmas
        self.detect_dist_sig = 1
        self.detect_ang_sig = 5 * np.pi / 180

        # ===== Optimizer and values =====
        self.optimizer = None
        self.initial_estimate = None
        self.current_estimate = None

        # ===== Visualization =====
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.colors = ['orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        self.plot_limits = [-15.0, 15.0, -2.5, 25.0]

    # ===== Visualization methods =====
    def visualize_raw(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        plt.title(f'Raw data\n ground truth ({self.gt_color}) and dead reckoning ({self.dr_color})')
        plt.axis(self.plot_limits)
        plt.grid(True)

        if self.n_detections > 0:
            ax.scatter(self.detections_graph[:, 0], self.detections_graph[:, 1], color='k')

        ax.scatter(self.gt_poses_graph[:, 0], self.gt_poses_graph[:, 1], color=self.gt_color)
        # TODO this whole method needs to be moved to the analysis
        # negative sign to fix coordinate problem
        ax.scatter(-self.dr_poses_graph[:, 0], self.dr_poses_graph[:, 1], color=self.dr_color)

        plt.show()
        return

    def visualize_clustering(self):
        # ===== Plot detected clusters =====
        fig, ax = plt.subplots()
        plt.title(f'Clusters\n{self.n_clusters} Detected')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)

        for cluster in range(self.n_clusters):
            inds = self.detection_clusterings == cluster
            ax.scatter(self.detections_graph[inds, 0],
                       self.detections_graph[inds, 1],
                       color=self.colors[cluster % len(self.colors)])

        plt.show()

        # ===== Plot true buoy locations w/ cluster means ====
        fig, ax = plt.subplots()
        plt.title('Buoys\nTrue buoy positions and associations\ncluster means')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)

        for ind_buoy in range(self.buoy_priors.shape[0]):
            cluster_num = self.buoy2cluster[ind_buoy]  # landmark_associations[ind_landmark]
            if cluster_num == -1:
                current_color = 'k'
            else:
                current_color = self.colors[cluster_num % len(self.colors)]
            # not all buoys have an associated have an associated cluster
            if cluster_num >= 0:
                ax.scatter(self.cluster_model.means_[cluster_num, 0],
                           self.cluster_model.means_[cluster_num, 1],
                           color=current_color,
                           marker='+',
                           s=75)

            ax.scatter(self.buoy_priors[ind_buoy, 0],
                       self.buoy_priors[ind_buoy, 1],
                       color=current_color)

        plt.show()
        return

    def visualize_posterior(self, plot_gt=True, plot_dr=True, plot_buoy=True):
        """
        Visualize The Posterior
        """
        # Check if Optimization has occurred
        if self.current_estimate is None:
            print('Need to perform optimization before it can be printed!')
            return

        # Build array for the pose and point posteriors
        slam_out_poses = np.zeros((len(self.x), 2))
        slam_out_points = np.zeros((len(self.b), 2))
        for i in range(len(self.x)):
            # TODO there has to be a better way to do this!!
            slam_out_poses[i, 0] = self.current_estimate.atPose2(self.x[i]).x()
            slam_out_poses[i, 1] = self.current_estimate.atPose2(self.x[i]).y()

        for i in range(len(self.b)):
            slam_out_points[i, 0] = self.current_estimate.atPoint2(self.b[i])[0]
            slam_out_points[i, 1] = self.current_estimate.atPoint2(self.b[i])[1]

        # ===== Matplotlip options =====
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        plt.title(f'Posterior\nG.T.({self.gt_color}), D.R.({self.dr_color}), Posterior({self.post_color})')
        plt.axis(self.plot_limits)
        plt.grid(True)

        # ==== Plot ground truth =====
        if plot_gt:
            ax.scatter(self.gt_poses_graph[:, 0],
                       self.gt_poses_graph[:, 1],
                       color=self.gt_color)

        # ===== Plot dead reckoning =====
        # TODO this whole method needs to be moved to the analysis
        # negative sign to fix coordinate problem
        if plot_dr:
            ax.scatter(-self.dr_poses_graph[:, 0],
                       self.dr_poses_graph[:, 1],
                       color=self.dr_color)

        # ===== Plot buoys w/ cluster colors =====
        if plot_buoy:
            # Plot the true location of the buoys
            for ind_buoy in range(self.n_buoys):
                # Determine cluster color
                cluster_num = self.buoy2cluster[ind_buoy]
                if cluster_num == -1:
                    current_color = 'k'
                else:
                    current_color = self.colors[cluster_num % len(self.colors)]

                # Plot all the buoys
                ax.scatter(self.buoy_priors[ind_buoy, 0],
                           self.buoy_priors[ind_buoy, 1],
                           color=current_color)

                # Plot buoy posteriors
                ax.scatter(slam_out_points[ind_buoy, 0],
                           slam_out_points[ind_buoy, 1],
                           color=current_color,
                           marker='+',
                           s=75)

        # Plot the posterior
        ax.scatter(slam_out_poses[:, 0], slam_out_poses[:, 1], color='g')

        plt.show()

    def show_graph_2d(self, label, show_final=True):
        """

        """
        # Select which values to graph
        if show_final:
            if self.current_estimate is None:
                print('Perform optimization before it can be graphed')
                return
            values = self.current_estimate
        else:
            if self.initial_estimate is None:
                print('Initialize estimate before it can be graphed')
                return
            values = self.initial_estimate
        # Initialize network
        G = nx.Graph()
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            for key_id, key in enumerate(factor.keys()):
                # Test if key corresponds to a pose
                if key in self.x.values():
                    pos = (values.atPose2(key).x(), values.atPose2(key).y())
                    G.add_node(key, pos=pos, color='black')

                # Test if key corresponds to points
                elif key in self.b.values():
                    pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])

                    # Set color according to clustering
                    if self.buoy2cluster is None:
                        node_color = 'black'
                    else:
                        # Find the buoy index -> cluster index -> cluster color
                        buoy_id = list(self.b.values()).index(key)
                        cluster_id = self.buoy2cluster[buoy_id]
                        # A negative cluster id indicates that the buoy was not assigned a cluster
                        if cluster_id < 0:
                            node_color = 'black'
                        else:
                            node_color = self.colors[cluster_id % len(self.colors)]
                    G.add_node(key, pos=pos, color=node_color)
                else:
                    print('There was a problem with a factor not corresponding to an available key')

                # Add edges that represent binary factor: Odometry or detection
                for key_2_id, key_2 in enumerate(factor.keys()):
                    if key != key_2 and key_id < key_2_id:
                        # detections will have key corresponding to a landmark
                        if key in self.b.values() or key_2 in self.b.values():
                            G.add_edge(key, key_2, color='red')
                        else:
                            G.add_edge(key, key_2, color='blue')

        # ===== Plot the graph using matplotlib =====
        # Matplotlib options
        fig, ax = plt.subplots()
        plt.title(f'Factor Graph\n{label}')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)
        plt.xticks(np.arange(self.plot_limits[0], self.plot_limits[1] + 1, 2.5))

        # Networkx Options
        pos = nx.get_node_attributes(G, 'pos')
        e_colors = nx.get_edge_attributes(G, 'color').values()
        n_colors = nx.get_node_attributes(G, 'color').values()
        options = {'node_size': 25, 'width': 3, 'with_labels': False}

        # Plot
        nx.draw_networkx(G, pos, edge_color=e_colors, node_color=n_colors, **options)
        np.arange(self.plot_limits[0], self.plot_limits[1] + 1, 2.5)
        plt.show()

    def show_error(self):
        # Convert the lists of Pose2s to np arrays
        dr_array = pose2_list_to_nparray(self.dr_Pose2s)
        gt_array = pose2_list_to_nparray(self.gt_Pose2s)
        post_array = pose2_list_to_nparray(self.post_Pose2s)

        # TODO figure out ground truth coordinate stuff
        # This is to correct problems with the way the gt pose is converted to the map frame...
        gt_array[:, 2] = np.pi - gt_array[:, 2]

        # Find the errors between gt<->dr and gt<->post
        dr_error = calc_pose_error(dr_array, gt_array)
        post_error = calc_pose_error(post_array, gt_array)

        # Calculate MSE
        dr_mse_error = np.square(dr_error).mean(0)
        post_mse_error = np.square(post_error).mean(0)

        # ===== Plot =====
        fig, (ax_x, ax_y, ax_t) = plt.subplots(1, 3)
        # X error
        ax_x.plot(dr_error[:, 0], self.dr_color)
        ax_x.plot(post_error[:, 0], self.post_color)
        ax_x.title.set_text(f'X Error\nD.R. MSE: {dr_mse_error[0]:.4f}\n Posterior MSE: {post_mse_error[0]:.4f}')
        # Y error
        ax_y.plot(dr_error[:, 1], self.dr_color)
        ax_y.plot(post_error[:, 1], self.post_color)
        ax_y.title.set_text(f'Y Error\nD.R. MSE: {dr_mse_error[1]:.4f}\n Posterior MSE: {post_mse_error[1]:.4f}')
        # Theta error
        ax_t.plot(dr_error[:, 2], self.dr_color)
        ax_t.plot(post_error[:, 2], self.post_color)
        ax_t.title.set_text(f'Theta Error\nD.R. MSE: {dr_mse_error[2]:.4f}\n Posterior MSE: {post_mse_error[2]:.4f}')

        plt.show()

    # ===== Clustering and data association methods =====
    def fit_cluster_model(self):
        # Check for empty detections_graph
        if self.n_detections < 1:
            print("No detections were detected, improve detector")
            return

        if self.cluster_model is not None:
            # TODO changed to make it work with corrected dr poses
            self.detection_clusterings = self.cluster_model.fit_predict(self.detect_locs[:, 0:2])

        else:
            print('Need to initialize cluster_model first')

    def cluster_data(self):
        # =============================================================================
        # Multiple methods are available maybe some combination could be used
        # 1. GMM (clustering) - offline
        # 2. Max likelihood - online/offline
        # =============================================================================

        # Init the model
        self.n_clusters = self.n_buoys
        self.cluster_model = GaussianMixture(n_components=self.n_clusters)
        # fit and predict w.r.t. the detection data
        self.fit_cluster_model()

        # check for missed buoys, if detected redo the
        # TODO this can remove to many clusters
        indices = list(range(self.n_clusters))
        for pair in itertools.combinations(indices, 2):
            mean_a = self.cluster_model.means_[pair[0]]
            mean_b = self.cluster_model.means_[pair[1]]

            dist = ((mean_a[0] - mean_b[0]) ** 2 + (mean_a[1] - mean_b[1]) ** 2)
            if dist < self.cluster_mean_threshold ** 2:
                if self.n_clusters > 1:
                    self.n_clusters -= 1

        if self.n_clusters != self.n_buoys:
            self.cluster_model = GaussianMixture(n_components=self.n_clusters)
            self.fit_cluster_model()

    def cluster_to_landmark(self):
        # Use least squares to find the best mapping of clusters onto landmarks
        # All permutation of buoy_ids and cluster_ids are tested
        # for len(buoy_ids) >= len(cluster_ids)
        buoy_ids = list(range(self.n_buoys))
        cluster_ids = list(range(self.n_clusters))
        permutations = [list(zip(x, cluster_ids)) for x in itertools.permutations(buoy_ids, len(cluster_ids))]

        #
        best_perm_score = np.inf
        best_perm_ind = -1

        for perm_ind, perm in enumerate(permutations):
            perm_score = 0
            for buoy_id, cluster_id in perm:
                perm_score += (self.buoy_priors[buoy_id, 0] - self.cluster_model.means_[cluster_id, 0]) ** 2
                perm_score += (self.buoy_priors[buoy_id, 1] - self.cluster_model.means_[cluster_id, 1]) ** 2

            if perm_score < best_perm_score:
                best_perm_score = perm_score
                best_perm_ind = perm_ind

        # Populate mappings between landmark ids and category ids
        self.buoy2cluster = -1 * np.ones(self.n_buoys, dtype=np.int8)
        self.cluster2buoy = -1 * np.ones(self.n_buoys, dtype=np.int8)

        for buoy_id, cluster_id in permutations[best_perm_ind]:
            self.buoy2cluster[buoy_id] = cluster_id
            self.cluster2buoy[cluster_id] = buoy_id

    # ===== GTSAM data processing =====
    def convert_poses_to_Pose2(self):
        """
        Poses is self.
        [x,y,z,q_w,q_x,q_,y,q_z]
        """
        self.dr_Pose2s = []
        self.gt_Pose2s = []
        self.between_Pose2s = []

        # ===== DR =====
        for dr_pose in self.dr_poses_graph:
            self.dr_Pose2s.append(correct_dr(create_Pose2(dr_pose)))

        # ===== GT =====
        for gt_pose in self.gt_poses_graph:
            self.gt_Pose2s.append(create_Pose2(gt_pose))

        # ===== DR between =====
        for i in range(1, len(self.dr_Pose2s)):
            between_odometry = self.dr_Pose2s[i - 1].between(self.dr_Pose2s[i])
            self.between_Pose2s.append(between_odometry)

    def Bearing_range_from_detection_2d(self):
        self.detect_locs = np.zeros((self.n_detections, 5))
        for i_d, detection in enumerate(self.detections_graph):
            dr_id = int(detection[-1])
            detection_pose = self.dr_Pose2s[dr_id]
            true_pose = self.gt_Pose2s[dr_id]
            # This Method uses the map coordinates to calc bearing and range
            est_detect_loc = detection_pose.transformFrom(np.array(detection[3:5], dtype=np.float64))
            true_detect_loc = true_pose.transformFrom(np.array(detection[3:5], dtype=np.float64))

            measurement = gtsam.BearingRange2D.Measure(pose=detection_pose, point=est_detect_loc)
            # measurement = gtsam.BearingRange2D.Measure(pose=detection_pose, point=detection[0:2])
            # This method uses the relative position of the detection, as it is registered in sam/base_link
            # pose_null = self.create_Pose2([0, 0, 0, 1, 0, 0, 0])
            # measurement = gtsam.BearingRange3D.Measure(pose_null, detection[3:5])
            self.bearings_ranges.append(measurement)

            # ===== Debugging =====
            # self.da_check_proto[dr_id] = np.hstack((est_detect_loc, true_detect_loc))
            self.da_check_proto.append(np.hstack((est_detect_loc, true_detect_loc)))
            self.detect_locs[i_d, :] = [est_detect_loc[0], est_detect_loc[1],
                                        true_detect_loc[0], true_detect_loc[1],
                                        dr_id]

    def construct_graph_2d(self):
        """
        Graph made up of gtsam.Pose2 and gtsam.Point2
        """
        self.graph = gtsam.NonlinearFactorGraph()

        # labels
        self.b = {k: gtsam.symbol('b', k) for k in range(self.n_buoys)}
        self.x = {k: gtsam.symbol('x', k) for k in range(len(self.dr_poses_graph))}

        # ===== Prior factors =====
        # Agent pose
        prior_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.dist_sig_init,
                                                                 self.dist_sig_init,
                                                                 self.ang_sig_init]))

        self.graph.add(gtsam.PriorFactorPose2(self.x[0], self.dr_Pose2s[0], prior_model))

        # Buoys
        prior_model_lm = gtsam.noiseModel.Diagonal.Sigmas((self.buoy_dist_sig_init, self.buoy_dist_sig_init))

        for id_buoy in range(self.n_buoys):
            self.graph.add(gtsam.PriorFactorPoint2(self.b[id_buoy],
                                                   np.array((self.buoy_priors[id_buoy, 0],
                                                             self.buoy_priors[id_buoy, 1]),
                                                            dtype=np.float64),
                                                   prior_model_lm))

        # ===== Odometry Factors =====
        odometry_model = gtsam.noiseModel.Diagonal.Sigmas((self.dist_sig, self.dist_sig, self.ang_sig))

        for pose_id in range(len(self.dr_Pose2s) - 1):
            between_Pose2 = self.dr_Pose2s[pose_id].between(self.dr_Pose2s[pose_id + 1])
            self.graph.add(
                gtsam.BetweenFactorPose2(self.x[pose_id], self.x[pose_id + 1], between_Pose2, odometry_model))

        # ===== Detection Factors =====
        detection_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.detect_dist_sig, self.detect_ang_sig]))

        if self.n_detections > 0:
            for det_id, detection in enumerate(self.detections_graph):
                dr_id = detection[-1]
                buoy_id = self.cluster2buoy[self.detection_clusterings[det_id]]
                # check for a association problem
                if buoy_id < 0:
                    continue
                self.graph.add(gtsam.BearingRangeFactor2D(self.x[dr_id],
                                                          self.b[buoy_id],
                                                          self.bearings_ranges[det_id].bearing(),
                                                          self.bearings_ranges[det_id].range(),
                                                          detection_model))

        # Create the initial estimate, using measured poses
        self.initial_estimate = gtsam.Values()
        for pose_id, dr_Pose2 in enumerate(self.dr_Pose2s):
            self.initial_estimate.insert(self.x[pose_id], dr_Pose2)

        for buoy_id in range(self.n_buoys):
            self.initial_estimate.insert(self.b[buoy_id],
                                         np.array((self.buoy_priors[buoy_id, 0], self.buoy_priors[buoy_id, 1]),
                                                  dtype=np.float64))

    def optimize_graph(self):
        if self.graph.size() == 0:
            print('Need to build the graph before is can be optimized!')
            return
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.current_estimate = self.optimizer.optimize()

        # Save the posterior results
        self.post_Pose2s = []
        self.post_Point2s = []

        for i in range(len(self.x)):
            self.post_Pose2s.append(self.current_estimate.atPose2(self.x[i]))

        for i in range(len(self.b)):
            self.post_Point2s.append(self.current_estimate.atPoint2(self.b[i]))

    # ===== Higher level methods =====
    def perform_offline_slam(self):
        self.convert_poses_to_Pose2()
        self.Bearing_range_from_detection_2d()
        self.cluster_data()
        self.cluster_to_landmark()
        self.construct_graph_2d()
        self.optimize_graph()

    def output_results(self, verbose_level=1):
        """

        """
        if verbose_level >= 1:
            self.visualize_clustering()
            self.visualize_raw()
            self.visualize_posterior()
        if verbose_level >= 2:
            self.show_graph_2d('Initial', False)
            self.show_graph_2d('Final', True)
        if verbose_level >= 3:
            self.show_error()


class online_slam_2d:
    """
    The fruit of my labor.

    This class is responsible for maintaining the pose estimates of sam and the algae farm buoys.
    Raw detections are sent to an instance of online_slam_2d by an instance of sam_slam_listener.
    The listener class is responsible for interfacing with the buoy and rope detectors.
    Data association is handled within this class.

    This class is also able to publish visualization for RViz.

    === Initialization ===

    === Updates ===
    - add_first_pose():
    - online_update(): This method is called by the sam_slam_listener. This method does not use a buffer and should not
                       be used currently. See below for perfered update method, online_update_queued()
    - online_update_queued(): This method is called by the sam_slam_listener. This method can use either the built-in
                              methods for data association or use manually provided associations. This is controlled by
                              what arguments are passed to the update from the sam_slam_listener. This method uses a
                              buffer so updates are not lost during the graph update.

                              NOTE: The graph is updated needlessly during purely DR updates

    === Data association ===

    === Visualizations ===

    === Helpers ===

    """

    def __init__(self, path_name=None, ropes_by_buoy_ind=None):
        # ===== File path for data logging =====
        self.file_path = path_name
        if self.file_path is None or not isinstance(path_name, str):
            self.file_path = ''
        else:
            self.file_path = path_name + '/'

        # ===== Frames =====
        self.map_frame = rospy.get_param("frame", "map")
        self.robot_frame = rospy.get_param("robot_frame", "")
        self.est_frame = rospy.get_param("est_frame", "estimated/base_link")

        # TF broadcaster
        self.tf_br = tf.TransformBroadcaster()

        # ===== Graph parameters =====
        self.graph = gtsam.NonlinearFactorGraph()
        self.parameters = gtsam.ISAM2Params()
        # self.parameters.setRelinearizeThreshold(0.1)
        # self.parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(self.parameters)

        self.current_x_ind = -1
        self.current_r_ind = -1
        self.x = None  # Poses
        self.b = None  # Buoys
        self.r = None  # Ropes
        self.r_associations = None  # Rope associations, used to update inferred priors of rope detections
        # While the inferred priors are updated the associations are not
        self.l = None  # Lines

        self.buffer = queue.Queue()

        # === dr ===
        self.dr_Pose2s = None
        self.dr_Pose3s = None
        self.dr_pose_raw = None
        self.dr_pose_rpd = None  # roll pitch depth
        self.between_Pose2s = None

        # === gt ===
        self.gt_Pose2s = None
        self.gt_Pose3s = None
        self.gt_pose_raw = None

        # === Estimated ===
        self.post_Pose2s = None
        self.post_Point2s = None
        self.online_Pose2s = None  # Save the current estimate, for comparison to final estimate

        # === Sensors and detections
        self.bearings_ranges = []  # Not used for online
        self.sensor_string_at_key = {}  # camera and sss data is associated with graph nodes here

        # === DA parameters ===
        # TODO improve data association threshold
        self.manual_associations = rospy.get_param("manual_associations", False)
        self.buoy_seq_ids = ast.literal_eval(rospy.get_param("buoy_seq_ids", "[]"))
        self.buoy_line_ids = ast.literal_eval(rospy.get_param("buoy_line_ids", "[]"))
        if len(self.buoy_seq_ids) != len(self.buoy_line_ids) or len(self.buoy_seq_ids) == 0:
            self.manual_associations = False

        # currently only applied to buoys
        self.da_distance_threshold = rospy.get_param('da_distance_threshold', -1.0)
        self.da_m_distance_threshold = rospy.get_param('da_m_distance_threshold', -1.0)

        if self.da_m_distance_threshold > 0:
            self.da_distance_threshold = -1

        # === Rope detection parameters ===
        self.individual_rope_detections = rospy.get_param("individual_rope_detections", True)
        self.use_rope_detections = rospy.get_param("use_rope_detections", True)
        self.use_naive_rope_priors = rospy.get_param("use_naive_rope_priors", False)
        self.rope_batch_size = rospy.get_param("rope_batch_size", 0)

        # = Rope update by count =
        self.rope_batch_current_size = 0
        self.rope_batch_factors = []
        self.rope_batch_priors = []
        self.rope_batch_initial_estimates = []

        # = Rope update method using line DA =
        # This method will override the behaviour set by self.rope_batch_size
        self.rope_batch_by_line = rospy.get_param("rope_batch_by_line", False)
        self.rope_last_line = None
        self.rope_current_line = None
        self.rope_last_time = None
        self.rope_last_line_timeout = rospy.get_param("rope_batch_by_line_timeout", 100)
        self.rope_current_lines = []  # Debugging of rope batching by lines
        # this method will override the rope batching
        self.batch_by_swath = rospy.get_param("batch_by_swath", False)

        # = Swath update method =
        self.swath_seq_ids = ast.literal_eval(rospy.get_param("swath_seq_ids", "[]"))
        # Allows for manual DA of rope detections by swath
        self.swath_line_inds = ast.literal_eval(rospy.get_param("swath_line_ids", "[]"))
        self.batch_by_swath_manual_rope_da = rospy.get_param("batch_by_swath_manual_rope_da", False)
        # Check for valid, matching size, between seq ids and line inds
        if len(self.swath_seq_ids) != len(self.swath_line_inds):
            self.batch_by_swath_manual_rope_da = False
        self.batch_swath_factors = [[] for _ in range(len(self.swath_seq_ids))]
        self.batch_swath_priors = [[] for _ in range(len(self.swath_seq_ids))]
        self.current_swath_ind = -1
        self.current_optimize = False

        # === Prior parameters ===
        # Currently this only applies to the rope priors
        self.update_priors = rospy.get_param("update_priors", False)

        # ===== Sigmas =====
        # Agent prior sigmas
        self.prior_ang_sig = rospy.get_param('prior_ang_sig_deg', 1.0) * np.pi / 180
        self.prior_dist_sig = rospy.get_param('prior_dist_sig', 1.0)

        # buoy prior sigmas
        self.buoy_dist_sig_init = rospy.get_param('buoy_dist_sig', 1.0)

        # rope prior sigmas
        # Used in the naive
        self.rope_dist_sig_init = rospy.get_param('rope_dist_sig', 15.0)
        # Used for less naive rope priors
        self.rope_along_sig_init = rospy.get_param('rope_along_sig', 15.0)
        self.rope_cross_sig_init = rospy.get_param('rope_cross_sig', 2.0)

        # agent odometry sigmas
        self.odo_ang_sig = rospy.get_param('odo_ang_sig_deg', 0.1) * np.pi / 180
        self.odo_dist_sig = rospy.get_param('dist_sig', 0.1)

        # detection sigmas
        # When individual detections are used buoys and ropes should have the same noise model
        # buoy
        self.buoy_detect_ang_sig = rospy.get_param('buoy_detect_ang_sig_deg', 1.0) * np.pi / 180
        self.buoy_detect_dist_sig = rospy.get_param('buoy_detect_dist_sig', .1)

        # ropes
        if self.individual_rope_detections:
            self.detect_ang_sig = self.buoy_detect_ang_sig
            self.detect_dist_sig = self.buoy_detect_dist_sig
        else:
            self.detect_ang_sig = rospy.get_param('detect_ang_sig_deg', 1.0) * np.pi / 180
            self.detect_dist_sig = rospy.get_param('detect_dist_sig', .1)

        # ===== Noise models =====
        self.prior_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.prior_dist_sig,
                                                                      self.prior_dist_sig,
                                                                      self.prior_ang_sig]))

        self.prior_model_lm = gtsam.noiseModel.Diagonal.Sigmas((self.buoy_dist_sig_init,
                                                                self.buoy_dist_sig_init))

        self.prior_model_rope = gtsam.noiseModel.Diagonal.Sigmas((self.rope_dist_sig_init,
                                                                  self.rope_dist_sig_init))

        self.odometry_model = gtsam.noiseModel.Diagonal.Sigmas((self.odo_dist_sig,
                                                                self.odo_dist_sig,
                                                                self.odo_ang_sig))

        self.buoy_detection_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.buoy_detect_dist_sig,
                                                                               self.buoy_detect_ang_sig]))

        # This should be renamed rope_detection_model
        self.detection_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.detect_dist_sig,
                                                                          self.detect_ang_sig]))

        # Noise models of rope detections constructed during rope_setup()

        # ===== buoy prior map =====
        self.n_buoys = None
        self.buoy_priors = None
        self.buoy_average = None

        # ===== rope prior map =====
        self.n_ropes = None
        self.rope_prior_ends = None  # [[[start_x, start_y],[end_x, end_y]], ...]
        self.rope_prior_centers = None
        self.rope_noise_models = None
        self.rope_infos = None  # Per line info, [[center_x, center_y, sig_xx, sig_xy, sig_yx, sig_yy], ...]
        # currently ropes an buoys locations are sent as rviz markers
        # The structure is defined in the sam_listener_online_slam_node.py
        # Format: [[line_1_start_buoy_ind, line_1_end_buoy_ind], ... ,[line_n_start_buoy_ind, line_n_end_buoy_ind]]
        # This can also be defined in the launch file as a parameter with the name 'pipeline_lines'
        self.rope_buoy_ind = ropes_by_buoy_ind

        # ===== Optimizer and values =====
        # self.optimizer = None
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.slam_result = None
        self.current_marginals = None

        # ===== Graph states =====
        self.buoy_map_present = False
        self.rope_map_present = False
        self.initial_pose_set = False
        self.busy = False
        self.busy_queue = False

        # ===== DA and outlier info =====
        self.buoy_detection_info = []

        # ===== Performance info =====
        self.performance_metrics = []

        # ===== Debugging =====
        self.da_check = {}
        self.est_detect_loc = None
        self.true_detect_loc = None

        # ===== Rviz publishers =====
        # Buoy detections and associations
        self.est_detect_loc_pub = rospy.Publisher('/sam_slam/est_detection_positions', Marker, queue_size=10)
        self.da_pub = rospy.Publisher('/sam_slam/da_positions', Marker, queue_size=10)
        self.marker_scale = 1.0
        # Current estimated pose
        self.est_pos_pub = rospy.Publisher('/sam_slam/est_positions', Marker, queue_size=10)
        self.est_marker_x, self.est_marker_y, self.est_marker_z = 3, .5, .5
        # Complete estimated trajectory
        self.est_path_pub = rospy.Publisher('/sam_slam/est_path', Path, queue_size=10)
        # Rope detections
        self.est_rope_pub = rospy.Publisher('/sam_slam/est_rope', MarkerArray, queue_size=10)
        self.rope_marker_scale = 1.0
        self.rope_marker_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.75)
        # Buoy detections
        self.est_buoy_pub = rospy.Publisher('/sam_slam/est_buoys', MarkerArray, queue_size=10)
        self.buoy_marker_scale = 1.25
        self.buoy_marker_color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        self.buoy_valid_color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)  # color used for valid buoy detections
        self.buoy_invalid_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # color used for invalid buoy detections
        # Line landmarks
        # Not actually a line, just a point landmark that is being treated as a line
        self.est_line_pub = rospy.Publisher('/sam_slam/est_line', MarkerArray, queue_size=10)
        self.est_line_pub_verbose = rospy.Publisher('/sam_slam/est_line_verbose',
                                                    PoseWithCovarianceStamped,
                                                    queue_size=1)
        self.line_marker_scale = 1.0
        self.line_marker_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)

        # ===== Verboseness parameters =====
        self.verbose_graph_update = rospy.get_param('verbose_graph_update', False)
        self.verbose_graph_rope_detections = rospy.get_param('verbose_graph_rope_detections', False)
        self.verbose_graph_rope_batching = rospy.get_param('verbose_graph_rope_batching', False)
        self.verbose_graph_buoy_detections = rospy.get_param('verbose_graph_buoy_detections', False)
        self.verbose_graph_rope_associations = rospy.get_param('verbose_graph_rope_associations',
                                                               False)
        self.verbose_graph_buoy_associations = rospy.get_param('verbose_graph_buoy_associations',
                                                               False)
        print(f'swath list {self.swath_seq_ids}')
        print('Graph Class Initialized')

    def buoy_setup(self, buoys):
        '''
        Initializes the buoy landmarks in the factor graph, currently the only way to add landmarks.

        :param buoys: list of buoy coords in the map frame
        :return:
        '''
        print("Buoys being added to online graph")
        if len(buoys) == 0:
            print("Invalid buoy object used!")
            return -1

        self.buoy_priors = np.array(buoys, dtype=np.float64)
        self.n_buoys = len(self.buoy_priors)

        # labels
        self.b = {k: gtsam.symbol('b', k) for k in range(self.n_buoys)}

        # ===== Add buoy priors and initial estimates =====
        for id_buoy in range(self.n_buoys):
            prior = np.array((self.buoy_priors[id_buoy, 0], self.buoy_priors[id_buoy, 1]), dtype=np.float64)

            # Prior
            self.graph.addPriorPoint2(self.b[id_buoy], prior, self.prior_model_lm)

            # Initial estimate
            self.initial_estimate.insert(self.b[id_buoy], prior)

        self.buoy_map_present = True

        # Calculate average buoy position - rough prior for ropes
        self.buoy_average = np.sum(self.buoy_priors, axis=0) / self.n_buoys
        print("Buoy setup: Complete")
        return

    def rope_setup(self, ropes):
        """
        Calculate relevant stuff for adding rope priors to the detections

        Assumptions:
        - nominally horizontal rows, more checks need to handle vertical. This will affect the cross and along variances
        - along variance is not currently scaled to the length of the rope
            This is done in testing_covar.py
        :param ropes:
        :return:
        """
        # check that buoys have been setup
        # if self.b is None:
        #     return

        print("Rope setup starting ")

        if self.rope_map_present:
            print(f"Rope(s) already set")
            return

        # In general, we want to work with the ropes wrt to the b indices that make up the ropes
        # TODO bring into agreement with the update
        # Ropes are added via two methods
        # 1) markers --> this was the original but its kind of weird and doesnt work with the pipeline scenario??
        # 2) A list of buoy indices, [[start_ind, end_ind], ...]
        if ropes is None:
            n_ropes_spatial = 0
        else:
            n_ropes_spatial = len(ropes)

        n_ropes_inds = len(self.rope_buoy_ind)

        # Change to using supplied indices by default
        if n_ropes_inds > 0 and self.buoy_map_present is False:
            print("Rope Setup is waiting for buoys to be defined")
            return -1

        if n_ropes_inds > 0:
            self.n_ropes = n_ropes_inds
        else:
            self.n_ropes = n_ropes_spatial

        # if n_ropes_inds != n_ropes_spatial:
        #     raise Exception("Rope layout mismatch: check node node script and rope marker publisher script")

        # Initialize the rope means and covariances
        self.rope_prior_ends = []  # List of rope end points, each rope in the form: [start_x, start_y],[end_x, ]]
        self.rope_prior_centers = {}
        self.rope_noise_models = {}
        self.rope_infos = []  # Per line info for debugging, [[center_x, center_y, sig_xx, sig_xy, sig_yx, sig_yy], ...]

        for rope_ind in range(self.n_ropes):
            if n_ropes_inds:
                start_buoy = self.rope_buoy_ind[rope_ind][0]
                end_buoy = self.rope_buoy_ind[rope_ind][1]
                start_x, start_y = self.buoy_priors[start_buoy, 0], self.buoy_priors[start_buoy, 1]
                end_x, end_y = self.buoy_priors[end_buoy, 0], self.buoy_priors[end_buoy, 1]
            else:
                rope = ropes[rope_ind]
                start_x, start_y = rope[0][0], rope[0][1]
                end_x, end_y = rope[1][0], rope[1][1]

            print(f"Rope {rope_ind}: ({start_x},{start_y}) ({end_x},{end_y})")

            # Record the rope prior end points
            self.rope_prior_ends.append([[start_x, start_y], [end_x, end_y]])

            # Determines and record rope prior mean
            centers = online_slam_2d.calculate_center(start_x, start_y, end_x, end_y)
            self.rope_prior_centers[rope_ind] = centers

            # === Determine covariances of rope priors ===
            # These covariances are define with an along rope and cross rope components.
            # Two methods are used to determine the along rope component.
            # 1) Scaled: negative values indicate the along rope variance will be scaled
            # 2) Fixed: The specified value is used, no scaling is applied
            # Scaled covariance
            # negative values indicate that the value, N, is to be scaled with the length of the line, L.
            # Such that sigma = L/(2 * N).Default is 3.
            if self.rope_along_sig_init <= 0:
                if self.rope_along_sig_init == 0:
                    n_sigmas = 3  # n_sigmas does not have to be a int
                else:
                    n_sigmas = abs(self.rope_along_sig_init)

                rope_length = self.calculate_distance(start_x, start_y, end_x, end_y)
                scaled_rope_along_sig_init = rope_length / (2 * n_sigmas)

                cov_matrix = np.array([[scaled_rope_along_sig_init ** 2, 0.0],
                                       [0.0, self.rope_cross_sig_init ** 2]])

            # This is the 'basic' covariance, it needs to be rotated to align with the rope
            else:
                cov_matrix = np.array([[self.rope_along_sig_init ** 2, 0.0],
                                       [0.0, self.rope_cross_sig_init ** 2]])

            # Rotate the cov_matrix to align with the rope
            rope_angle = online_slam_2d.calculate_angle(start_x, start_y, end_x, end_y)
            rotation_matrix = np.array([[np.cos(rope_angle), -np.sin(rope_angle)],
                                        [np.sin(rope_angle), np.cos(rope_angle)]])
            rot_cov_matrix = rotation_matrix @ cov_matrix @ rotation_matrix.transpose()

            # Record noise model, used for data and prior associations
            self.rope_noise_models[rope_ind] = gtsam.noiseModel.Gaussian.Covariance(rot_cov_matrix)

            # Record rope info for debugging and post processing checks
            rope_info = [centers[0], centers[1],
                         rot_cov_matrix[0, 0], rot_cov_matrix[0, 1],
                         rot_cov_matrix[1, 0], rot_cov_matrix[1, 1]]
            print(f"Rope info:{rope_info}")
            self.rope_infos.append(rope_info)

        # labels
        self.l = {k: gtsam.symbol('l', k) for k in range(self.n_ropes)}

        # ===== Add buoy priors and initial estimates =====
        # I'm sorry for the naming convention
        # self.r are rope detections
        # self.l are the 'line' objects
        for id_rope in range(self.n_ropes):
            # Prior
            prior = np.array((self.rope_prior_centers[id_rope][0], self.rope_prior_centers[id_rope][1]),
                             dtype=np.float64)

            # Add prior factor
            self.graph.addPriorPoint2(self.l[id_rope], prior, self.rope_noise_models[rope_ind])

            # Initial estimate
            self.initial_estimate.insert(self.l[id_rope], prior)

        # Indicate that the rope setup is complete
        self.rope_map_present = True

        print("Rope setup: Complete")

        np.savetxt(self.file_path + 'rope_info.csv',
                   np.array(self.rope_infos), delimiter=',', fmt='%.3f')
        print(f"Rope info saved: {self.file_path + 'rope_info.csv'}")
        return 1

    def rope_update(self, debug=False):
        """
        Calculate relevant stuff for updating inferred rope priors to the detections

        Assumptions:
        - nominally horizontal rows, more checks need to handle vertical. This will effect the cross and along variances
        - along variance is not currently scaled to the length of the rope
            This is done in testing_covar.py

        :return:
        """
        # TODO rope_setup() and rope_update() should both use the index of buoys to define ropes

        if self.rope_prior_centers is None or self.rope_noise_models is None:
            return

        self.rope_prior_ends = []
        self.rope_prior_centers = {}
        self.rope_noise_models = {}

        for rope_ind, rope in enumerate(self.rope_buoy_ind):
            start_buoy_ind = rope[0]
            end_buoy_ind = rope[1]

            start_x, start_y = self.current_estimate.atPoint2(self.b[start_buoy_ind])
            end_x, end_y = self.current_estimate.atPoint2(self.b[end_buoy_ind])

            self.rope_prior_ends.append([[start_x, start_y], [end_x, end_y]])

            if debug:
                original = self.rope_prior_ends[rope_ind]
                print(f"old rope {rope_ind}: "
                      f"({original[0][0]:.2f}, {original[0][1]:.2f}) ({original[1][0]:.2f}, {original[1][1]:.2f})")
                print(f"New rope {rope_ind}: ({start_x:.2f},{start_y:.2f}) ({end_x:.2f},{end_y:.2f})")

            self.rope_prior_centers[rope_ind] = online_slam_2d.calculate_center(start_x, start_y, end_x, end_y)

            # This is the 'basic' covariance, it needs to be rotated to align with the rope
            cov_matrix = np.array([[self.rope_along_sig_init ** 2, 0.0],
                                   [0.0, self.rope_cross_sig_init ** 2]])

            # Rotate the cov_matrix to align with the rope
            rope_angle = online_slam_2d.calculate_angle(start_x, start_y, end_x, end_y)
            rotation_matrix = np.array([[np.cos(rope_angle), -np.sin(rope_angle)],
                                        [np.sin(rope_angle), np.cos(rope_angle)]])
            rot_cov_matrix = rotation_matrix @ cov_matrix @ rotation_matrix.transpose()

            self.rope_noise_models[rope_ind] = gtsam.noiseModel.Gaussian.Covariance(rot_cov_matrix)

    def add_first_pose(self, dr_pose, gt_pose, id_string=None):
        """
        Pose format [x, y, z, q_w, q_x, q_y, q_z]
        """
        # Wait to start building graph until the prior is received
        # if not self.buoy_map_present:
        #     print("Waiting for buoy prior map")
        #     return -1

        if self.current_x_ind != -1:
            print("add_first_pose() called with a graph that already has a pose added")
            return -1

        # === Record relevant poses ===
        """
        Both dr and gt are saved as Pose2, Pose3 and the raw list sent to the slam processing
        dr_pose format: [x, y, z, q_w, q_x, q_y, q_z, roll, pitch, depth]
        gt_pose format: [x, y, z, q_w, q_x, q_y, q_z, time]
        """
        # dr
        self.dr_Pose2s = [correct_dr(create_Pose2(dr_pose[:7]))]
        # TODO Pose3 also need to be corrected in the same way Pose2 is corrected
        self.dr_Pose3s = [create_Pose3(dr_pose)]
        self.dr_pose_raw = [dr_pose]
        if len(dr_pose) == 10:
            self.dr_pose_rpd = [dr_pose[7:10]]
        self.between_Pose2s = []

        # gt
        self.gt_Pose2s = [create_Pose2(gt_pose)]
        self.gt_Pose3s = [create_Pose3(gt_pose)]
        self.gt_pose_raw = [gt_pose]

        # Add label
        self.current_x_ind += 1
        self.x = {self.current_x_ind: gtsam.symbol('x', self.current_x_ind)}
        self.r = {}
        self.r_associations = {}

        # Add type or sensor identifier
        # This used to associate nodes of the graph with sensor reading, processed offline
        if id_string is None:
            self.sensor_string_at_key[self.current_x_ind] = 'odometry'
        else:
            self.sensor_string_at_key[self.current_x_ind] = id_string

        # Add prior factor
        self.graph.add(gtsam.PriorFactorPose2(self.x[0], self.dr_Pose2s[0], self.prior_model))

        # ===== Add initial estimate =====
        self.initial_estimate.insert(self.x[0], self.dr_Pose2s[0])
        self.current_estimate = self.initial_estimate

        # === Save initial pose2 ===
        # Online estimate is saved for later analysis
        self.online_Pose2s = [self.current_estimate.atPose2(self.x[self.current_x_ind])]

        self.initial_pose_set = True
        print("Done with first pose - x0")
        if self.x is None:
            print("problem")
        return

    # def online_update(self, dr_pose, gt_pose, relative_detection=None, id_string=None, da_id=None):
    #     """
    #     OLD, DO NOT USE!!!
    #
    #     Pose format [x, y, z, q_w, q_x, q_y, q_z]
    #     """
    #     if not self.initial_pose_set:
    #         print("Attempting to update before initial pose")
    #         self.add_first_pose(dr_pose=dr_pose, gt_pose=gt_pose, id_string=id_string)
    #         return
    #
    #     # Attempt at preventing saturation
    #     self.busy = True
    #
    #     # === Record relevant poses ===
    #     # dr
    #     self.dr_Pose2s.append(correct_dr(create_Pose2(dr_pose[:7])))
    #     self.dr_Pose3s.append(create_Pose3(dr_pose))
    #     self.dr_pose_raw.append(dr_pose)
    #     if self.dr_pose_rpd is not None and len(dr_pose) == 10:
    #         self.dr_pose_rpd.append(dr_pose[7:10])
    #
    #     # gt
    #     self.gt_Pose2s.append(create_Pose2(gt_pose))
    #     self.gt_Pose3s.append(create_Pose3(gt_pose))
    #     self.gt_pose_raw.append(gt_pose)
    #
    #     # Find the relative odometry between dr_poses
    #     between_odometry = self.dr_Pose2s[-2].between(self.dr_Pose2s[-1])
    #     self.between_Pose2s.append(between_odometry)
    #
    #     # Add label
    #     self.current_x_ind += 1
    #     self.x[self.current_x_ind] = gtsam.symbol('x', self.current_x_ind)
    #
    #     # Add type or sensor identifier
    #     # This used to associate nodes of the graph with sensor reading, processed offline
    #     if relative_detection is None and id_string is None:
    #         self.sensor_string_at_key[self.current_x_ind] = 'odometry'
    #     elif relative_detection is not None and id_string is None:
    #         self.sensor_string_at_key[self.current_x_ind] = 'detection'
    #     else:
    #         self.sensor_string_at_key[self.current_x_ind] = id_string
    #
    #     # ===== Add the between factor =====
    #     self.graph.add(gtsam.BetweenFactorPose2(self.x[self.current_x_ind - 1],
    #                                             self.x[self.current_x_ind],
    #                                             between_odometry,
    #                                             self.odometry_model))
    #
    #     # Compute initialization value from the current estimate and odometry
    #     computed_est = self.current_estimate.atPose2(self.x[self.current_x_ind - 1]).compose(between_odometry)
    #
    #     # Update initial estimate
    #     self.initial_estimate.insert(self.x[self.current_x_ind], computed_est)
    #
    #     # ===== Process detection =====
    #     # === Buoy ===
    #     # TODO this might need to be more robust, not assume detections will lead to graph update
    #     if relative_detection is not None and da_id != -ObjectID.ROPE.value:
    #         # Calculate the map location of the detection given relative measurements and current estimate
    #         self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
    #         detect_bearing = computed_est.bearing(self.est_detect_loc)
    #         detect_range = computed_est.range(self.est_detect_loc)
    #
    #         if self.manual_associations and da_id != -ObjectID.BUOY.value:
    #             buoy_association_id = da_id
    #         else:
    #             buoy_association_id, buoy_association_dist = self.associate_detection(self.est_detect_loc)
    #
    #             # ===== DA debugging =====
    #             # Apply relative detection to gt to find the true DA
    #             self.true_detect_loc = self.gt_Pose2s[-1].transformFrom(np.array(relative_detection, dtype=np.float64))
    #             true_association_id, true_association_dist = self.associate_detection(self.true_detect_loc)
    #
    #             if buoy_association_id == true_association_id:
    #                 self.da_check[self.current_x_ind] = [True,
    #                                                      buoy_association_id, true_association_id,
    #                                                      buoy_association_dist, true_association_dist]
    #             else:
    #                 self.da_check[self.current_x_ind] = [False,
    #                                                      buoy_association_id, true_association_id,
    #                                                      buoy_association_dist, true_association_dist]
    #
    #         if self.verbose_graph_buoy_detections and relative_detection is not None:
    #             print(
    #                 f'Buoy detection - range: {detect_range:.2f}  bearing: {detect_bearing.theta():.2f}  DA: {buoy_association_id}')
    #
    #         # ===== Add detection to graph =====
    #         self.graph.add(gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
    #                                                   self.b[buoy_association_id],
    #                                                   detect_bearing,
    #                                                   detect_range,
    #                                                   self.buoy_detection_model))
    #
    #     # === Rope ===
    #     if relative_detection is not None and da_id == -ObjectID.ROPE.value:
    #         # Calculate the map location of the detection given relative measurements and current estimate
    #         self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
    #         detect_bearing = computed_est.bearing(self.est_detect_loc)
    #         detect_range = computed_est.range(self.est_detect_loc)
    #
    #         if self.verbose_graph_rope_detections:
    #             print(f'Rope detection - range: {detect_range:.2f}  bearing: {detect_bearing.theta():.2f}')
    #
    #         # ===== Add detection to graph =====
    #         # Add the rope landmark
    #         self.current_r_ind += 1
    #         self.r[self.current_r_ind] = gtsam.symbol('r', self.current_r_ind)
    #
    #         # Add new landmarks prior and noise
    #         rope_prior = np.array((self.buoy_average[0], self.buoy_average[1]), dtype=np.float64)
    #         self.graph.addPriorPoint2(self.r[self.current_r_ind], rope_prior, self.prior_model_rope)
    #
    #         # Initial estimate
    #         if self.rope_batch_size >= 0:
    #             self.initial_estimate.insert(self.r[self.current_r_ind], self.est_detect_loc)
    #         else:
    #             self.rope_batch_initial_estimates.append([self.r[self.current_r_ind], self.est_detect_loc])
    #
    #         # Add factor between current x and current r
    #
    #         self.graph.add(gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
    #                                                   self.r[self.current_r_ind],
    #                                                   detect_bearing,
    #                                                   detect_range,
    #                                                   self.detection_model))
    #
    #     # Time update process
    #     start_time = rospy.Time.now()
    #
    #     # Incremental update
    #     self.isam.update(self.graph, self.initial_estimate)
    #     self.current_estimate = self.isam.calculateEstimate()
    #
    #     # self.graph.resize(0)
    #     self.initial_estimate.clear()
    #
    #     # === Save online estimated pose2 ===
    #     # Online estimate is saved for later analysis
    #     self.online_Pose2s.append(self.current_estimate.atPose2(self.x[self.current_x_ind]))
    #
    #     end_time = rospy.Time.now()
    #     update_time = (end_time - start_time).to_sec()
    #
    #     # Release the graph
    #     self.busy = False
    #
    #     # ===== debugging and visualizations =====
    #     # Log to terminal
    #     if self.verbose_graph_update:
    #         print(f"Done with update - x{self.current_x_ind}: {update_time} s")
    #
    #     # === Debug publishing ===
    #     self.publish_estimated_pos_marker_and_transform(computed_est)
    #     # TODO publish path
    #     # self.publish_est_path()
    #     # Buoy detection
    #     if relative_detection is not None and da_id != -ObjectID.ROPE.value:
    #         self.publish_detection_markers(buoy_association_id)
    #
    #     if self.x is None:
    #         print("problem")
    #
    #     return

    def online_update_queued(self, dr_pose, gt_pose, relative_detection=None, id_string=None, da_id=None, seq_id=None):
        """

        :param dr_pose: Pose format [x, y, z, q_w, q_x, q_y, q_z]
        :param gt_pose: Pose format [x, y, z, q_w, q_x, q_y, q_z]
        :param relative_detection:
        :param id_string: string used to identify if new node corresponds
        :param da_id:
        :param seq_id:
        :return:
        """

        if not self.initial_pose_set:
            print("Attempting to update before initial pose")
            self.add_first_pose(dr_pose=dr_pose, gt_pose=gt_pose, id_string=id_string)
            return

        if seq_id is not None:
            seq_id = int(seq_id)

        # Queue
        self.buffer.put((dr_pose, gt_pose, relative_detection, id_string, da_id, seq_id))

        if not self.busy_queue:
            self.busy_queue = True
            while not self.buffer.empty():
                try:
                    buffered_data = self.buffer.get(block=False)
                except queue.Empty:
                    print("Buffer is empty.")
                    break

                dr_pose, gt_pose, relative_detection, id_string, da_id, seq_id = buffered_data

                # === Record relevant poses ===
                # dr
                self.dr_Pose2s.append(correct_dr(create_Pose2(dr_pose[:7])))
                self.dr_Pose3s.append(create_Pose3(dr_pose))
                self.dr_pose_raw.append(dr_pose)
                if self.dr_pose_rpd is not None and len(dr_pose) == 10:
                    self.dr_pose_rpd.append(dr_pose[7:10])

                # gt
                self.gt_Pose2s.append(create_Pose2(gt_pose))
                self.gt_Pose3s.append(create_Pose3(gt_pose))
                self.gt_pose_raw.append(gt_pose)

                # Find the relative odometry between dr_poses
                between_odometry = self.dr_Pose2s[-2].between(self.dr_Pose2s[-1])
                self.between_Pose2s.append(between_odometry)

                # Add label
                self.current_x_ind += 1
                self.x[self.current_x_ind] = gtsam.symbol('x', self.current_x_ind)

                # Add type or sensor identifier
                # This used to associate nodes of the graph with sensor reading, processed offline
                if relative_detection is None and id_string is None:
                    self.sensor_string_at_key[self.current_x_ind] = 'odometry'
                elif relative_detection is not None and id_string is None:
                    self.sensor_string_at_key[self.current_x_ind] = 'detection'
                else:
                    self.sensor_string_at_key[self.current_x_ind] = id_string

                # === Marginals ===
                if relative_detection is not None:
                    self.current_marginals = gtsam.Marginals(self.graph, self.current_estimate)
                else:
                    self.current_marginals = None

                # record detection type for performance metrics
                relative_detection_state = 0
                factor_counter = 0  # keep track of added factors
                prior_counter = 0

                # ===== Add the between factor =====
                self.graph.add(gtsam.BetweenFactorPose2(self.x[self.current_x_ind - 1],
                                                        self.x[self.current_x_ind],
                                                        between_odometry,
                                                        self.odometry_model))
                factor_counter += 1

                # Compute initialization value from the current estimate and odometry
                computed_est = self.current_estimate.atPose2(self.x[self.current_x_ind - 1]).compose(between_odometry)

                # Update initial estimate
                self.initial_estimate.insert(self.x[self.current_x_ind], computed_est)

                # ===== Process detection =====
                # flags for updating the factor graph
                valid_buoy_da = False
                valid_rope = False

                # === Find the most current covariance ===
                # Find the most recent valid key
                current_key = None
                for key_ind in range(self.current_x_ind - 1, -1, -1):
                    check_key = self.x[key_ind]
                    if check_key in self.current_estimate.keys():
                        current_key = check_key
                        break

                # Find covariance of the current key
                if self.current_marginals is not None and current_key is not None:
                    current_covariance = self.current_marginals.marginalCovariance(current_key)
                else:
                    current_covariance = None

                # === Buoy ===
                if relative_detection is not None and da_id != -ObjectID.ROPE.value and da_id != -ObjectID.PIPE.value:
                    # Detection state info
                    relative_detection_state = 1  # perfomance metrics

                    # data association flag
                    valid_buoy_da = True

                    # Swath update info
                    # Determine if current update is a member of a designated swath, and if optimization is warranted
                    if seq_id is not None:
                        self.current_swath_ind, self.current_optimize = self.check_seq_id(current_seq_id=seq_id)

                    # Calculate the map location of the detection given relative measurements and current estimate
                    self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
                    detect_bearing = computed_est.bearing(self.est_detect_loc)
                    detect_range = computed_est.range(self.est_detect_loc)

                    # if current_covariance is not None:
                    #     self.associate_detection_likelihood(self.est_detect_loc, current_covariance, marginals)

                    # TODO Old method of manual buoy DA, why did I do it this way
                    # if self.manual_associations and da_id != -ObjectID.BUOY.value:
                    #     buoy_association_id = da_id

                    if self.manual_associations and seq_id is not None:
                        try:
                            index = self.buoy_seq_ids.index(seq_id)
                            buoy_association_id = self.buoy_line_ids[index]
                            valid_buoy_da = buoy_association_id != -1
                        except ValueError:
                            buoy_association_id = -1
                            valid_buoy_da = False

                        # Record detection for debugging
                        self.buoy_detection_info.append([buoy_association_id,
                                                         0,  # buoy_association_dist
                                                         0])  # buoy_association_m_dist

                    else:
                        # If no covariance estimate is available DA uses only euclidean distance
                        if current_covariance is None:
                            buoy_association_id, buoy_association_dist = self.associate_detection(self.est_detect_loc)
                            # dummy m distance
                            buoy_association_m_dist = 0

                        # If covariance is present M distance is also available
                        else:
                            buoy_association_id, buoy_association_dist, buoy_association_m_dist = (
                                self.associate_detection_likelihood(self.est_detect_loc, current_covariance))

                        # Debug
                        print(f'Buoy DA Distance: {buoy_association_dist} ({self.da_distance_threshold})')

                        # outlier thresholding
                        if self.da_distance_threshold > 0 and buoy_association_dist > self.da_distance_threshold:
                            if self.verbose_graph_buoy_detections:
                                print(f'DA euclidean distance threshold exceeded, buoy detection ignored. (euclidean)')
                            valid_buoy_da = False

                        if self.da_m_distance_threshold > 0 and buoy_association_m_dist > self.da_m_distance_threshold:
                            if self.verbose_graph_buoy_detections:
                                print(f'DA mahalanobis distance threshold exceeded, buoy detection ignored. ')
                            valid_buoy_da = False

                        if valid_buoy_da:
                            recorded_association = buoy_association_id
                        else:
                            recorded_association = -1

                        # Record detection for debugging
                        self.buoy_detection_info.append([recorded_association,
                                                         buoy_association_dist,
                                                         buoy_association_m_dist])

                        # ===== DA debugging =====
                        # Apply relative detection to gt to find the true DA
                        self.true_detect_loc = self.gt_Pose2s[-1].transformFrom(
                            np.array(relative_detection, dtype=np.float64))
                        true_association_id, true_association_dist = self.associate_detection(self.true_detect_loc)

                        if buoy_association_id == true_association_id:
                            self.da_check[self.current_x_ind] = [True,
                                                                 buoy_association_id, true_association_id,
                                                                 buoy_association_dist, true_association_dist]
                        else:
                            self.da_check[self.current_x_ind] = [False,
                                                                 buoy_association_id, true_association_id,
                                                                 buoy_association_dist, true_association_dist]

                    if self.verbose_graph_buoy_detections and relative_detection is not None:
                        print(
                            f'Buoy detection - range: {detect_range:.2f}  bearing: {detect_bearing.theta():.2f}' +
                            f'  Seq id: {seq_id}  DA: {buoy_association_id}')

                    # ===== Add detection to graph =====
                    if valid_buoy_da:
                        x_b_range_bearing_factor = gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
                                                                              self.b[buoy_association_id],
                                                                              detect_bearing,
                                                                              detect_range,
                                                                              self.buoy_detection_model)

                        if self.batch_by_swath:
                            if self.current_swath_ind > -1:
                                self.batch_swath_factors[self.current_swath_ind].append(x_b_range_bearing_factor)
                        else:
                            self.graph.add(x_b_range_bearing_factor)
                            factor_counter += 1

                # === Rope ===
                if relative_detection is not None and (da_id == -ObjectID.ROPE.value or da_id == -ObjectID.PIPE.value):
                    # Detection state info
                    relative_detection_state = 2  # performance metrics
                    valid_rope = True

                    # Batch update info
                    if self.rope_batch_size > 0:
                        self.rope_batch_current_size += 1

                    # Swath update info
                    # Determine if current update is a member of a designated swath, and if optimization is warranted
                    if seq_id is not None:
                        self.current_swath_ind, self.current_optimize = self.check_seq_id(current_seq_id=seq_id)

                    # Calculate the map location of the detection given relative measurements and current estimate
                    self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
                    detect_bearing = computed_est.bearing(self.est_detect_loc)
                    detect_range = computed_est.range(self.est_detect_loc)

                    if self.verbose_graph_rope_detections:
                        # Verbose rope
                        if da_id == -ObjectID.ROPE.value:
                            print(f'Rope detection - range: {detect_range:.2f}  bearing: {detect_bearing.theta():.2f}' +
                                  f' Seq id: {seq_id}')
                        # Verbose pipe
                        if da_id == -ObjectID.PIPE.value:
                            print(f'Pipe detection - range: {detect_range:.2f}  bearing: {detect_bearing.theta():.2f}' +
                                  f' Seq id: {seq_id}')

                    # ===== Add detection to graph =====
                    # Add the rope landmark
                    self.current_r_ind += 1
                    self.r[self.current_r_ind] = gtsam.symbol('r', self.current_r_ind)

                    # Initial estimate
                    self.initial_estimate.insert(self.r[self.current_r_ind], self.est_detect_loc)

                    # TODO figure out when initial initial values should be inserted into the graph
                    # if self.rope_batch_size >= 0:
                    #     self.rope_batch_initial_estimates.append([self.r[self.current_r_ind], self.est_detect_loc])
                    # else:
                    #     self.initial_estimate.insert(self.r[self.current_r_ind], self.est_detect_loc)

                    # if current_covariance is not None:
                    #     self.associate_rope_detection_likelihood(self.est_detect_loc, current_covariance)

                    # Add new landmarks prior and noise
                    # DA can be handled by Euclidean distance or max-likelihood
                    # Priors are only added if individual_rope_detections is set to True

                    # Sets rope prior bases on average of buoy positions
                    # This will allow for plotting/mapping of rope detections but doesn't make them very useful for
                    # localization.
                    if self.rope_prior_ends is None:
                        if self.verbose_graph_rope_associations:
                            print("Data Association: Naive")
                        avg_rope_position = np.array((self.buoy_average[0], self.buoy_average[1]), dtype=np.float64)
                        if self.individual_rope_detections:
                            # Swath Updating
                            if self.batch_by_swath:
                                if self.current_swath_ind > -1:
                                    self.batch_swath_priors[self.current_swath_ind].append([self.r[self.current_r_ind],
                                                                                            avg_rope_position,
                                                                                            self.prior_model_rope])

                            # Batch updating of rope detections
                            elif self.rope_batch_size >= 0:
                                self.rope_batch_priors.append([self.r[self.current_r_ind],
                                                               avg_rope_position,
                                                               self.prior_model_rope])

                            # Individual updating of rope detections
                            else:
                                self.graph.addPriorPoint2(self.r[self.current_r_ind],
                                                          avg_rope_position,
                                                          self.prior_model_rope)
                                prior_counter += 1

                            self.rope_current_line = -1

                    # === Data Association of Rope Detection
                    # This method of assigning Priors first attempts to associate each rope detection with a particular
                    # rope. This association can be based on Euclidean distance or max likelihood. The likelihood method
                    # uses the marginals calculated by GTSAM, this might require us to optimize at each time step
                    # Currently there is no outlier detection/rejection for ropes
                    # Also the option for manual associations
                    else:
                        # Manual associations based on seq
                        if self.batch_by_swath_manual_rope_da and self.current_swath_ind > -1:
                            if self.verbose_graph_rope_associations:
                                print("Data Association: Manual")
                            rope_association_ind = self.swath_line_inds[self.current_swath_ind]

                        # Data association based on Euclidean distance
                        elif current_covariance is None or True:
                            if self.verbose_graph_rope_associations:
                                print("Data Association: Euclidean")
                            rope_association_ind = self.associate_rope_detection(self.est_detect_loc)

                        # Data association based on max likelihood, requires marginal computation by GTSAM
                        else:
                            if self.verbose_graph_rope_associations:
                                print("Data Association: Likelihood")
                            rope_association_ind, _, _ = self.associate_rope_detection_likelihood(self.est_detect_loc,
                                                                                                  current_covariance)
                        self.r_associations[self.current_r_ind] = rope_association_ind
                        self.rope_current_line = rope_association_ind  # this is used for the line batching

                        # Here individual_rope_detection refers to the assignment of landmarks to detections.
                        # If true, detection is assigned a unique landmark.
                        if self.individual_rope_detections:
                            if self.use_rope_detections:
                                # Handling the prior associated with the current rope detection
                                # Swath Updating
                                if self.batch_by_swath:
                                    if self.current_swath_ind > -1:
                                        self.batch_swath_priors[self.current_swath_ind].append(
                                            [self.r[self.current_r_ind],
                                             self.rope_prior_centers[
                                                 rope_association_ind],
                                             self.rope_noise_models[
                                                 rope_association_ind]])
                                        print("Adding to swatch prior")
                                # Batch updating of rope detections
                                elif self.rope_batch_size >= 0:
                                    self.rope_batch_priors.append([self.r[self.current_r_ind],
                                                                   self.rope_prior_centers[rope_association_ind],
                                                                   self.rope_noise_models[rope_association_ind]])
                                    print("Adding batch prior")
                                # Not adding prior, detection not used for localization
                                elif self.use_naive_rope_priors:
                                    pass
                                    print("not adding prior!!")
                                # Individual updating of rope detection
                                else:
                                    self.graph.addPriorPoint2(self.r[self.current_r_ind],
                                                              self.rope_prior_centers[rope_association_ind],
                                                              self.rope_noise_models[rope_association_ind])
                                    prior_counter += 1
                                    print("Adding 'normal' prior!!")

                        # The Nacho method is used when the individual_rope_detections is set to False
                        else:
                            # Define the new factor between current x and detected l < single point landmark for rope
                            x_l_range_bearing_factor = gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
                                                                                  self.l[rope_association_ind],
                                                                                  detect_bearing,
                                                                                  detect_range,
                                                                                  self.detection_model)
                            # Handling the factor associated with the current rope detection -> 'line' landmark
                            # Swath Updating
                            if self.batch_by_swath:
                                if self.current_swath_ind > -1:
                                    self.batch_swath_factors[self.current_swath_ind].append(x_l_range_bearing_factor)
                                    print(f"Appending swath factor: {self.l[rope_association_ind]}")


                            elif self.rope_batch_size >= 0:
                                self.rope_batch_factors.append(x_l_range_bearing_factor)
                                print(f"appending batch factor: {self.l[rope_association_ind]}")

                            # Individual updating of rope detections
                            else:
                                self.graph.add(x_l_range_bearing_factor)
                                factor_counter += 1
                                print(f"Adding 'normal' factor: {self.l[rope_association_ind]}")

                    # === Add factor between current x and current r ===
                    if self.use_rope_detections:
                        # Define the new factor between current x and detected r < single point landmark for detection
                        x_r_range_bearing_factor = gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
                                                                              self.r[self.current_r_ind],
                                                                              detect_bearing,
                                                                              detect_range,

                                                                              self.detection_model)
                        if self.batch_by_swath:
                            if self.current_swath_ind > -1:
                                self.batch_swath_factors[self.current_swath_ind].append(x_r_range_bearing_factor)

                        if self.rope_batch_size >= 0:
                            # TODO I disabled adding r
                            # self.rope_batch_factors.append(x_r_range_bearing_factor)
                            pass
                        else:
                            # TODO TESTING: disabled adding x-r constraints for the 'normal' method
                            # self.graph.add(x_r_range_bearing_factor)
                            # factor_counter += 1
                            pass

                # Rope detections are added Using three methods
                # (1) Individually -> self.rope_batch_size <= 0
                # (s) By defined swath
                # (2) Batches of specified number -> self.rope_batch_size > 0
                # (3) By lines (Newest method) -> self.rope_batch_by_line == True
                # The line method (3) overrides the other two methods.
                # This method was added in the rope_batching_by_line branch

                # Swath method
                if self.batch_by_swath:
                    if self.current_optimize:
                        if self.verbose_graph_rope_batching:
                            print(f'Starting swath Batch Update - {self.current_swath_ind}')
                            print(f"Initial estimates: {len(self.rope_batch_initial_estimates)}")
                            print(f"Priors: {len(self.batch_swath_priors)}")
                            print(f"Factors: {len(self.batch_swath_factors)}")

                        # # Initial value
                        # for rope_initial in self.rope_batch_initial_estimates:
                        #     self.initial_estimate.insert(*rope_initial)
                        #     if self.verbose_graph_rope_batching:
                        #         print(rope_initial)
                        # self.rope_batch_initial_estimates = []

                        # Priors
                        # if self.verbose_graph_rope_batching:
                        #     print("Priors:")
                        for rope_prior in self.batch_swath_priors[self.current_swath_ind]:
                            self.graph.addPriorPoint2(*rope_prior)
                            prior_counter += 1
                            print(" adding swath priors")
                            # TODO check this out!!!
                            print("skipping some priors")
                            break

                            # if self.verbose_graph_rope_batching:
                            #     print(rope_prior)
                        # self.batch_swath_priors[self.current_swath_ind] = []

                        # Factors
                        # if self.verbose_graph_rope_batching:
                        #     print("Factors:")
                        added_factor_count = 0
                        for rope_factor in self.batch_swath_factors[self.current_swath_ind]:
                            self.graph.add(rope_factor)
                            factor_counter += 1
                            added_factor_count += 1
                            print("adding swath factors")
                        print(f"Added {added_factor_count} factors")

                            # if self.verbose_graph_rope_batching:
                            #     print(rope_factor)

                        # TODO figure out why this was commented out, feels like emptying would be required
                        self.batch_swath_factors[self.current_swath_ind] = []

                        # Reset optimization status
                        self.current_optimize = False

                # Method 3 - check if batching by line conditions have been met
                elif self.rope_batch_by_line:
                    if self.rope_current_line is not None:
                        # 2 conditions trigger adding the rope detection factors
                        # (1) Detection of a different line
                        # (2) last time of a set of detections associated with a particular line exceeds timeout

                        # new line detected
                        if self.rope_last_time is None:
                            different_rope_condition = False
                        elif self.rope_current_line == -1:
                            different_rope_condition = True  # -1 used to indicate no priors of ropes
                        else:
                            different_rope_condition = self.rope_current_line != self.rope_last_line

                        # timeout condition
                        if self.rope_last_time is None:
                            timeout_condition = False
                        else:
                            timeout_condition = (rospy.Time.now() - self.rope_last_time).to_sec() >= self.rope_last_line_timeout

                        # Update and reset
                        self.rope_current_lines.append(self.rope_current_line)
                        self.rope_last_line = self.rope_current_line
                        self.rope_last_time = rospy.Time.now()
                        self.rope_current_line = None

                        # Note: in the case of different_rope_condition we should exclude the last element from the
                        # update and saved for later

                        # Method 3 - different rope condition
                        if different_rope_condition:
                            if self.verbose_graph_rope_batching:
                                print('Starting Rope Batch Update - Method 3 - New rope')
                                print(f"Line indices: {self.rope_current_lines[:-1]}")
                                print(f"Initial estimates: {len(self.rope_batch_initial_estimates)}")
                                print(f"Priors: {len(self.rope_batch_priors)}")
                                print(f"Factors: {len(self.rope_batch_factors)}")

                            # clear line DA indices
                            self.rope_current_lines = self.rope_current_lines[-1:]

                            # Initial values
                            for rope_initial in self.rope_batch_initial_estimates[:-1]:
                                self.initial_estimate.insert(*rope_initial)
                            self.rope_batch_initial_estimates = self.rope_batch_initial_estimates[-1:]

                            # Priors
                            for rope_prior in self.rope_batch_priors[:-1]:
                                self.graph.addPriorPoint2(*rope_prior)
                                prior_counter += 1
                            self.rope_batch_priors = self.rope_batch_priors[-1:]

                            # Factors
                            for rope_factor in self.rope_batch_factors[:-1]:
                                self.graph.add(rope_factor)
                                factor_counter += 1
                            self.rope_batch_factors = self.rope_batch_factors[-1:]
                            self.rope_batch_current_size = len(self.rope_batch_factors)

                        # What is this time out for, forces detections to be added based on timeout
                        elif timeout_condition:
                            if self.verbose_graph_rope_batching:
                                print('Starting Rope Batch Update - Method 3 - Timeout')
                                print(f"Line indices: {self.rope_current_lines}")
                                print(f"Initial estimates: {len(self.rope_batch_initial_estimates)}")
                                print(f"Priors: {len(self.rope_batch_priors)}")
                                print(f"Factors: {len(self.rope_batch_factors)}")

                                # clear line DA indices
                                self.rope_current_lines = []

                            # Initial values
                            for rope_initial in self.rope_batch_initial_estimates:
                                self.initial_estimate.insert(*rope_initial)
                            self.rope_batch_initial_estimates = []

                            # Priors
                            for rope_prior in self.rope_batch_priors:
                                self.graph.addPriorPoint2(*rope_prior)
                                prior_counter += 1
                            self.rope_batch_priors = []

                            # Factors
                            for rope_factor in self.rope_batch_factors:
                                self.graph.add(rope_factor)
                                factor_counter += 1
                            self.rope_batch_factors = []
                            self.rope_batch_current_size = 0

                # Method 2 - Check if the specified number of rope detections have been accumulated
                elif self.rope_batch_current_size >= self.rope_batch_size:
                    if self.verbose_graph_rope_batching:
                        print('Starting Rope Batch Update - Method 2')
                        print(f"Initial estimates: {len(self.rope_batch_initial_estimates)}")
                        print(f"Priors: {len(self.rope_batch_priors)}")
                        print(f"Factors: {len(self.rope_batch_factors)}")

                    # Initial value
                    for rope_initial in self.rope_batch_initial_estimates:
                        self.initial_estimate.insert(*rope_initial)
                        if self.verbose_graph_rope_batching:
                            print(rope_initial)
                    self.rope_batch_initial_estimates = []

                    # Priors
                    if self.verbose_graph_rope_batching:
                        print("Priors:")
                    for rope_prior in self.rope_batch_priors:
                        self.graph.addPriorPoint2(*rope_prior)
                        prior_counter += 1
                        if self.verbose_graph_rope_batching:
                            print(rope_prior)
                    self.rope_batch_priors = []

                    # Factors
                    if self.verbose_graph_rope_batching:
                        print("Factors:")
                    for rope_factor in self.rope_batch_factors:
                        self.graph.add(rope_factor)
                        factor_counter += 1

                        if self.verbose_graph_rope_batching:
                            print(rope_factor)
                    self.rope_batch_factors = []
                    self.rope_batch_current_size = 0

                # Time update process
                start_time = rospy.Time.now()

                # Incremental update
                # TODO Only calculate when required, the current way is required for valid marginals for DA
                self.isam.update(self.graph, self.initial_estimate)
                self.current_estimate = self.isam.calculateEstimate()

                # self.graph.resize(0)
                self.initial_estimate.clear()

                # === Save online estimated pose2 ===
                # Online estimate is saved for later analysis
                self.online_Pose2s.append(self.current_estimate.atPose2(self.x[self.current_x_ind]))

                end_time = rospy.Time.now()
                update_time = (end_time - start_time).to_sec()

                # === record performance metrics of update ===
                # [time(s), factor count, detection state]
                factor_count = self.graph.nrFactors()
                # detection state: 0 = odometry only, 1 = Buoy detection, 2 = rope detection
                self.performance_metrics.append([update_time, factor_count, relative_detection_state, factor_counter,
                                                 prior_counter])

                # === Update inferred priors ===
                if self.update_priors:
                    self.rope_update()
                    self.update_rope_priors()

                # ===== debugging and visualizations =====
                # Log to terminal
                if self.verbose_graph_update:
                    print(f"Done with update - x{self.current_x_ind}: {update_time} s")

                # === Debug publishing ===
                self.publish_estimated_pos_marker_and_transform(computed_est, plot_full_state=True)
                self.publish_est_path(plot_full_state=True)
                self.publish_rope_detections()
                self.publish_est_buoys()
                if not self.individual_rope_detections:
                    self.publish_est_line()
                    # self.publish_est_line_verbose(0)
                    # self.publish_est_line_verbose(1)
                    # self.publish_est_line_verbose(2)

                # Buoy detection
                if relative_detection is not None and da_id != -ObjectID.ROPE.value:
                    if valid_buoy_da:
                        self.publish_detection_markers(buoy_association_id)
                    else:
                        self.publish_detection_markers(-1)

            self.busy_queue = False
            return
        return

    def associate_detection(self, detection_map_location):
        """
        Basic association of detection with a buoys, currently using the prior locations
        Will return the id of the closest buoy and the distance between the detection and that buoy
        """
        # TODO update to use the current estimated buoy locations
        best_id = -1
        best_range_2 = np.inf

        for i, buoy_loc in enumerate(self.buoy_priors):
            range_2 = (buoy_loc[0] - detection_map_location[0]) ** 2 + (buoy_loc[1] - detection_map_location[1]) ** 2

            if range_2 < best_range_2:
                best_range_2 = range_2
                best_id = i

        return best_id, best_range_2 ** (1 / 2)

    def associate_detection_likelihood(self, detection_map_location, detection_covariance):
        """
        Updated version of the data association
        Basic association of detection with a buoys, currently using the prior locations
        Will return the id of the closest buoy and the distance between the detection and that buoy

        :param detection_map_location:
        :param detection_covariance:
        :param current_marginals:
        :return: max likelihood index, 2-norm distance, mahalanobis distance
        """

        likelihoods = []
        distances_2_norm = []
        distances_m = []

        # check detection covariance shape
        # The raw covariance is x,y,theta in this implementation the non-linear aspects are ignored.
        # The uncertainty of the measurement is taken to be the positional uncertainty of the agent
        # The mean value of the measurement is the mean value of the agent with the relative detection applied.
        if detection_covariance.shape[0] == 3:
            detection_covariance = detection_covariance[0:2, 0:2]

        # check the mean shape, same as above
        detection_mean = np.array(detection_map_location[0:2])

        for buoy_ind, buoy_key in enumerate(self.b):
            # check if key is present, if not use priors
            if buoy_key in self.current_estimate.keys():
                buoy_mean = self.current_estimate.atPoint2(buoy_key)
                buoy_covar = self.current_marginals.marginalCovariance(buoy_key)
            else:
                buoy_mean = np.array(self.buoy_priors[buoy_ind][0:2])
                buoy_covar = np.identity(2) * 0.001  # if no covariance is supplied give very small uncertainty

            # Calculate likelihood
            total_covar = detection_covariance + buoy_covar  # Assuming they are independent
            likelihood = stats.multivariate_normal.pdf(detection_mean,
                                                             mean=buoy_mean,
                                                             cov=total_covar)

            likelihoods.append(likelihood)

            # Calculate distance, 2-norm
            distance = scipy.linalg.norm(detection_mean - buoy_mean)

            distances_2_norm.append(distance)

            # calculate distances, mahalanobis
            total_covar_inv = scipy.linalg.inv(total_covar)
            distance_m = scipy.spatial.distance.mahalanobis(detection_mean, buoy_mean, total_covar_inv)

            distances_m.append(distance_m)

        # Analysis
        likelihood_max_ind = np.argmax(likelihoods)
        # distance_min_ind = np.argmin(distances_2_norm)

        if self.verbose_graph_buoy_associations:
            print("Buoy detection - stats")
            print(f"Likelihood: {likelihoods[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, Mahalanobis: {distances_m[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, 2-norm: {distances_2_norm[likelihood_max_ind]:.2e} ({likelihood_max_ind})")

        return likelihood_max_ind, distances_2_norm[likelihood_max_ind], distances_m[likelihood_max_ind]

    def associate_rope_detection(self, detection_map_location):
        """
        Basic association of rope detections to rope priors.

        See test_covar.py for
        :param detection_map_location:
        :return:
        """

        # Find the correspondence between detection and rope prior, closest for now
        min_distance = np.inf
        min_ind = -1
        for rope_ind, rope in enumerate(self.rope_prior_ends):
            start_x, start_y = rope[0][0], rope[0][1]
            end_x, end_y = rope[1][0], rope[1][1]

            distance = online_slam_2d.calculate_line_point_distance(start_x, start_y, end_x, end_y,
                                                                    detection_map_location[0],
                                                                    detection_map_location[1])

            if distance < min_distance:
                min_distance = distance
                min_ind = rope_ind

        return min_ind

    def associate_rope_detection_likelihood(self, detection_map_location, detection_covariance):
        """
        This is an updated version of the basic association of rope detections to rope priors.
        Follows the form of associate_detection_likelihood.

        See test_covar.py for

        :param detection_map_location:
        :param detection_covariance:
        :return:
        """

        likelihoods = []
        distances_2_norm = []
        distances_m = []

        # check detection covariance shape
        # The raw covariance is x,y,theta in this implementation the non-linear aspects are ignored.
        # The uncertainty of the measurement is taken to be the positional uncertainty of the agent
        # The mean value of the measurement is the mean value of the agent with the relative detection applied.
        if detection_covariance.shape[0] == 3:
            detection_covariance = detection_covariance[0:2, 0:2]

        # Find the correspondence between detection and rope prior, closest for now
        min_distance = np.inf
        min_ind = -1
        for rope_ind, rope in enumerate(self.rope_prior_ends):
            # Rope start and end coords
            start_x, start_y = rope[0][0], rope[0][1]
            end_x, end_y = rope[1][0], rope[1][1]

            # rope mean and covariance
            rope_mean = np.array(self.rope_prior_centers[rope_ind])
            rope_covar = self.rope_noise_models[rope_ind].covariance()

            # Total covariance, assuming they are independent
            total_covar = detection_covariance + rope_covar

            # Calculate likelihood
            likelihood = stats.multivariate_normal.pdf(detection_map_location,
                                                             mean=rope_mean,
                                                             cov=total_covar)

            likelihoods.append(likelihood)

            # Calculate distance, 2-norm
            distance = online_slam_2d.calculate_line_point_distance(start_x, start_y, end_x, end_y,
                                                                    detection_map_location[0],
                                                                    detection_map_location[1])
            distances_2_norm.append(distance)

            # calculate distances, mahalanobis
            total_covar_inv = scipy.linalg.inv(total_covar)
            distance_m = scipy.spatial.distance.mahalanobis(detection_map_location, rope_mean, total_covar_inv)

            distances_m.append(distance_m)

        # Analysis
        likelihood_max_ind = np.argmax(likelihoods)
        # distance_min_ind = np.argmin(distances_2_norm)

        if self.verbose_graph_rope_associations:
            print("Rope detection - stats")
            print(f"Likelihood: {likelihoods[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, Mahalanobis: {distances_m[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, 2-norm: {distances_2_norm[likelihood_max_ind]:.2e} ({likelihood_max_ind})")

        return likelihood_max_ind, distances_2_norm[likelihood_max_ind], distances_m[likelihood_max_ind]

    def associate_rope_detection_likelihood_2(self, detection_map_location, detection_covariance):
        """
        This is an updated version of the updated association of rope detections to rope priors.
        Follows the form of associate_detection_likelihood.

        See test_covar.py for

        # TODO figure out a beter method

        :param detection_map_location:
        :param detection_covariance:
        :return:
        """

        likelihoods = []
        distances_2_norm = []
        distances_m = []

        # check detection covariance shape
        # The raw covariance is x,y,theta in this implementation the non-linear aspects are ignored.
        # The uncertainty of the measurement is taken to be the positional uncertainty of the agent
        # The mean value of the measurement is the mean value of the agent with the relative detection applied.
        if detection_covariance.shape[0] == 3:
            detection_covariance = detection_covariance[0:2, 0:2]

        # Find the correspondence between detection and rope prior, closest for now
        min_distance = np.inf
        min_ind = -1
        for rope_ind, rope in enumerate(self.rope_prior_ends):
            # Rope start and end coords
            start_x, start_y = rope[0][0], rope[0][1]
            end_x, end_y = rope[1][0], rope[1][1]

            # rope mean and covariance
            rope_mean = np.array(self.rope_prior_centers[rope_ind])
            rope_covar = self.rope_noise_models[rope_ind].covariance()

            # Total covariance, assuming they are independent
            total_covar = detection_covariance + rope_covar

            # Calculate likelihood
            likelihood = stats.multivariate_normal.pdf(detection_map_location,
                                                             mean=rope_mean,
                                                             cov=total_covar)

            likelihoods.append(likelihood)

            # Calculate distance, 2-norm
            distance = online_slam_2d.calculate_line_point_distance(start_x, start_y, end_x, end_y,
                                                                    detection_map_location[0],
                                                                    detection_map_location[1])
            distances_2_norm.append(distance)

            # calculate distances, mahalanobis
            total_covar_inv = scipy.linalg.inv(total_covar)
            distance_m = scipy.spatial.distance.mahalanobis(detection_map_location, rope_mean, total_covar_inv)

            distances_m.append(distance_m)

        # Analysis
        likelihood_max_ind = np.argmax(likelihoods)
        # distance_min_ind = np.argmin(distances_2_norm)

        if self.verbose_graph_rope_associations:
            print("Rope detection - stats")
            print(f"Likelihood: {likelihoods[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, Mahalanobis: {distances_m[likelihood_max_ind]:.2e} ({likelihood_max_ind})")
            print(f"Distance, 2-norm: {distances_2_norm[likelihood_max_ind]:.2e} ({likelihood_max_ind})")

        return likelihood_max_ind, distances_2_norm[likelihood_max_ind], distances_m[likelihood_max_ind]

    def update_rope_priors(self, debug=False):
        """
        The rope priors are currently based on the buoy positions. As the estimate of the buoy position is updated
        the inferred rope priors of past rope detections should be updated. Detections are not re-associated.
        :return:
        """

        for factor_idx in range(self.graph.size()):
            factor = self.graph.at(factor_idx)

            if isinstance(factor, gtsam.PriorFactorPoint2):
                # a prior should only have a single key
                factor_key = factor.keys()[0]
                key_idx = None  # index of self.r corresponding key of interest

                # find if prior is attached to a rope detection landmark
                for dict_key, dict_val in self.r.items():
                    if dict_val == factor_key:
                        key_idx = dict_key

                if key_idx is None:
                    continue

                if debug:
                    print("Original factor")
                    print(factor)

                new_prior = gtsam.PriorFactorPoint2(factor_key,
                                                    self.rope_prior_centers[self.r_associations[key_idx]],
                                                    self.rope_noise_models[self.r_associations[key_idx]])
                self.graph.replace(factor_idx, new_prior)

                if debug:
                    print("Updated factor")
                    print(self.graph.at(factor_idx))
                    debug = False

    # RViz visualization publishers
    def publish_detection_markers(self, da_id):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """

        # === Estimated detection location ===
        # Form detection marker message
        detection_marker = Marker()
        detection_marker.header.frame_id = self.map_frame
        detection_marker.type = Marker.SPHERE
        detection_marker.action = Marker.ADD
        detection_marker.id = 0
        detection_marker.lifetime = rospy.Duration(10)
        detection_marker.pose.position.x = self.est_detect_loc[0]
        detection_marker.pose.position.y = self.est_detect_loc[1]
        detection_marker.pose.position.z = 0.0
        detection_marker.pose.orientation.x = 0
        detection_marker.pose.orientation.y = 0
        detection_marker.pose.orientation.z = 0
        detection_marker.pose.orientation.w = 1
        detection_marker.scale.x = self.marker_scale
        detection_marker.scale.y = self.marker_scale
        detection_marker.scale.z = self.marker_scale
        # Color to indicate valid or invalid
        if da_id < 0:
            detection_marker.color = self.buoy_invalid_color
        else:
            detection_marker.color = self.buoy_valid_color

        # Publish detection marker message
        self.est_detect_loc_pub.publish(detection_marker)

        ### Data association marker
        # Only publish marker for valid DA
        if da_id > 0:
            # Form DA marker message
            da_marker = Marker()
            da_marker.header.frame_id = self.map_frame
            da_marker.type = Marker.CYLINDER
            da_marker.action = Marker.ADD
            da_marker.id = 0
            da_marker.lifetime = rospy.Duration(10)
            da_marker.pose.position.x = self.buoy_priors[da_id][0]
            da_marker.pose.position.y = self.buoy_priors[da_id][1]
            da_marker.pose.position.z = 0.0
            da_marker.pose.orientation.x = 0
            da_marker.pose.orientation.y = 0
            da_marker.pose.orientation.z = 0
            da_marker.pose.orientation.w = 1
            da_marker.scale.x = self.marker_scale / 2
            da_marker.scale.y = self.marker_scale / 2
            da_marker.scale.z = self.marker_scale * 2
            da_marker.color.r = 1.0
            da_marker.color.g = 0.0
            da_marker.color.b = 1.0
            da_marker.color.a = 1.0

            # Publish DA marker message
            self.da_pub.publish(da_marker)

    def publish_estimated_pos_marker_and_transform(self, est_pos, plot_full_state=False):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """
        if plot_full_state:
            # TODO explicitly look up correct roll, pitch, depth
            # Currently assuming the most recent corresponds to the correct values
            heading_quat = quaternion_from_euler(self.dr_pose_rpd[-1][0], self.dr_pose_rpd[-1][1], est_pos.theta())
            heading_quaternion_type = Quaternion(*heading_quat)
            est_z = self.dr_pose_rpd[-1][2]  # depth
        else:
            heading_quat = quaternion_from_euler(0, 0, est_pos.theta())
            heading_quaternion_type = Quaternion(*heading_quat)
            est_z = 0.0

        # Estimated detection location
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = 0
        marker.lifetime = rospy.Duration(0)
        marker.pose.position.x = est_pos.x()
        marker.pose.position.y = est_pos.y()
        marker.pose.position.z = est_z
        marker.pose.orientation = heading_quaternion_type
        marker.scale.x = self.est_marker_x
        marker.scale.y = self.est_marker_y
        marker.scale.z = self.est_marker_z
        marker.color.r = 0.1
        marker.color.g = 0.1
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.est_pos_pub.publish(marker)

        # print(f"Transform broadcaster: estimated pose {self.map_frame} -> {self.est_frame}")
        try:
            self.tf_br.sendTransform(translation=(est_pos.x(), est_pos.y(), 0),
                                     rotation=(heading_quaternion_type.x,
                                               heading_quaternion_type.y,
                                               heading_quaternion_type.z,
                                               heading_quaternion_type.w),
                                     time=rospy.Time.now(),
                                     child=self.est_frame,
                                     parent=self.map_frame)

        except rospy.ROSException as e:
            rospy.logerr('Error broadcasting tf transform: {}'.format(str(e)))

    def publish_est_path(self, plot_full_state=False):

        poses_path = Path()
        poses_path.header.frame_id = self.map_frame

        # Copy so elements can be added to the graph will looping over the elements
        key_dict_copy = self.x.copy()  # {x index: gtsam key}

        for x_index, gtsam_key in key_dict_copy.items():
            if not self.current_estimate.exists(gtsam_key):
                print('publish_est_path: Key not found!')
                continue
            if plot_full_state and 0 <= x_index < len(self.dr_pose_rpd):
                roll = self.dr_pose_rpd[x_index][0]
                pitch = self.dr_pose_rpd[x_index][1]
                depth = self.dr_pose_rpd[x_index][2]
            else:
                roll = 0
                pitch = 0
                depth = 0

            pose = self.current_estimate.atPose2(gtsam_key)
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = pose.x()
            pose_stamped.pose.position.y = pose.y()
            pose_stamped.pose.position.z = depth

            heading_quat = quaternion_from_euler(roll, pitch, pose.theta())
            heading_quaternion_type = Quaternion(*heading_quat)
            pose_stamped.pose.orientation = heading_quaternion_type

            pose_stamped.header.frame_id = self.map_frame
            pose_stamped.header.stamp = rospy.Time.now()

            poses_path.poses.append(pose_stamped)

        self.est_path_pub.publish(poses_path)

    def publish_rope_detections(self):
        """
        This is responsible for publishing the current estimated rope detections
        :return:
        """
        # Check that self.r has been initialized
        if self.r is None:
            return

        # Copy as elements can be added to the graph will looping over the elements
        key_dict_copy = self.r.copy()

        # Check for the existence of rope detections
        if len(key_dict_copy) <= 0:
            return

        rope_detections = MarkerArray()
        valid_detection_count = 0  # Was having a problem with keys being present in self.r but not in the values

        for detection_ind, key in enumerate(key_dict_copy.values()):
            if not self.current_estimate.exists(key):
                print('publish_rope_detections: Key not found!')
                continue
            valid_detection_count += 1
            detection_point = self.current_estimate.atPoint2(key)
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = detection_ind
            marker.lifetime = rospy.Duration(0)
            marker.pose.position.x = detection_point[0]
            marker.pose.position.y = detection_point[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = self.rope_marker_scale
            marker.scale.y = self.rope_marker_scale
            marker.scale.z = self.rope_marker_scale
            marker.color = self.rope_marker_color

            rope_detections.markers.append(marker)

        if valid_detection_count > 0:
            self.est_rope_pub.publish(rope_detections)

    def publish_est_buoys(self):
        """
        This is responsible for publishing the current estimated buoy locations

        :return:
        """
        # Check that self.r has been initialized
        if self.b is None:
            return

        # Copy as elements can be added to the graph will looping over the elements
        key_dict_copy = self.b.copy()

        # Check for the existence of rope detections
        if len(key_dict_copy) <= 0:
            return

        est_buoys = MarkerArray()
        valid_detection_count = 0  # Was having a problem with keys being present in self.b but not in the values

        for buoy_ind, key in enumerate(key_dict_copy.values()):
            if not self.current_estimate.exists(key):
                print('publish_est_buoys: Key not found!')
                continue
            valid_detection_count += 1
            detection_point = self.current_estimate.atPoint2(key)
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = buoy_ind
            marker.lifetime = rospy.Duration(0)
            marker.pose.position.x = detection_point[0]
            marker.pose.position.y = detection_point[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = self.buoy_marker_scale
            marker.scale.y = self.buoy_marker_scale
            marker.scale.z = self.buoy_marker_scale
            marker.color = self.buoy_marker_color

            est_buoys.markers.append(marker)

        if valid_detection_count > 0:
            self.est_buoy_pub.publish(est_buoys)

    def publish_est_line(self):
        """
        This is responsible for publishing the current estimated line centers.
        These will only be updated if individual_rope_detections is set to False.

        This is part of the Nacho method

        :return:
        """
        # Check that self.l has been initialized
        if self.l is None:
            return

        # Copy as elements can be added to the graph will looping over the elements
        key_dict_copy = self.l.copy()

        # Check for the existence of rope detections
        if len(key_dict_copy) <= 0:
            return

        est_lines = MarkerArray()
        valid_detection_count = 0  # Was having a problem with keys being present in self.l but not in the values

        for buoy_ind, key in enumerate(key_dict_copy.values()):
            if not self.current_estimate.exists(key):
                print('publish_est_buoys: Key not found!')
                continue
            valid_detection_count += 1
            detection_point = self.current_estimate.atPoint2(key)
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = buoy_ind
            marker.lifetime = rospy.Duration(0)
            marker.pose.position.x = detection_point[0]
            marker.pose.position.y = detection_point[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = self.line_marker_scale
            marker.scale.y = self.line_marker_scale
            marker.scale.z = self.line_marker_scale
            marker.color = self.line_marker_color

            est_lines.markers.append(marker)

        if valid_detection_count > 0:
            self.est_line_pub.publish(est_lines)

    def publish_est_line_verbose(self, rope_id):
        # Check that self.l has been initialized
        if self.l is None:
            return

        # Check for current marginals
        if self.current_marginals is None:
            return

        if 0 > rope_id >= self.n_ropes:
            return

        # Check for valid key
        key = self.l[rope_id]
        if key not in self.current_estimate.keys():
            return

        # Find covariance of the current key
        rope_covariance = self.current_marginals.marginalCovariance(key)

        landmark_point = self.current_estimate.atPoint2(key)

        print(f"ROPE {rope_id}: {rope_covariance}")

        # gtsam_plot.plot_point2(0,
        #                        self.current_estimate.atPoint2(key),
        #                        'g',
        #                        P=self.current_marginals.marginalCovariance(key))
        #
        # plt.show()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = landmark_point[0]
        msg.pose.pose.position.y = landmark_point[1]
        msg.pose.pose.position.z = 0.0

        # Set the orientation (you can use Euler angles or quaternion)
        # Example using Euler angles:
        roll, pitch, yaw = 0.0, 0.0, 0.0
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        msg.pose.pose.orientation.x = quaternion[0]
        msg.pose.pose.orientation.y = quaternion[1]
        msg.pose.pose.orientation.z = quaternion[2]
        msg.pose.pose.orientation.w = quaternion[3]

        # Set the covariance matrix (adjust this as needed)
        xx = rope_covariance[0, 0]
        xy = rope_covariance[0, 1]
        yy = rope_covariance[1, 1]
        msg.pose.covariance = [xx, xy, 0, 0, 0, 0,
                               xy, yy, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0]

        self.est_line_pub_verbose.publish(msg)

    def check_seq_id(self, current_seq_id):
        # checks if current_seq_id is within the bounds of the defined
        current_swath_ind = -1
        optimize = False
        for swath_ind, swath in enumerate(self.swath_seq_ids):
            # Check if current_seq_id is in bounds
            for sub_swath in swath:
                if sub_swath[0] <= current_seq_id <= sub_swath[-1]:
                    current_swath_ind = swath_ind

            # Check if current_seq_ids is max -> time to optimize
            swath_max = max(max(sub_swath) for sub_swath in swath)
            if current_seq_id == swath_max:
                optimize = True

        return current_swath_ind, optimize

    # Utility static methods
    @staticmethod
    def calculate_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        return angle

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    @staticmethod
    def calculate_center(x1, y1, x2, y2):
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        return np.array((x_center, y_center), dtype=np.float64)

    @staticmethod
    def calculate_line_point_distance(x1, y1, x2, y2, x3, y3):
        """
        points 1 and 2 form a line segment, point 3 is
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param x3:
        :param y3:
        :return:
        """
        if x1 == x2 and y1 == y2:
            return -1

        # Calculate the length of the line segment
        line_mag_sqrd = (x2 - x1) ** 2 + (y2 - y1) ** 2
        u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / line_mag_sqrd

        if 0 < u < 1.0:
            x_perpendicular = x1 + u * (x2 - x1)
            y_perpendicular = y1 + u * (y2 - y1)

            return math.sqrt((x_perpendicular - x3) ** 2 + (y_perpendicular - y3) ** 2)

        else:
            # Calculate the distance from the third point to each endpoint of the line segment
            distance_line_end_1 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
            distance_line_end_2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

            # Find the minimum distance
            min_distance = min(distance_line_end_1, distance_line_end_2)

            return min_distance


class analyze_slam:
    """
    Responsible for analysis of slam results
    # TODO Need to think about how to unify the online and offline classes
    slam_object.graph: gtsam.NonlinearFactorGraph
    slam_object.dr_pose2s: list[gtsam.Pose2]
    slam_object.gt_pose2s: list[gtsam.Pose2]

    === Saving for post processing ===
    - save_for_sensor_processing():
    - save_2d_poses():

    === Plotting ===
    - visualize_raw():
        Basic plotting of gt, dr, and estimated poses
    - visualize_final():
        This is the primary graphing method that uses matplotlib
        > can plot rope detections as well as
    - visualize_online():
    - show_graph_2d():
        This uses networkx for visualizing the graph. Slower and harder to work with than matplotlib.
    === Error analysis ===


    """

    # Saving methods
    def __init__(self, slam_object: offline_slam_2d | online_slam_2d, output_path=None):
        # Directory for saving analysis
        # self.file_path = output_path
        if output_path is None or not os.path.isdir(output_path):
            self.file_path = ''
        else:
            if output_path[-1] != '/':
                output_path = output_path + '/'
            self.file_path = output_path

        # unpack slam object
        self.slam = slam_object
        self.graph = slam_object.graph
        self.current_estimate = slam_object.current_estimate
        self.x = slam_object.x  # pose keys
        self.b = slam_object.b  # point keys
        self.r = slam_object.r  # rope keys (not found in offline processing)
        self.r_associations = slam_object.r_associations

        # Dead reckoning poses and the between poses, ground truth poses
        self.dr_poses = pose2_list_to_nparray(slam_object.dr_Pose2s)
        self.gt_poses = pose2_list_to_nparray(slam_object.gt_Pose2s)
        self.between_Pose2s = pose2_list_to_nparray(slam_object.between_Pose2s)

        # Added for data analysis
        if hasattr(slam_object, 'online_Pose2s'):
            self.online_Pose2s = slam_object.online_Pose2s
            self.online_poses = pose2_list_to_nparray(slam_object.online_Pose2s)
        else:
            self.online_Pose2s = None
            self.online_poses = None

        # ===== Buoys =====
        self.buoy_priors = slam_object.buoy_priors
        if self.buoy_priors is None:
            self.n_buoys = 0
        else:
            self.n_buoys = len(self.buoy_priors)

        # ===== Ropes =====
        if self.r is None:
            self.n_rope_detects = 0
        else:
            self.n_rope_detects = len(self.r)

        self.corresponding_detections = None  # This is calculating the error metric
        self.corresponding_distances = None

        # ===== Build arrays for the poses and points of the posterior =====
        # These arrays might make it easier to plot stuff
        self.posterior_poses = np.zeros((len(self.x), 3))
        for i in range(len(self.x)):
            x_key_i = self.x[i]
            if self.current_estimate.exists(x_key_i):
                self.posterior_poses[i, 0] = self.current_estimate.atPose2(x_key_i).x()
                self.posterior_poses[i, 1] = self.current_estimate.atPose2(x_key_i).y()
                self.posterior_poses[i, 2] = self.current_estimate.atPose2(x_key_i).theta()
            else:
                rospy.logwarn(f"X key {x_key_i} does not exist in the current estimate.")

        if self.n_buoys > 0:
            self.posterior_buoys = np.zeros((self.n_buoys, 2))
            for i in range(self.n_buoys):
                b_key_i = self.b[i]
                if self.current_estimate.exists(b_key_i):
                    self.posterior_buoys[i, 0] = self.current_estimate.atPoint2(b_key_i)[0]
                    self.posterior_buoys[i, 1] = self.current_estimate.atPoint2(b_key_i)[1]
                else:
                    rospy.logwarn(f"B key {b_key_i} does not exist in the current estimate.")

        if self.n_rope_detects > 0:
            self.posterior_rope_detects = np.zeros((self.n_rope_detects, 2))
            self.posterior_rope_associations = np.zeros((self.n_rope_detects,))
            for i in range(self.n_rope_detects):
                r_key_i = self.r[i]
                if self.current_estimate.exists(r_key_i):
                    self.posterior_rope_detects[i, 0] = self.current_estimate.atPoint2(self.r[i])[0]
                    self.posterior_rope_detects[i, 1] = self.current_estimate.atPoint2(self.r[i])[1]
                else:
                    rospy.logwarn(f"R key {r_key_i} does not exist in the current estimate.")

                self.posterior_rope_associations[i] = self.r_associations.get(i, -1)

        # ===== Unpack more relevant data from the slam object =====
        """
        Current differences include:
        detection_graph: list of detections with id to relate the to dr indices, only found in offline version
        buoy2cluster: mapping from buoy to cluster, used for plotting offline buoys and clustering
        initial_estimate: The dr poses serve as the initial estimate for the offline version
        da_check: The online version uses the ground truth and the relative detection data to find the DA ground truth
        """
        # TODO Unify the online and offline versions of the analysis
        # TODO Maybe just drop offline version or learn how to code to make this less of a mess...

        if hasattr(slam_object, 'detections_graph'):
            self.detections = slam_object.detections_graph
            self.n_detections = len(self.detections)
        else:
            self.detections_graph = None
            self.n_detections = 0

        if hasattr(slam_object, 'buoy2cluster'):
            self.buoy2cluster = slam_object.buoy2cluster
        else:
            self.buoy2cluster = None

        if hasattr(slam_object, 'initial_estimate'):
            self.initial_estimate = slam_object.initial_estimate
        else:
            self.initial_estimate = None

        if hasattr(slam_object, 'da_check'):
            self.da_check = slam_object.da_check
        else:
            self.initial_estimate = None

        if hasattr(slam_object, 'n_clusters'):
            self.n_clusters = slam_object.n_clusters
        else:
            self.n_clusters = None

        if hasattr(slam_object, 'rope_buoy_ind'):
            self.rope_buoy_ind = slam_object.rope_buoy_ind
            self.n_ropes = len(slam_object.rope_buoy_ind)
        else:
            self.rope_buoy_ind = None
            self.n_ropes = 0

        if hasattr(slam_object, 'buoy_detection_info'):
            self.buoy_detection_info = np.array(slam_object.buoy_detection_info)
        else:
            self.buoy_detection_info = None

        if hasattr(slam_object, 'da_distance_threshold'):
            self.da_distance_threshold = np.array(slam_object.da_distance_threshold)
        else:
            self.buoy_detection_info = -1

        if hasattr(slam_object, 'da_m_distance_threshold'):
            self.da_m_distance_threshold = np.array(slam_object.da_m_distance_threshold)
        else:
            self.buoy_detection_info = -1

        if hasattr(slam_object, 'performance_metrics'):
            self.performance_metrics = np.array(slam_object.performance_metrics)
        else:
            self.performance_metrics = None

        if hasattr(slam_object, 'rope_infos'):
            self.rope_infos = np.array(slam_object.rope_infos)
        else:
            self.rope_infos = None

        # ===== Visualization parameters =====
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.online_color = 'm'
        self.rope_color = 'b'
        self.buoy_color = 'k'
        self.title_size = 16
        self.legend_size = 14
        self.label_size = 14
        self.colors = ['orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        # Set plot limits
        self.x_tick = 5
        self.y_tick = 5
        self.plot_limits = None
        self.find_plot_limits()
        self.figure_width = 8
        self.figure_height = 8

    def save_for_sensor_processing(self):
        """
        Saves three files related to the camera: camera_gt.csv, camera_dr.csv, camera_est.csv
        Saves three files related to the sss: sss_gt.csv, sss_dr.csv, sss_est.csv
        saves a file of the estimated buoy positions
        format: [[x, y, z, q_w, q_x, q_y, q_z, img seq #]]

        :return:
        """
        if self.file_path is None:
            print("Analysis output path not specified")
            return

        # ===== Save base link (wrt map) poses =====
        camera_gt = []
        camera_dr = []
        camera_est = []

        sss_gt = []
        sss_dr = []
        sss_est = []

        # form the required list of lists
        # exclude poses that do not correspond to captured images
        for key, value in self.slam.sensor_string_at_key.items():
            # Extract node type from sensor_string_at_key
            if value == 'odometry' or value == 'detection':
                continue
            if "_" in value:
                sensor_type, sensor_id = value.split("_")
                sensor_id = int(sensor_id)
            else:
                print("Malformed sensor information")
                continue

            # DR and GT for the sensor reading
            sensor_gt_pose = self.slam.gt_pose_raw[key][0:7]
            sensor_gt_pose.append(sensor_id)

            sensor_dr_pose = self.slam.dr_pose_raw[key][0:7]
            sensor_dr_pose.append(sensor_id)

            # Estimated pose for the sensor reading
            """
            Initially I saved the roll and pitch reported by dr odom and combined those with the estimated
            yaw to for the new estimated 3d pose but that was giving strange results...

            New plan is to extract the roll and pith in the NOW corrected dr pose info. Then combine with the estimated
            yaw to form the new 3d pose quaternion
            """

            # Roll, pitch, and depth are provided from the odometry
            roll_old = self.slam.dr_pose_rpd[key][0]
            pitch_old = self.slam.dr_pose_rpd[key][1]

            # This quaternion is stored [w, x, y, z]
            dr_q = self.slam.dr_pose_raw[key][3:7]
            # This function expects a quaternions of the form [x, y, z, w]
            dr_rpy = euler_from_quaternion([dr_q[1], dr_q[2], dr_q[3], dr_q[0]])
            roll = dr_rpy[0]
            pitch = dr_rpy[1]
            depth = self.slam.dr_pose_rpd[key][2]

            # X, Y, and yaw are estimated using the factor graph
            est_x = self.posterior_poses[key, 0]
            est_y = self.posterior_poses[key, 1]
            est_yaw = self.posterior_poses[key, 2]

            quats = quaternion_from_euler(roll, pitch, est_yaw)  # output: [x, y, z, w]

            # This quaternion is stored [w, x, y, z]
            sensor_est_pose = [est_x, est_y, -depth,
                               quats[3], quats[0], quats[1], quats[2],
                               sensor_id]

            if sensor_type == "cam":
                camera_gt.append(sensor_gt_pose)
                camera_dr.append(sensor_dr_pose)
                camera_est.append(sensor_est_pose)
            elif sensor_type == "sss":
                sss_gt.append(sensor_gt_pose)
                sss_dr.append(sensor_dr_pose)
                sss_est.append(sensor_est_pose)
            else:
                print("Unknown sensor type")

        # write to camera and sss files
        if len(camera_gt) > 0:
            write_array_to_csv(self.file_path + 'camera_gt.csv', camera_gt)
            write_array_to_csv(self.file_path + 'camera_dr.csv', camera_dr)
            write_array_to_csv(self.file_path + 'camera_est.csv', camera_est)

        if len(sss_gt) > 0:
            write_array_to_csv(self.file_path + 'sss_gt.csv', sss_gt)
            write_array_to_csv(self.file_path + 'sss_dr.csv', sss_dr)
            write_array_to_csv(self.file_path + 'sss_est.csv', sss_est)

        # ===== Save buoy estimated positions =====
        # only the x an y coords are estimated, buoys are assumed to have z = 0
        if self.n_buoys > 0:
            buoys_est = np.zeros((self.n_buoys, 3))

            for i in range(self.n_buoys):
                buoys_est[i, 0] = self.current_estimate.atPoint2(self.b[i])[0]
                buoys_est[i, 1] = self.current_estimate.atPoint2(self.b[i])[1]

            # Write to file
            write_array_to_csv(self.file_path + 'buoys_est.csv', buoys_est)

    def save_3d_poses(self):
        """
        Saves four files of the 3d poses of gt, dr, estimate or final, online
        format: [[timestamp, x, y, z, q_x, q_y, q_z, q_w]]

        Currently, the timestamp is just the index to allow for association between dr and estimation

        NOTE: The GT isn't the true GT because only the pose2 is passed to the analyzer

        :return:
        """

        print("Analysis: save_3d_poses")

        yaw_threshold = 0.01
        yaw_error_flag = False

        if self.file_path is None:
            print("Analysis output path not specified")
            return

        gt_3d_poses = []
        dr_3d_poses = []
        est_3d_poses = []
        online_3d_poses = []

        # form the required list of lists
        # There is some sort of mismatch between the length of self.x and self.gt_poses
        pose_count = len(self.gt_poses)  # All the below poses should have the same
        for x_ind in range(len(self.x)):
            try:
                # GT
                gt_x, gt_y, gt_yaw = self.gt_poses[x_ind, :3]

                # DR
                dr_x, dr_y, dr_yaw = self.dr_poses[x_ind, :3]

                # X, Y, and yaw are estimated using the factor graph
                est_x, est_y, est_yaw = self.posterior_poses[x_ind, :3]

                online_x, online_y, online_yaw = self.online_poses[x_ind, :3]
            except IndexError:
                print("save_3d_pose(): Mismatch in pose count")
                break

            # Estimated 3D pose (final)
            """
            Initially I saved the roll and pitch reported by dr odom and combined those with the estimated
            yaw to for the new estimated 3d pose but that was giving strange results...

            New plan is to extract the roll and pith in the NOW corrected dr pose info. Then combine with the estimated
            yaw to form the new 3d pose quaternion
            """

            # Roll, pitch, and depth are provided from the odometry
            # roll_old = self.slam.dr_pose_rpd[key][0]
            # pitch_old = self.slam.dr_pose_rpd[key][1]

            # Extract quaternions of current DR
            dr_quats = self.slam.dr_pose_raw[x_ind][3:7]  # This quaternion is stored [w, x, y, z]
            dr_quats_xyzw = [dr_quats[1], dr_quats[2], dr_quats[3], dr_quats[0]]

            # Convert from quaternions of the form [x, y, z, w] -> roll, pitch, yaw
            # This function expects a quaternions of the form [x, y, z, w]
            dr_rpy = euler_from_quaternion(dr_quats_xyzw)

            # Check for agreement between the DR data, dr_poses and dr_pose_raw
            if abs(dr_yaw - dr_rpy[2]) > yaw_threshold:
                yaw_error_flag = True
                break

            roll = dr_rpy[0]
            pitch = dr_rpy[1]
            depth = self.slam.dr_pose_rpd[x_ind][2]

            gt_quats_xyzw = quaternion_from_euler(roll, pitch, gt_yaw)  # output: [x, y, z, w]
            est_quats_xyzw = quaternion_from_euler(roll, pitch, est_yaw)  # output: [x, y, z, w]
            online_quats_xyzw = quaternion_from_euler(roll, pitch, online_yaw)  # output: [x, y, z, w]

            # Format: [timestamp, x, y, z, q_x, q_y, q_z, q_w]
            # NOTE: the order of of the quaternions, I'm sorry GTSAM is w,x,y,z while ROS is x,y,z,w

            gt_3d_pose = [x_ind,
                          gt_x, gt_y, -depth,
                          gt_quats_xyzw[0], gt_quats_xyzw[1], gt_quats_xyzw[2], gt_quats_xyzw[3]]

            dr_3d_pose = [x_ind,
                          dr_x, dr_y, -depth,
                          dr_quats_xyzw[0], dr_quats_xyzw[1], dr_quats_xyzw[2], dr_quats_xyzw[3]]

            est_3d_pose = [x_ind,
                           est_x, est_y, -depth,
                           est_quats_xyzw[0], est_quats_xyzw[1], est_quats_xyzw[2], est_quats_xyzw[3]]

            online_3d_pose = [x_ind,
                              online_x, online_y, -depth,
                              online_quats_xyzw[0], online_quats_xyzw[1], online_quats_xyzw[2], online_quats_xyzw[3]]

            gt_3d_poses.append(gt_3d_pose)
            dr_3d_poses.append(dr_3d_pose)
            est_3d_poses.append(est_3d_pose)
            online_3d_poses.append(online_3d_pose)

        if yaw_error_flag:
            print("Error: save_3d_poses(), yaw_threshold exceeding")

        write_array_to_csv(self.file_path + 'analysis_gt_3d.csv', gt_3d_poses)
        write_array_to_csv(self.file_path + 'analysis_dr_3d.csv', dr_3d_poses)
        write_array_to_csv(self.file_path + 'analysis_final_3d.csv', est_3d_poses)
        write_array_to_csv(self.file_path + 'analysis_online_3d.csv', online_3d_poses)

    def save_2d_poses(self):
        """
        Saves three things: camera_gt.csv, camera_dr.csv, camera_est.csv
        format: [[x, y, z, q_w, q_x, q_y, q_z, img seq #]]

        :return:
        """

        if self.file_path is None:
            print("Analysis output path not specified")
            return

        write_array_to_csv(self.file_path + 'analysis_gt.csv', self.gt_poses)
        write_array_to_csv(self.file_path + 'analysis_dr.csv', self.dr_poses)
        write_array_to_csv(self.file_path + 'analysis_est.csv', self.posterior_poses)

        if self.online_poses is not None:
            write_array_to_csv(self.file_path + 'analysis_online.csv', self.online_poses)

    def save_performance_metrics(self):
        """
        Saves three thing: camera_gt.csv, camera_dr.csv, camera_est.csv
        format: [[update time, factor count]]

        :return:
        """

        if self.file_path is None:
            print("Analysis output path not specified")
            return

        if self.performance_metrics is None:
            print("Performance metrics are not defined")
            return

        write_array_to_csv(self.file_path + 'performance_metrics.csv', self.performance_metrics)

    def save_rope_info(self):
        """
        Saves three thing: camera_gt.csv, camera_dr.csv, camera_est.csv
        format: [[update time, factor count]]

        :return:
        """

        if self.file_path is None:
            print("Analysis output path not specified")
            return

        if self.rope_infos is None:
            print("Rope info is not defined")
            return

        write_array_to_csv(self.file_path + 'rope_infos.csv', self.rope_infos)

    # Plotting methods
    def find_plot_limits(self):

        gt_max_x, gt_max_y = np.max(self.gt_poses[:, 0:2], axis=0)
        gt_min_x, gt_min_y = np.min(self.gt_poses[:, 0:2], axis=0)
        dr_max_x, dr_max_y = np.max(self.dr_poses[:, 0:2], axis=0)
        dr_min_x, dr_min_y = np.min(self.dr_poses[:, 0:2], axis=0)
        post_max_x, post_max_y = np.max(self.posterior_poses[:, 0:2], axis=0)
        post_min_x, post_min_y = np.min(self.posterior_poses[:, 0:2], axis=0)

        min_x = (min(dr_min_x, gt_min_x, post_min_x) // self.x_tick) * self.x_tick
        max_x = self.ceiling_division(max(dr_max_x, gt_max_x, post_max_x), self.x_tick) * self.x_tick

        min_y = (min(dr_min_y, gt_min_y, post_min_y) // self.y_tick) * self.y_tick
        max_y = self.ceiling_division(max(dr_max_y, gt_max_y, post_max_y), self.y_tick) * self.y_tick

        self.plot_limits = [min_x, max_x, min_y, max_y]

    def visualize_raw(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(self.figure_width, self.figure_height)
        ax.set_aspect('equal')
        plt.title(f'Raw data')
        plt.axis(self.plot_limits)
        plt.grid(True)

        if self.n_detections > 0:
            ax.scatter(self.detections[:, 0], self.detections[:, 1], color='k', label='Detections')

        ax.scatter(self.gt_poses[:, 0], self.gt_poses[:, 1], color=self.gt_color, label='Ground truth')
        ax.scatter(self.dr_poses[:, 0], self.dr_poses[:, 1], color=self.dr_color, label='Dead reckoning')

        ax.legend(fontsize=self.legend_size)

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-raw.png", dpi=300)

        plt.show()
        return

    def visualize_final(self, plot_gt=True, plot_dr=True, plot_buoy=True,
                        plot_rope_lines=True, plot_rope_detects=True,
                        rope_colors=None):
        """
        Visualize The Posterior
        """
        print("Analysis: visualize_final")

        # Check if Optimization has occurred
        if self.current_estimate is None:
            print('Need to perform optimization before it can be printed!')
            return

        # ===== Matplotlip options =====
        fig, ax = plt.subplots()
        fig.set_size_inches(self.figure_width, self.figure_height)
        ax.set_aspect('equal')
        plt.title(f'Final Estimate', fontsize=self.title_size)
        plt.xlabel('x [m]', fontsize=self.label_size)
        plt.ylabel('y [m]', fontsize=self.label_size)
        plt.axis(self.plot_limits)
        plt.grid(True)

        # ==== Plot ground truth =====
        if plot_gt:
            ax.scatter(self.gt_poses[:, 0],
                       self.gt_poses[:, 1],
                       color=self.gt_color,
                       label='Ground truth')

        # ===== Plot dead reckoning =====
        if plot_dr:
            ax.scatter(self.dr_poses[:, 0],
                       self.dr_poses[:, 1],
                       color=self.dr_color,
                       label='Dead reckoning')

        # ===== Plot buoys w/ cluster colors =====
        if plot_buoy and self.n_buoys > 0:
            # TODO: Improve visualizations for online slam
            # Plot prior and posterior buoy positions for online processing
            if self.buoy2cluster is None:
                buoy_prior_color = self.buoy_color
                buoy_post_color = self.buoy_color

                # Calculate MAE of buoy estimate
                buoy_errors = analyze_slam.calculate_distances(self.buoy_priors[:, :2], self.posterior_buoys[:, :2])
                buoy_rmse = np.sqrt(np.mean(buoy_errors ** 2))

                # Plot buoy priors
                ax.scatter(self.buoy_priors[:, 0],
                           self.buoy_priors[:, 1],
                           color=buoy_prior_color,
                           marker='o',
                           label='Prior buoys')

                # Plot buoy posteriors
                ax.scatter(self.posterior_buoys[:, 0],
                           self.posterior_buoys[:, 1],
                           color=buoy_post_color,
                           marker='+',
                           s=100,
                           label=f'Estimated buoys, RMSE: {buoy_rmse:.2f}')

            # Plot prior and posterior buoy positions for offline processing
            else:
                for ind_buoy in range(self.n_buoys):
                    # buoys can be plotted to show the clustering results
                    if self.buoy2cluster[ind_buoy] == -1:
                        current_color = 'k'
                    else:
                        cluster_num = self.buoy2cluster[ind_buoy]
                        current_color = self.colors[cluster_num % len(self.colors)]

                    # Plot buoy priors
                    ax.scatter(self.buoy_priors[ind_buoy, 0],
                               self.buoy_priors[ind_buoy, 1],
                               color=current_color)

                    # Plot buoy posteriors
                    ax.scatter(self.posterior_buoys[ind_buoy, 0],
                               self.posterior_buoys[ind_buoy, 1],
                               color=current_color,
                               marker='+',
                               s=100)

        # ===== Plot ropes =====
        if self.rope_buoy_ind is not None and len(self.rope_buoy_ind) > 0 and plot_rope_lines:
            for rope_ind, rope in enumerate(self.rope_buoy_ind):
                if len(rope) != 2:
                    continue
                rope_start_ind = int(rope[0])
                rope_end_ind = int(rope[1])

                x1, y1 = self.posterior_buoys[rope_start_ind, :2]
                x2, y2 = self.posterior_buoys[rope_end_ind, :2]

                if rope_colors is None:
                    current_color = self.rope_color
                else:
                    current_color = np.array(rope_colors[rope_ind])

                ax.plot([x1, x2], [y1, y2], color=current_color)

        # Plot the posterior poses
        ax.scatter(self.posterior_poses[:, 0], self.posterior_poses[:, 1], color='g', label='Final estimated poses')

        if plot_rope_detects and self.n_rope_detects > 0:
            # form color array
            normal_rope_color = [0.5, 0.5, 0.5]
            if rope_colors is None:
                ax.scatter(self.posterior_rope_detects[:, 0], self.posterior_rope_detects[:, 1],
                           color=normal_rope_color,
                           label='Rope detections')
            else:
                # form detection colors
                colors_array = np.zeros((self.n_rope_detects, 3))
                for i in range(self.n_rope_detects):
                    if 0 <= self.posterior_rope_associations[i] < len(rope_colors):
                        colors_array[i, :] = rope_colors[int(self.posterior_rope_associations[i])]
                    else:
                        colors_array[i, :] = normal_rope_color

                ax.scatter(self.posterior_rope_detects[:, 0], self.posterior_rope_detects[:, 1],
                           color=colors_array,
                           label='Rope detections')

        if self.corresponding_detections is not None:
            rope_errors = analyze_slam.calculate_distances(self.corresponding_detections[:, :2],
                                                           self.posterior_rope_detects[:, :2])

            rope_rmse = np.sqrt(np.mean(rope_errors ** 2))

            label_flag = True  # Only want to add a single label for all correspondence lines
            for start, end in zip(self.corresponding_detections, self.posterior_rope_detects):
                x_vals = [start[0], end[0]]
                y_vals = [start[1], end[1]]
                if label_flag:  # only label the first line segment
                    ax.plot(x_vals, y_vals, color='orange', label=f'Rope detection error, RMSE: {rope_rmse:.2f}')
                    label_flag = False
                else:
                    ax.plot(x_vals, y_vals, color='orange')

        ax.legend(fontsize=self.legend_size)

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-final.png", dpi=300)

        plt.show()

    def visualize_online(self, plot_dr=False, plot_final=False, plot_buoy=True, plot_correspondence=False):
        """
        Visualize the online estimate compared to the final estimate
        """

        print("Analysis: visualize_online")

        # Check if Optimization has occurred
        if self.current_estimate is None:
            print('Need to perform optimization before it can be printed!')
            return

        # ===== Matplotlip options =====
        fig, ax = plt.subplots()
        fig.set_size_inches(self.figure_width, self.figure_height)
        ax.set_aspect('equal')
        plt.title(f'Online Estimate', fontsize=self.title_size)
        plt.xlabel('x [m]', fontsize=self.label_size)
        plt.ylabel('y [m]', fontsize=self.label_size)
        plt.axis(self.plot_limits)
        plt.grid(True)

        # ===== Plot dead reckoning =====
        if plot_dr:
            ax.scatter(self.dr_poses[:, 0],
                       self.dr_poses[:, 1],
                       color=self.dr_color,
                       label='Dead reckoning')

        # ===== Plot correspondence =====
        # This will plot a line between the online and final estimates
        if plot_correspondence:
            for start, end in zip(self.online_poses[:, :2], self.posterior_poses[:, :2]):
                x_vals = [start[0], end[0]]
                y_vals = [start[1], end[1]]
                ax.plot(x_vals, y_vals, color='orange')

        # ===== Plot the final estimated poses =====
        if plot_final:
            ax.scatter(self.posterior_poses[:, 0], self.posterior_poses[:, 1],
                       color=self.post_color, label='Final estimate')

        # ===== Plot the final estimated poses =====
        ax.scatter(self.online_poses[:, 0], self.online_poses[:, 1],
                   color=self.online_color, label='Online estimate')

        # ===== Plot buoys w/ cluster colors =====
        if plot_buoy and self.n_buoys > 0:
            # TODO: Improve visualizations for online slam
            # Plot prior and posterior buoy positions for online processing
            if self.buoy2cluster is None:
                buoy_prior_color = 'k'
                buoy_post_color = self.post_color

                # Plot buoy priors
                ax.scatter(self.buoy_priors[:, 0],
                           self.buoy_priors[:, 1],
                           color=buoy_prior_color,
                           label='Prior buoys')

                # Plot buoy posteriors
                ax.scatter(self.posterior_buoys[:, 0],
                           self.posterior_buoys[:, 1],
                           color=buoy_post_color,
                           marker='+',
                           s=75,
                           label='Estimated buoys')

            # Plot prior and posterior buoy positions for offline processing
            else:
                for ind_buoy in range(self.n_buoys):
                    # buoys can be plotted to show the clustering results
                    if self.buoy2cluster[ind_buoy] == -1:
                        current_color = 'k'
                    else:
                        cluster_num = self.buoy2cluster[ind_buoy]
                        current_color = self.colors[cluster_num % len(self.colors)]

                    # Plot buoy priors
                    ax.scatter(self.buoy_priors[ind_buoy, 0],
                               self.buoy_priors[ind_buoy, 1],
                               color=current_color)

                    # Plot buoy posteriors
                    ax.scatter(self.posterior_buoys[ind_buoy, 0],
                               self.posterior_buoys[ind_buoy, 1],
                               color=current_color,
                               marker='+',
                               s=75)

        # ===== Plot ropes =====
        # see visualizing_posterior() for example

        # ===== Plot rope detections =====
        # see visualizing_posterior() for example

        ax.legend(fontsize=self.legend_size)

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-online.png", dpi=300)

        plt.show()

    def plot_error_positions(self, gt_baseline=False, plot_dr=True, plot_online=True, plot_final=True):
        """
        Plots the error between the dr and baseline and the online estimate and the baseline.
        The baseline is set by the gt_baseline argument.
        - dr and final or gt
        - online and final or gt

        :param plot_final:
        :param plot_online:
        :param plot_dr:
        :param gt_baseline: Bool, true specifies that the gt will be used
        :return:
        """

        if gt_baseline:
            print("Analysis: plot_error_positions - Ground truth baseline")
            dr_error = analyze_slam.calculate_distances(self.dr_poses[:, :2], self.gt_poses[:, :2])
            online_error = analyze_slam.calculate_distances(self.online_poses[:, :2], self.gt_poses[:, :2])
            final_error = analyze_slam.calculate_distances(self.online_poses[:, :2], self.gt_poses[:, :2])

            dr_rmse = np.sqrt(np.mean(dr_error ** 2))
            online_rmse = np.sqrt(np.mean(online_error ** 2))
            final_rmse = np.sqrt(np.mean(final_error ** 2))

        else:
            print("Analysis: plot_error_positions - Posterior baseline")
            dr_error = analyze_slam.calculate_distances(self.dr_poses[:, :2], self.posterior_poses[:, :2])
            online_error = analyze_slam.calculate_distances(self.online_poses[:, :2], self.posterior_poses[:, :2])

            dr_rmse = np.sqrt(np.mean(dr_error ** 2))
            online_rmse = np.sqrt(np.mean(online_error ** 2))

        # Create a plot
        plt.figure()

        # Plot squared error
        if plot_dr:
            plt.plot(dr_error, label=f'DR error, RMSE: {dr_rmse:.2f}', color=self.dr_color)
        if plot_online:
            plt.plot(online_error, label=f'Online error, RMSE: {online_rmse:.2f}', color=self.online_color)
        if gt_baseline and plot_final:
            plt.plot(final_error, label=f'final error, RMSE: {final_rmse:.2f}', color=self.post_color)

        # Add labels and title
        plt.xlabel('Poses', fontsize=self.label_size)
        plt.ylabel('Error [m]', fontsize=self.label_size)
        plt.title('Error Comparison', fontsize=self.title_size)
        plt.legend(fontsize=self.legend_size)
        plt.grid(True)

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-position-error.png", dpi=300)

        # Show the plot
        plt.show()

        if gt_baseline:
            data = np.vstack((dr_error, online_error, final_error))
        else:
            data = np.vstack((dr_error, online_error))
        np.savetxt(self.file_path + 'dr_online_error.csv', data, delimiter=',')

    def show_graph_2d(self, label, show_final=True, show_dr=True):
        """

        """
        # Check that the graph is has initial and estimated values
        if self.current_estimate is None:
            print('Perform optimization before it can be graphed')
            return
        if self.initial_estimate is None:
            print('Initialize estimate before it can be graphed')
            return

        # Select which values to graph
        if show_final:
            values = self.current_estimate
        else:
            values = self.initial_estimate

        # ===== Unpack the factor graph using networkx =====
        # Initialize network
        G = nx.Graph()

        # Add the raw DR poses to the graph
        if show_dr:
            for dr_i, dr_pose in enumerate(self.dr_poses):
                pos = (dr_pose[0], dr_pose[1])
                G.add_node(dr_i, pos=pos, color='red')

        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            for key_id, key in enumerate(factor.keys()):
                # Test if key corresponds to a pose
                if key in self.x.values():
                    pos = (values.atPose2(key).x(), values.atPose2(key).y())
                    G.add_node(key, pos=pos, color='green')

                # keys of nodes corresponding to buoy detections
                elif self.b is not None and key in self.b.values():
                    pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])

                    # Set color according to clustering
                    if self.buoy2cluster is None:
                        node_color = 'black'
                    else:
                        # Find the buoy index -> cluster index -> cluster color
                        buoy_id = list(self.b.values()).index(key)
                        cluster_id = self.buoy2cluster[buoy_id]
                        # A negative cluster id indicates that the buoy was not assigned a cluster
                        if cluster_id < 0:
                            node_color = 'black'
                        else:
                            node_color = self.colors[cluster_id % len(self.colors)]
                    G.add_node(key, pos=pos, color=node_color)

                # keys of nodes corresponding to rope detections
                elif self.r is not None and key in self.r.values():
                    pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])
                    node_color = 'gray'
                    G.add_node(key, pos=pos, color=node_color)

                else:
                    print('There was a problem with a factor not corresponding to an available key')

                # Add edges that represent binary factor: Odometry or detection
                # This does not plot the edges that involve a rope detection node
                for key_2_id, key_2 in enumerate(factor.keys()):
                    if key != key_2 and key_id < key_2_id:
                        # detections will have key corresponding to a landmark
                        if self.b is not None and (key in self.b.values() or key_2 in self.b.values()):
                            G.add_edge(key, key_2, color='blue')
                        elif self.r is not None and (key not in self.r.values() and key_2 not in self.r.values()):
                            G.add_edge(key, key_2, color='red')

        # ===== Plot the graph using matplotlib =====
        # Matplotlib options
        fig_x_ticks = np.arange(self.plot_limits[0], self.plot_limits[1] + 1, self.x_tick)
        fig_y_ticks = np.arange(self.plot_limits[2], self.plot_limits[3] + 1, self.y_tick)
        fig, ax = plt.subplots()
        plt.title(f'Factor Graph\n{label}\n')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)

        # Networkx Options
        pos = nx.get_node_attributes(G, 'pos')
        e_colors = nx.get_edge_attributes(G, 'color').values()
        n_colors = nx.get_node_attributes(G, 'color').values()
        options = {'node_size': 25, 'width': 3, 'with_labels': False}

        # Plot
        nx.draw_networkx(G, pos, edge_color=e_colors, node_color=n_colors, **options)

        # Wasn't plotting
        plt.xticks(fig_x_ticks)
        plt.yticks(fig_y_ticks)

        plt.xlabel(fig_x_ticks)
        plt.ylabel(fig_y_ticks)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.show()

    def visualize_clustering(self):
        """
        !OFFLINE SLAM ONLY!

        This method is for visualizing the clustering of detections. This is only relevant for the offline SLAM.
        :return:
        """
        # ===== Plot detected clusters =====
        fig, ax = plt.subplots()
        plt.title(f'Clusters\n{self.n_clusters} Detected')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)

        for cluster in range(self.n_clusters):
            inds = self.detection_clusterings == cluster
            ax.scatter(self.detections_graph[inds, 0],
                       self.detections_graph[inds, 1],
                       color=self.colors[cluster % len(self.colors)])

        plt.show()

        # ===== Plot true buoy locations w/ cluster means ====
        fig, ax = plt.subplots()
        plt.title('Buoys\nTrue buoy positions and associations\ncluster means')
        ax.set_aspect('equal', 'box')
        plt.axis(self.plot_limits)
        plt.grid(True)

        for ind_buoy in range(self.n_buoys):
            cluster_num = self.buoy2cluster[ind_buoy]  # landmark_associations[ind_landmark]
            if cluster_num == -1:
                current_color = 'k'
            else:
                current_color = self.colors[cluster_num % len(self.colors)]
            # not all buoys have an associated have an associated cluster
            if cluster_num >= 0:
                ax.scatter(self.cluster_model.means_[cluster_num, 0],
                           self.cluster_model.means_[cluster_num, 1],
                           color=current_color,
                           marker='+',
                           s=75)

            ax.scatter(self.buoy_priors[ind_buoy, 0],
                       self.buoy_priors[ind_buoy, 1],
                       color=current_color)

        plt.show()
        return

    # Error metric methods
    def show_error(self):
        """
        !CURRENTLY BROKEN!
        Plots the errors for x,y and yaw with the ground truth as a baseline
        :return:
        """

        # TODO make a nice little error graph function for x, y, yaw

        # Find the errors between gt<->dr and gt<->post
        dr_error = calc_pose_error(self.dr_poses, self.gt_poses)
        post_error = calc_pose_error(self.posterior_poses, self.gt_poses)

        # Calculate MSE
        dr_rmse_error = np.sqrt(np.sum(np.square(dr_error), axis=0))
        post_rmse_error = np.sqrt(np.sum(np.square(post_error), axis=0))

        # ===== Plot =====
        fig, (ax_x, ax_y, ax_t) = plt.subplots(1, 3)
        # X error
        ax_x.plot(dr_error[:, 0], self.dr_color, label='Dead reckoning')
        ax_x.plot(post_error[:, 0], self.post_color, label='Posterior')
        ax_x.title.set_text(f'X Error\nD.R. RMSE: {dr_rmse_error[0]:.4f}\n Posterior RMSE: {post_rmse_error[0]:.4f}')
        ax_x.legend()
        # Y error
        ax_y.plot(dr_error[:, 1], self.dr_color, label='Dead reckoning')
        ax_y.plot(post_error[:, 1], self.post_color, label='Posterior')
        ax_y.title.set_text(f'Y Error\nD.R. RMSE: {dr_rmse_error[1]:.4f}\n Posterior RMSE: {post_rmse_error[1]:.4f}')
        ax_y.legend()
        # Theta error
        ax_t.plot(dr_error[:, 2], self.dr_color, label='Dead reckoning')
        ax_t.plot(post_error[:, 2], self.post_color, label='Posterior')
        ax_t.title.set_text(f'Theta Error\nD.R. RMSE: {dr_rmse_error[2]:.4f}\n Posterior RMSE: {post_rmse_error[2]:.4f}')
        ax_t.legend()

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-pose-error.png", dpi=300)

        plt.show()

    def show_buoy_info(self):
        """
        Plots associations and distances of buoy detections
        :return:
        """
        if self.buoy_detection_info is None:
            return

        if self.buoy_detection_info.size == 0:
            print("Buoy detection info unpopulated!")
            return

        associations = self.buoy_detection_info[:, 0]
        e_distances = self.buoy_detection_info[:, 1]
        m_distances = self.buoy_detection_info[:, 2]

        true_associations = np.array((3, -1, 2,
                                      0, 5,
                                      4, 1,
                                      4,
                                      3, -1, 2))

        print(f"Associations: {associations}")
        print(f"True associations: {true_associations}")

        # ===== Plot =====
        fig, (ax_da, ax_e_dist, ax_m_dist) = plt.subplots(1, 3)
        fig.set_size_inches(self.figure_width, self.figure_height)
        # Data associations
        ax_da.plot(associations)
        ax_da.plot(true_associations, color='green', label='Truth')
        ax_da.set_title('Data associations')
        ax_da.legend()

        # Euclidean distances
        ax_e_dist.set_title('Euclidean distances')
        ax_e_dist.plot(e_distances, label='Euclidean distances')
        # plot threshold
        if self.da_distance_threshold > 0:
            ax_e_dist.axhline(self.da_distance_threshold, color='red', linestyle='--', linewidth=1, label='Threshold')

        # Mahalanobis distances
        ax_m_dist.set_title('Mahalanobis distances')
        ax_m_dist.plot(m_distances, label='Mahalanobis distances')
        # plot threshold
        if self.da_m_distance_threshold > 0:
            ax_m_dist.axhline(self.da_m_distance_threshold, color='red', linestyle='--', linewidth=1, label='Threshold')

        if self.file_path is not None:
            plt.savefig(self.file_path + "fig-info.png", dpi=300)

        plt.show()

    # def print_residuals(self):
    #     # Print residuals
    #     # Print residuals
    #     return
    #     for factor_key in self.r.values():
    #         factor_key_to_access = gtsam.Key(factor_key)
    #         factor = self.graph.at(factor_key_to_access)
    #         factor_residual = factor.error(self.current_estimate)
    #         print("Factor Key:", factor_key)
    #         print("At: ", factor_key_to_access)
    #         print("Factor:", factor)
    #         print("Residual:", factor_residual)

    def calculate_corresponding_points(self, debug=False):
        """
        This method finds the closest point of each rope detection to linearly fit rope, the estimated buoy positions
        are used to define the rope line segments.

        :return:
        """

        if self.n_rope_detects == 0 or self.n_ropes == 0:
            print("Unable to calc corresponding points")
            return

        self.corresponding_detections = np.zeros((self.n_rope_detects, 2))
        self.corresponding_distances = np.zeros(self.n_rope_detects)

        for detect_i in range(self.n_rope_detects):
            detection = [self.posterior_rope_detects[detect_i, 0], self.posterior_rope_detects[detect_i, 1]]
            point = None
            distance = np.inf
            for rope in self.rope_buoy_ind:
                if len(rope) != 2:
                    continue
                rope_start_ind = int(rope[0])
                rope_end_ind = int(rope[1])

                x1, y1 = self.posterior_buoys[rope_start_ind, :2]
                x2, y2 = self.posterior_buoys[rope_end_ind, :2]

                cur_point, cur_dist = self.closest_point_distance_to_line_segment([x1, y1], [x2, y2], detection)

                if cur_dist < distance:
                    distance = cur_dist
                    point = cur_point

            if point is not None:
                self.corresponding_detections[detect_i, :] = point
                self.corresponding_distances[detect_i] = distance

        if debug:
            print("DEBUG: calculate_corresponding_points")
            print(self.corresponding_detections)

    # Utility static methods
    @staticmethod
    def ceiling_division(n, d):
        return -(n // -d)

    @staticmethod
    def closest_point_distance_to_line_segment(A, B, P):
        """
        Calculates the closest point, Q, on a line, defined by A and B, with point P.
        Also returns the the distance between P and Q.

        :param A: [x, y] start of line segment
        :param B: [x, y] end of line segment
        :param P: [x, y] point of interest
        :return: [Qx, Qy], distance
        """
        ABx = B[0] - A[0]
        ABy = B[1] - A[1]
        APx = P[0] - A[0]
        APy = P[1] - A[1]

        dot_product = ABx * APx + ABy * APy
        length_squared_AB = ABx * ABx + ABy * ABy

        t = dot_product / length_squared_AB

        if t < 0:
            Qx, Qy = A[0], A[1]
        elif t > 1:
            Qx, Qy = B[0], B[1]
        else:
            Qx = A[0] + t * ABx
            Qy = A[1] + t * ABy

        QPx = P[0] - Qx
        QPy = P[1] - Qy

        distance = math.sqrt(QPx * QPx + QPy * QPy)

        return [Qx, Qy], distance

    @staticmethod
    def calculate_distances(array1, array2):
        # Check if arrays have the same number of points
        if array1.shape != array2.shape:
            print("Size mismatch between the arrays.")
            return None

        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((array1 - array2) ** 2, axis=1))

        return distances
