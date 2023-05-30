#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation
"""

# %% Imports
from __future__ import annotations
import itertools

# Maths
import numpy as np
import matplotlib.pyplot as plt

# Clustering
from sklearn.mixture import GaussianMixture

# Graphing graphs
import networkx as nx

# Slam
import gtsam

# ROS
import rospy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker
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
    def __init__(self, path_name=None):
        graph = gtsam.NonlinearFactorGraph()
        self.manual_associations = rospy.get_param("manual_associations", False)

        print(f"Manual associations: {self.manual_associations}")

        # ===== File path for data logging =====
        self.file_path = path_name
        if self.file_path is None or not isinstance(path_name, str):
            self.file_path = ''
        else:
            self.file_path = path_name + '/'

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

        # === Sensors and detections
        self.bearings_ranges = []  # Not used for online
        self.sensor_string_at_key = {}  # camera and sss data is associated with graph nodes here

        # ===== Sigmas =====
        # Agent prior sigmas
        self.prior_ang_sig = rospy.get_param('prior_ang_sig_deg', 1.0) * np.pi / 180
        self.prior_dist_sig = rospy.get_param('prior_dist_sig', 1.0)
        # buoy prior sigmas
        self.buoy_dist_sig_init = rospy.get_param('buoy_dist_sig', 1.0)
        # buoy prior sigmas
        self.rope_dist_sig_init = rospy.get_param('rope_dist_sig', 15.0)
        # agent odometry sigmas
        self.odo_ang_sig = rospy.get_param('odo_ang_sig_deg', 0.1) * np.pi / 180
        self.odo_dist_sig = rospy.get_param('dist_sig', 0.1)
        # detection sigmas
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

        self.detection_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.detect_dist_sig,
                                                                          self.detect_ang_sig]))

        # ===== buoy prior map =====
        self.n_buoys = None
        self.buoy_priors = None
        self.buoy_average = None

        # ===== Optimizer and values =====
        # self.optimizer = None
        self.initial_estimate = gtsam.Values()
        self.current_estimate = None
        self.slam_result = None

        # ===== Graph states =====
        self.buoy_map_present = False
        self.initial_pose_set = False
        self.busy = False

        # ===== Debugging =====
        self.da_check = {}
        self.est_detect_loc = None
        self.true_detect_loc = None

        # Rviz publishers
        self.est_detect_loc_pub = rospy.Publisher('/sam_slam/est_detection_positions', Marker, queue_size=10)
        self.da_pub = rospy.Publisher('/sam_slam/da_positions', Marker, queue_size=10)
        self.est_pos_pub = rospy.Publisher('/sam_slam/est_positions', Marker, queue_size=10)
        self.marker_scale = 1.0
        self.est_marker_x, self.est_marker_y, self.est_marker_z = 3, .5, .5
        self.est_path_pub = rospy.Publisher('/sam_slam/est_path', Path, queue_size=10)

        # ===== Verboseness parameters =====
        self.verbose_graph_update = rospy.get_param('verbose_graph_update', False)
        self.verbose_graph_detections = rospy.get_param('verbose_graph_detections', False)

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
        print("Done with  buoy setup")
        return

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

        self.initial_pose_set = True
        print("Done with first pose - x0")
        if self.x is None:
            print("problem")
        return

    def online_update(self, dr_pose, gt_pose, relative_detection=None, id_string=None, da_id=None):
        """
        Pose format [x, y, z, q_w, q_x, q_y, q_z]
        """
        if not self.initial_pose_set:
            print("Attempting to update before initial pose")
            self.add_first_pose(dr_pose=dr_pose, gt_pose=gt_pose, id_string=id_string)
            return

        # Attempt at preventing saturation
        self.busy = True

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

        # ===== Add the between factor =====
        self.graph.add(gtsam.BetweenFactorPose2(self.x[self.current_x_ind - 1],
                                                self.x[self.current_x_ind],
                                                between_odometry,
                                                self.odometry_model))

        # Compute initialization value from the current estimate and odometry
        computed_est = self.current_estimate.atPose2(self.x[self.current_x_ind - 1]).compose(between_odometry)

        # Update initial estimate
        self.initial_estimate.insert(self.x[self.current_x_ind], computed_est)

        # ===== Process detection =====
        # === Buoy ===
        # TODO this might need to be more robust, not assume detections will lead to graph update
        if relative_detection is not None and da_id != -ObjectID.ROPE.value:
            # Calculate the map location of the detection given relative measurements and current estimate
            self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
            detect_bearing = computed_est.bearing(self.est_detect_loc)
            detect_range = computed_est.range(self.est_detect_loc)

            if self.manual_associations and da_id != -ObjectID.BUOY.value:
                buoy_association_id = da_id
            else:
                buoy_association_id, buoy_association_dist = self.associate_detection(self.est_detect_loc)

                # ===== DA debugging =====
                # Apply relative detection to gt to find the true DA
                self.true_detect_loc = self.gt_Pose2s[-1].transformFrom(np.array(relative_detection, dtype=np.float64))
                true_association_id, true_association_dist = self.associate_detection(self.true_detect_loc)

                if buoy_association_id == true_association_id:
                    self.da_check[self.current_x_ind] = [True,
                                                         buoy_association_id, true_association_id,
                                                         buoy_association_dist, true_association_dist]
                else:
                    self.da_check[self.current_x_ind] = [False,
                                                         buoy_association_id, true_association_id,
                                                         buoy_association_dist, true_association_dist]

            if self.verbose_graph_detections and relative_detection is not None:
                print(
                    f'Detection - range: {detect_range}  bearing: {detect_bearing.theta()}  DA: {buoy_association_id}')

            # ===== Add detection to graph =====
            self.graph.add(gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
                                                      self.b[buoy_association_id],
                                                      detect_bearing,
                                                      detect_range,
                                                      self.detection_model))

        # === Rope ===
        if relative_detection is not None and da_id == -ObjectID.ROPE.value:
            # Calculate the map location of the detection given relative measurements and current estimate
            self.est_detect_loc = computed_est.transformFrom(np.array(relative_detection, dtype=np.float64))
            detect_bearing = computed_est.bearing(self.est_detect_loc)
            detect_range = computed_est.range(self.est_detect_loc)

            if self.verbose_graph_detections:
                print(f'Rope detection - range: {detect_range}  bearing: {detect_bearing.theta()}')

            # ===== Add detection to graph =====
            # Add the rope landmark
            self.current_r_ind += 1
            self.r[self.current_r_ind] = gtsam.symbol('r', self.current_r_ind)

            # Add new landmarks prior and noise
            rope_prior = np.array((self.buoy_average[0], self.buoy_average[1]), dtype=np.float64)
            self.graph.addPriorPoint2(self.r[self.current_r_ind], rope_prior, self.prior_model_rope)

            # Initial estimate
            self.initial_estimate.insert(self.r[self.current_r_ind], self.est_detect_loc)

            # Add factor between current x and current r

            self.graph.add(gtsam.BearingRangeFactor2D(self.x[self.current_x_ind],
                                                      self.r[self.current_r_ind],
                                                      detect_bearing,
                                                      detect_range,
                                                      self.detection_model))

        # Time update process
        start_time = rospy.Time.now()

        # Incremental update
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()

        # self.graph.resize(0)
        self.initial_estimate.clear()

        end_time = rospy.Time.now()
        update_time = (end_time - start_time).to_sec()

        # Release the graph
        self.busy = False

        # ===== debugging and visualizations =====
        # Log to terminal
        if self.verbose_graph_update:
            print(f"Done with update - x{self.current_x_ind}: {update_time} s")

        # === Debug publishing ===
        self.publish_estimated_pos_marker_and_transform(computed_est)
        self.publish_est_path()
        # Buoy detection
        if relative_detection is not None and da_id != -ObjectID.ROPE.value:
            self.publish_detection_markers(buoy_association_id)

        if self.x is None:
            print("problem")

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

    def publish_detection_markers(self, da_id):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """

        # Estimated detection location
        detection_marker = Marker()
        detection_marker.header.frame_id = 'map'
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
        detection_marker.color.r = 1.0
        detection_marker.color.g = 0.0
        detection_marker.color.b = 1.0
        detection_marker.color.a = 1.0

        # Data association marker
        da_marker = Marker()
        da_marker.header.frame_id = 'map'
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

        self.est_detect_loc_pub.publish(detection_marker)
        self.da_pub.publish(da_marker)

    def publish_estimated_pos_marker_and_transform(self, est_pos):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """
        heading_quat = quaternion_from_euler(0, 0, est_pos.theta())
        heading_quaternion_type = Quaternion(*heading_quat)

        # Estimated detection location
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = 0
        marker.lifetime = rospy.Duration(0)
        marker.pose.position.x = est_pos.x()
        marker.pose.position.y = est_pos.y()
        marker.pose.position.z = 0.0
        marker.pose.orientation = heading_quaternion_type
        marker.scale.x = self.est_marker_x
        marker.scale.y = self.est_marker_y
        marker.scale.z = self.est_marker_z
        marker.color.r = 0.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.est_pos_pub.publish(marker)

        # publish transform of estimate
        br = tf.TransformBroadcaster()
        try:
            br.sendTransform((est_pos.x(), est_pos.y(), 0),
                             (heading_quaternion_type.x,
                              heading_quaternion_type.y,
                              heading_quaternion_type.z,
                              heading_quaternion_type.w),
                             rospy.Time.now(),
                             "estimated/base_link",
                             "map")
        except rospy.ROSException as e:
            rospy.logerr('Error broadcasting tf transform: {}'.format(str(e)))

    def publish_est_path(self):

        poses_path = Path()
        poses_path.header.frame_id = 'map'

        for key in self.x.values():

            pose = self.current_estimate.atPose2(key)
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = pose.x()
            pose_stamped.pose.position.y = pose.y()

            heading_quat = quaternion_from_euler(0, 0, pose.theta())
            heading_quaternion_type = Quaternion(*heading_quat)
            pose_stamped.pose.orientation = heading_quaternion_type

            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = rospy.Time.now()

            poses_path.poses.append(pose_stamped)

        self.est_path_pub.publish(poses_path)


class analyze_slam:
    """
    Responsible for analysis of slam results
    # TODO Need to think about how to unify the online and offline classes
    slam_object.graph: gtsam.NonlinearFactorGraph
    slam_object.dr_pose2s: list[gtsam.Pose2]
    slam_object.gt_pose2s: list[gtsam.Pose2]
    """

    def __init__(self, slam_object: offline_slam_2d | online_slam_2d):
        # unpack slam object
        self.slam = slam_object
        self.graph = slam_object.graph
        self.current_estimate = slam_object.current_estimate
        self.x = slam_object.x  # pose keys
        self.b = slam_object.b  # point keys
        self.r = slam_object.r  # rope keys (not found in offline processing)

        # Dead reckoning poses and the between poses, ground truth poses
        self.dr_poses = pose2_list_to_nparray(slam_object.dr_Pose2s)
        self.gt_poses = pose2_list_to_nparray(slam_object.gt_Pose2s)
        self.between_Pose2s = pose2_list_to_nparray(slam_object.between_Pose2s)

        # ===== Buoys =====
        self.buoy_priors = slam_object.buoy_priors
        if self.buoy_priors is None:
            self.n_buoys = 0
        else:
            self.n_buoys = len(self.buoy_priors)

        # ===== Build arrays for the poses and points of the posterior =====
        self.posterior_poses = np.zeros((len(self.x), 3))
        for i in range(len(self.x)):
            self.posterior_poses[i, 0] = self.current_estimate.atPose2(self.x[i]).x()
            self.posterior_poses[i, 1] = self.current_estimate.atPose2(self.x[i]).y()
            self.posterior_poses[i, 2] = self.current_estimate.atPose2(self.x[i]).theta()

        if self.n_buoys > 0:
            self.posterior_points = np.zeros((self.n_buoys, 2))
            for i in range(self.n_buoys):
                self.posterior_points[i, 0] = self.current_estimate.atPoint2(self.b[i])[0]
                self.posterior_points[i, 1] = self.current_estimate.atPoint2(self.b[i])[1]

        # ===== Unpack more relevant data from the slam object =====
        """
        Current differences include:
        detection_graph: list of detections with id to relate the to dr indices, only found in offline version
        buoy2cluster: mapping from buoy to cluster, used for plotting offline buoys and clustering
        initial_estimate: The dr poses serve as the initial estimate for the offline version
        da_check: The online version uses the ground truth and the relative detection data to find the DA ground truth
        """
        # TODO Unify the online and offline versions of the analysis

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

        # ===== Visualization parameters =====
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.colors = ['orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        # Set plot limits
        self.x_tick = 5
        self.y_tick = 5
        self.plot_limits = None
        self.find_plot_limits()

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

    @staticmethod
    def ceiling_division(n, d):
        return -(n // -d)

    def visualize_raw(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.title(f'Raw data')
        plt.axis(self.plot_limits)
        plt.grid(True)

        if self.n_detections > 0:
            ax.scatter(self.detections[:, 0], self.detections[:, 1], color='k', label='Detections')

        ax.scatter(self.gt_poses[:, 0], self.gt_poses[:, 1], color=self.gt_color, label='Ground truth')
        ax.scatter(self.dr_poses[:, 0], self.dr_poses[:, 1], color=self.dr_color, label='Dead reckoning')

        ax.legend()
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

        # ===== Matplotlip options =====
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.title(f'Posterior')
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
            for ind_buoy in range(self.n_buoys):
                # TODO: Improve visualizations for online slam
                # Plot prior and posterior buoy positions for online
                # These buoys are not clustered
                if self.buoy2cluster is None:
                    buoy_prior_color = 'k'
                    buoy_post_color = self.post_color

                    # Plot buoy priors
                    ax.scatter(self.buoy_priors[ind_buoy, 0],
                               self.buoy_priors[ind_buoy, 1],
                               color=buoy_prior_color)

                    # Plot buoy posteriors
                    ax.scatter(self.posterior_points[ind_buoy, 0],
                               self.posterior_points[ind_buoy, 1],
                               color=buoy_post_color,
                               marker='+',
                               s=75)

                # Plot prior and posterior buoy positions for offline processing
                # buoys can be plotted to show the clustering results
                else:
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
                    ax.scatter(self.posterior_points[ind_buoy, 0],
                               self.posterior_points[ind_buoy, 1],
                               color=current_color,
                               marker='+',
                               s=75)

        # Plot the posterior
        ax.scatter(self.posterior_poses[:, 0], self.posterior_poses[:, 1], color='g', label='Posterior')

        ax.legend()
        plt.show()

    def show_error(self):
        # Find the errors between gt<->dr and gt<->post
        dr_error = calc_pose_error(self.dr_poses, self.gt_poses)
        post_error = calc_pose_error(self.posterior_poses, self.gt_poses)

        # Calculate MSE
        dr_mse_error = np.square(dr_error).mean(0)
        post_mse_error = np.square(post_error).mean(0)

        # ===== Plot =====
        fig, (ax_x, ax_y, ax_t) = plt.subplots(1, 3)
        # X error
        ax_x.plot(dr_error[:, 0], self.dr_color, label='Dead reckoning')
        ax_x.plot(post_error[:, 0], self.post_color, label='Posterior')
        ax_x.title.set_text(f'X Error\nD.R. MSE: {dr_mse_error[0]:.4f}\n Posterior MSE: {post_mse_error[0]:.4f}')
        ax_x.legend()
        # Y error
        ax_y.plot(dr_error[:, 1], self.dr_color, label='Dead reckoning')
        ax_y.plot(post_error[:, 1], self.post_color, label='Posterior')
        ax_y.title.set_text(f'Y Error\nD.R. MSE: {dr_mse_error[1]:.4f}\n Posterior MSE: {post_mse_error[1]:.4f}')
        ax_y.legend()
        # Theta error
        ax_t.plot(dr_error[:, 2], self.dr_color, label='Dead reckoning')
        ax_t.plot(post_error[:, 2], self.post_color, label='Posterior')
        ax_t.title.set_text(f'Theta Error\nD.R. MSE: {dr_mse_error[2]:.4f}\n Posterior MSE: {post_mse_error[2]:.4f}')
        ax_t.legend()

        plt.show()

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

    def save_for_sensor_processing(self, file_path=''):
        """
        Saves three files related to the camera: camera_gt.csv, camera_dr.csv, camera_est.csv
        Saves three files related to the sss: sss_gt.csv, sss_dr.csv, sss_est.csv
        saves a file of the estimated buoy positions
        format: [[x, y, z, q_w, q_x, q_y, q_z, img seq #]]

        :return:
        """

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

            quats = quaternion_from_euler(roll, pitch, est_yaw)

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
            write_array_to_csv(file_path + 'camera_gt.csv', camera_gt)
            write_array_to_csv(file_path + 'camera_dr.csv', camera_dr)
            write_array_to_csv(file_path + 'camera_est.csv', camera_est)

        if len(sss_gt) > 0:
            write_array_to_csv(file_path + 'sss_gt.csv', sss_gt)
            write_array_to_csv(file_path + 'sss_dr.csv', sss_dr)
            write_array_to_csv(file_path + 'sss_est.csv', sss_est)

        # ===== Save buoy estimated positions =====
        # only the x an y coords are estimated, buoys are assumed to have z = 0
        if self.n_buoys > 0:
            buoys_est = np.zeros((self.n_buoys, 3))

            for i in range(self.n_buoys):
                buoys_est[i, 0] = self.current_estimate.atPoint2(self.b[i])[0]
                buoys_est[i, 1] = self.current_estimate.atPoint2(self.b[i])[1]

            # Write to file
            write_array_to_csv(file_path + 'buoys_est.csv', buoys_est)

    def save_2d_poses(self, file_path=''):
        """
        Saves three thing: camera_gt.csv, camera_dr.csv, camera_est.csv
        format: [[x, y, z, q_w, q_x, q_y, q_z, img seq #]]

        :return:
        """
        write_array_to_csv(file_path + 'analysis_gt.csv', self.gt_poses)
        write_array_to_csv(file_path + 'analysis_dr.csv', self.dr_poses)
        write_array_to_csv(file_path + 'analysis_est.csv', self.posterior_poses)
