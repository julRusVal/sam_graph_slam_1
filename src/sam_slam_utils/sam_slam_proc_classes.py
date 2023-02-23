#!/usr/bin/env python3

"""
Script for processing data from SMaRC's Stonefish simulation
"""

# %% Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import itertools
import gtsam
import networkx as nx
from sam_slam_utils.sam_slam_helper_funcs import angle_between_rads


# %% Classes

class process_2d_data:
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
            self.dr_poses_graph = self.read_csv_to_array('dr_poses_graph.csv')
            self.gt_poses_graph = self.read_csv_to_array('gt_poses_graph.csv')
            self.detections_graph = self.read_csv_to_array('detections_graph.csv')
            self.buoys = self.read_csv_to_array('buoys.csv')

        elif isinstance(input_data, str):
            self.dr_poses_graph = self.read_csv_to_array(input_data + '/dr_poses_graph.csv')
            self.gt_poses_graph = self.read_csv_to_array(input_data + '/gt_poses_graph.csv')
            self.detections_graph = self.read_csv_to_array(input_data + '/detections_graph.csv')
            self.buoys = self.read_csv_to_array(input_data + '/buoys.csv')

        # Extract data from an instance of sam_slam_listener
        else:
            self.dr_poses_graph = np.array(input_data.dr_poses_graph)
            self.gt_poses_graph = np.array(input_data.gt_poses_graph)
            self.detections_graph = np.array(input_data.detections_graph)
            self.buoys = np.array(input_data.buoys)

        # ===== Clustering and data association =====
        self.n_buoys = len(self.buoys)
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
        self.post_Pose2s = None
        self.post_Point2s = None
        self.bearings_ranges = []

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
        self.slam_result = None

        # ===== Visualization =====
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.colors = ['orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']
        self.plot_limits = [-12.5, 12.5, -5, 20]

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
        ax.scatter(self.dr_poses_graph[:, 0], self.dr_poses_graph[:, 1], color=self.dr_color)

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

        for ind_buoy in range(self.buoys.shape[0]):
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

            ax.scatter(self.buoys[ind_buoy, 0],
                       self.buoys[ind_buoy, 1],
                       color=current_color)

        plt.show()
        return

    def visualize_posterior(self, plot_gt=True, plot_dr=True, plot_buoy=True):
        """
        Visualize The Posterior
        """
        # Check if Optimization has occurred
        if self.slam_result is None:
            print('Need to perform optimization before it can be printed!')
            return

        # Build array for the  pose and point posteriors
        slam_out_poses = np.zeros((len(self.x), 2))
        slam_out_points = np.zeros((len(self.b), 2))
        for i in range(len(self.x)):
            # TODO there has to be a better way to do this!!
            slam_out_poses[i, 0] = self.slam_result.atPose2(self.x[i]).x()
            slam_out_poses[i, 1] = self.slam_result.atPose2(self.x[i]).y()

        for i in range(len(self.b)):
            slam_out_points[i, 0] = self.slam_result.atPoint2(self.b[i])[0]
            slam_out_points[i, 1] = self.slam_result.atPoint2(self.b[i])[1]

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
        if plot_dr:
            ax.scatter(self.dr_poses_graph[:, 0],
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
                ax.scatter(self.buoys[ind_buoy, 0],
                           self.buoys[ind_buoy, 1],
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
            if self.slam_result is None:
                print('Perform optimization before it can be graphed')
                return
            values = self.slam_result
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

    @staticmethod
    def calc_pose_error(array_test, array_true):
        # Positional error
        pos_error = array_test[:, :2] - array_true[:, :2]

        theta_error = np.zeros((array_true.shape[0], 1))
        for i in range(array_true.shape[0]):
            theta_error[i] = angle_between_rads(array_test[i, 2], array_true[i, 2])

        return np.hstack((pos_error,theta_error))

    def show_error(self):
        # Convert the lists of Pose2s to np arrays
        dr_array = self.pose2_list_to_nparray(self.dr_Pose2s)
        gt_array = self.pose2_list_to_nparray(self.gt_Pose2s)
        post_array = self.pose2_list_to_nparray(self.post_Pose2s)

        # TODO figure out ground truth coordinate stuff
        # This is to correct problems with the way the gt pose is converted to the map frame...
        gt_array[:, 2] = np.pi - gt_array[:, 2]

        # Find the errors between gt<->dr and gt<->post
        dr_error = self.calc_pose_error(dr_array, gt_array)
        post_error = self.calc_pose_error(post_array, gt_array)

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

    # ===== Data loading and Pre-process Methods =====
    @staticmethod
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

            f.close()
            return np.array(data, dtype=np.float64)
        except OSError:
            return -1

    def correct_coord_problem(self):
        # TODO this is super hacky and I feel like it'll cause problems later!!
        # TODO Figure out the coordinate system! map and world ned are causing problems
        """
        The appears to be a problem with converting everything to the map frame....
        """
        print('Don\'t do this Julian!')

        self.gt_poses_graph[:, 0] = -self.gt_poses_graph[:, 0]

    # ===== Clustering and data association methods =====
    def fit_cluster_model(self):
        # Check for empty detections_graph
        if self.n_detections < 1:
            print("No detections were detected, improve detector")
            return

        if self.cluster_model is not None:
            self.detection_clusterings = self.cluster_model.fit_predict(self.detections_graph[:, 0:2])

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
                perm_score += (self.buoys[buoy_id, 0] - self.cluster_model.means_[cluster_id, 0]) ** 2
                perm_score += (self.buoys[buoy_id, 1] - self.cluster_model.means_[cluster_id, 1]) ** 2

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
    @staticmethod
    def create_Pose2(input_pose):
        """
        Create a GTSAM Pose3 from the recorded poses in the form:
        [x,y,z,q_w,q_x,q_,y,q_z]
        """
        rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
        rot3_yaw = rot3.yaw()
        # GTSAM Pose2: x, y, theta
        return gtsam.Pose2(input_pose[0], input_pose[1], rot3_yaw)

    @staticmethod
    def pose2_list_to_nparray(pose_list):
        out_array = np.zeros((len(pose_list), 3))

        for i, pose2 in enumerate(pose_list):
            out_array[i, :] = pose2.x(), pose2.y(), pose2.theta()

        return out_array

    def convert_poses_to_Pose2(self):
        """
        Poses is self.
        [x,y,z,q_w,q_x,q_,y,q_z]
        """
        self.dr_Pose2s = []
        self.gt_Pose2s = []

        for dr_pose in self.dr_poses_graph:
            self.dr_Pose2s.append(self.create_Pose2(dr_pose))

        for gt_pose in self.gt_poses_graph:
            self.gt_Pose2s.append(self.create_Pose2(gt_pose))

    def Bearing_range_from_detection_2d(self):
        for detection in self.detections_graph:
            dr_id = int(detection[-1])
            detection_pose = self.dr_Pose2s[dr_id]
            # This Method uses the map coordinates to calc bearing and range
            measurement = gtsam.BearingRange2D.Measure(detection_pose, detection[0:2])
            # This method uses the relative position of the detection, as it is registered in sam/base_link
            # pose_null = self.create_Pose2([0, 0, 0, 1, 0, 0, 0])
            # measurement = gtsam.BearingRange3D.Measure(pose_null, detection[3:5])
            self.bearings_ranges.append(measurement)

    def construct_graph_2d(self):
        """
        Graph made up of gtsam.Pose2 and gtsam.Point2
        """
        self.graph = gtsam.NonlinearFactorGraph()

        # labels
        self.b = {k: gtsam.symbol('l', k) for k in range(self.n_buoys)}
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
                                                   gtsam.Point2(self.buoys[id_buoy, 0],
                                                                self.buoys[id_buoy, 1]),
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
            self.initial_estimate.insert(self.b[buoy_id], gtsam.Point2(self.buoys[buoy_id, 0],
                                                                       self.buoys[buoy_id, 1]))

    def optimize_graph(self):
        if self.graph.size() == 0:
            print('Need to build the graph before is can be optimized!')
            return
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.slam_result = self.optimizer.optimize()

        # Save the posterior results
        self.post_Pose2s = []
        self.post_Point2s = []

        for i in range(len(self.x)):
            self.post_Pose2s.append(self.slam_result.atPose2(self.x[i]))

        for i in range(len(self.b)):
            self.post_Point2s.append(self.slam_result.atPoint2(self.b[i]))

    # ===== Higher level methods =====
    def perform_offline_slam(self):
        self.correct_coord_problem()
        self.cluster_data()
        self.cluster_to_landmark()
        self.convert_poses_to_Pose2()
        self.Bearing_range_from_detection_2d()
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