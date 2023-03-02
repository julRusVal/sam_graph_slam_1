#!/usr/bin/env python3

# %% Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import itertools
import gtsam

import networkx as nx

# from mpl_toolkits.mplot3d import Axes3D
# import math
# from gtsam import utils
# import plot


# %% Functions

# Data input
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


def show_graph_3d(graph, result, poses, label):
    G = nx.DiGraph()
    for i in range(graph.size()):
        factor = graph.at(i)
        for variable in factor.keys():
            if variable in poses.values():
                pos = (result.atPose3(variable).x(), result.atPose3(variable).y())
                G.add_node(variable, pos=pos)
            else:
                pos = (result.atPoint3(variable)[0], result.atPoint3(variable)[1])
                G.add_node(variable, pos=pos)
            for variable2 in factor.keys():
                if variable != variable2:
                    G.add_edge(variable, variable2)

    # Plot the graph using matplotlib
    options = {'node_color': 'black', 'node_size': 25, 'width': 3, 'with_labels': False}

    pos = nx.get_node_attributes(G, 'pos')
    fig = plt.figure()
    nx.draw(G, pos, **options)
    plt.title(f'Factor Graph\n{label}')
    plt.show()


# %% Class

class process_3d_data:
    def __init__(self):
        # Read
        self.dr_poses = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/dr_poses.csv')
        self.dr_poses_graph = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/dr_poses_graph.csv')

        self.gt_poses = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/gt_poses.csv')
        self.gt_poses_graph = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/gt_poses_graph.csv')

        # Detections
        """
        Format
        self.detection_graph: [x_map, y_map, z_map, x_rel, y_rel, z_rel, index of dr]
        """
        self.detections = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/detections_graph.csv')
        self.detection_graph = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/detections_graph.csv')
        # This format is a little in flux so the column indices are here
        self.detection_graph_x = 0
        self.detection_graph_z = 2

        self.buoys = read_csv_to_array('../../../../../KTH/Degree project/sam_slam/processing scripts/data/buoys.csv')
        self.n_buoys = len(self.buoys)

        # Clustering and data association
        self.cluster_model = None
        self.cluster_mean_threshold = 2.0  # means within this threshold will cause fewer clusters to be used
        self.n_clusters = -1
        self.detection_clusterings = None
        self.buoy2cluster = None
        self.cluster2buoy = None

        # Graph parameter
        self.graph = None
        self.x = None
        self.b = None
        self.dr_Pose3s = []
        self.bearings_ranges = []
        self.ang_sig_init = 5 * np.pi / 180
        self.dist_sig_init = 1
        self.ang_sig = 5 * np.pi / 180
        self.dist_sig = 1
        self.detect_dist_sig = 0.5
        self.detect_ang_sig = 25 * np.pi / 180
        self.initial_estimate = None
        self.slam_result = None

        # Visualization
        self.dr_color = 'r'
        self.gt_color = 'b'
        self.post_color = 'g'
        self.colors = ['orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']

    # Visualization methods
    def visualize_raw(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.detections[:, 0], self.detections[:, 1], self.detections[:, 2], color='k')
        ax.scatter(self.gt_poses_graph[:, 0], self.gt_poses_graph[:, 1], self.gt_poses_graph[:, 2], color=self.gt_color)
        ax.scatter(self.dr_poses_graph[:, 0], self.dr_poses_graph[:, 1], self.dr_poses_graph[:, 2], color=self.dr_color)

        plt.title(f'Raw data\n ground truth ({self.gt_color}) and dead reckoning ({self.dr_color})')
        plt.show()

    def visualize_clustering(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # for ind_detect in range(self.detection_graph.shape[0]):
        #     # cluster_num = self.detection_clusterings[ind_detect]
        #     ax.scatter(self.detections[ind_detect, self.detection_graph_x],
        #                self.detections[ind_detect, self.detection_graph_x + 1],
        #                self.detections[ind_detect, self.detection_graph_x + 2],
        #                # color=self.colors[cluster_num % len(self.colors)]
        #               )
        for cluster in range(self.n_clusters):
            inds = self.detection_clusterings == cluster
            ax.scatter(self.detections[inds, 0],
                       self.detections[inds, 1],
                       self.detections[inds, 2],
                       color=self.colors[cluster % len(self.colors)])
        plt.title(f'Clusters\n{self.n_clusters} Detected')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

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
                           self.cluster_model.means_[cluster_num, 2],
                           color=current_color,
                           marker='4')

            ax.scatter(self.buoys[ind_buoy, 0],
                       self.buoys[ind_buoy, 1],
                       self.buoys[ind_buoy, 2],
                       color=current_color)

        plt.title('Buoys\nShowing true buoy positions and associations and the cluster means')
        plt.show()
        return

    def visualize_otimized(self, plot_gt=True, plot_dr=True, plot_buoy=True):
        # Build array for the posteriors
        slam_out_3dpoints = np.zeros((len(self.x), 3))
        slam_out_buoys = np.zeros((len(self.b), 3))
        for i in range(len(self.x)):
            # TODO there has to be a better way to do this!!
            slam_out_3dpoints[i, 0] = self.slam_result.atPose3(self.x[i]).x()
            slam_out_3dpoints[i, 1] = self.slam_result.atPose3(self.x[i]).y()
            slam_out_3dpoints[i, 2] = self.slam_result.atPose3(self.x[i]).z()

        for i in range(len(self.b)):
            slam_out_buoys[i, 0] = self.slam_result.atPoint3(self.b[i])[0]
            slam_out_buoys[i, 1] = self.slam_result.atPoint3(self.b[i])[1]
            slam_out_buoys[i, 2] = self.slam_result.atPoint3(self.b[i])[2]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        if plot_gt:
            ax.scatter(self.gt_poses_graph[:, 0],
                       self.gt_poses_graph[:, 1],
                       self.gt_poses_graph[:, 2],
                       color=self.gt_color)

        if plot_dr:
            ax.scatter(self.dr_poses_graph[:, 0],
                       self.dr_poses_graph[:, 1],
                       self.dr_poses_graph[:, 2],
                       color=self.dr_color)

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
                           self.buoys[ind_buoy, 2],
                           color=current_color)

                # Plot buoy posteriors
                ax.scatter(slam_out_buoys[ind_buoy, 0],
                           slam_out_buoys[ind_buoy, 1],
                           slam_out_buoys[ind_buoy, 2],
                           color=current_color,
                           marker='4',
                           s=50)

        # Plot the posterior
        ax.scatter(slam_out_3dpoints[:, 0], slam_out_3dpoints[:, 1], slam_out_3dpoints[:, 2], color='g')

        plt.title('gt(blue) with posterior(green)')
        plt.show()
    # Pre-process
    # TODO this is super hacky and I feel like it'll cause problems later!!
    # TODO Figure out the coordinate system! map and world ned are causing problems
    def correct_coord_problem(self):
        """
        The appears to be a problem with converting everything to the map frame....
        """
        print('Don\'t do this Julian!')

        self.gt_poses_graph[:, 0] = -self.gt_poses_graph[:, 0]
        # self.gt_poses_graph[:, 2] = -self.gt_poses_graph[:, 2]



    # Clustering and data association methods
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
        self.detection_clusterings = self.cluster_model.fit_predict(
            self.detection_graph[:, self.detection_graph_x:self.detection_graph_z + 1])

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
            self.detection_clusterings = self.cluster_model.fit_predict(
                self.detection_graph[:, self.detection_graph_x:self.detection_graph_z + 1])

    def cluster_to_landmark(self):
        # Use least squares to find the best mapping of clusters onto landmarks
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

    # GTSAM stuff
    @staticmethod
    def create_Pose3(input_pose):
        """
        Create a GTSAM Pose3 from the recorded poses in the form:
        [x,y,z,q_w,q_x,q_,y,q_z]
        """
        rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
        rot3_xyz = rot3.xyz()
        return gtsam.Pose3(rot3, input_pose[0:3])

    def convert_poses_to_Pose3(self):
        """
        Poses is self.
        [x,y,z,q_w,q_x,q_,y,q_z]
        """
        for pose in self.dr_poses_graph:
            self.dr_Pose3s.append(self.create_Pose3(pose))

    def Bearing_range_from_detection(self):
        pose_null = self.create_Pose3([0, 0, 0, 1, 0, 0, 0])
        for detection in self.detection_graph:
            # self.detection_graph[n][3:6] = [x_rel, y_rel, z_rel]
            measurement = gtsam.BearingRange3D.Measure(pose_null, detection[3:6])
            self.bearings_ranges.append(measurement)

    def construct_graph(self):
        self.graph = gtsam.NonlinearFactorGraph()

        # labels
        self.b = {k: gtsam.symbol('l', k) for k in range(self.n_buoys)}
        self.x = {k: gtsam.symbol('x', k) for k in range(len(self.dr_poses_graph))}

        # Priors
        # Agent pose
        # TODO the dr also make covariances available but is not currently used
        prior_model = gtsam.noiseModel.Diagonal.Sigmas((self.ang_sig_init, self.ang_sig_init, self.ang_sig_init,
                                                        self.dist_sig_init, self.dist_sig_init, self.dist_sig_init))

        self.graph.add(gtsam.PriorFactorPose3(self.x[0], self.dr_Pose3s[0], prior_model))

        # Buoys
        prior_model_lm = gtsam.noiseModel.Diagonal.Sigmas((self.dist_sig_init, self.dist_sig_init, self.dist_sig_init))

        for id_buoy in range(self.n_buoys):
            self.graph.add(gtsam.PriorFactorPoint3(self.b[id_buoy],
                                                   gtsam.Point3(self.buoys[id_buoy, 0],
                                                                self.buoys[id_buoy, 1],
                                                                self.buoys[id_buoy, 2]),
                                                   prior_model_lm))

        # Odometry
        odometry_model = gtsam.noiseModel.Diagonal.Sigmas((self.ang_sig, self.ang_sig, self.ang_sig,
                                                           self.dist_sig, self.dist_sig, self.dist_sig))

        for pose_id in range(len(self.dr_Pose3s) - 1):
            between_Pose3 = self.dr_Pose3s[pose_id].between(self.dr_Pose3s[pose_id + 1])
            self.graph.add(
                gtsam.BetweenFactorPose3(self.x[pose_id], self.x[pose_id + 1], between_Pose3, odometry_model))

        # Detections
        # TODO what size should the noise matrix be
        detection_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([self.detect_ang_sig,
                                                                     self.detect_ang_sig,
                                                                     self.detect_dist_sig]))

        for det_id in range(len(self.detection_graph)):
            dr_id = self.detection_graph[det_id][-1]
            buoy_id = self.cluster2buoy[self.detection_clusterings[det_id]]
            # check for a association problem
            if buoy_id < 0:
                continue
            self.graph.add(gtsam.BearingRangeFactor3D(self.x[dr_id],
                                                      self.b[buoy_id],
                                                      self.bearings_ranges[det_id].bearing(),
                                                      self.bearings_ranges[det_id].range(),
                                                      detection_model))

        # Create the initial estimate, using measured poses
        self.initial_estimate = gtsam.Values()
        for pose_id in range(len(self.dr_Pose3s)):
            self.initial_estimate.insert(self.x[pose_id], self.dr_Pose3s[pose_id])

        for buoy_id in range(self.n_buoys):
            self.initial_estimate.insert(self.b[buoy_id], gtsam.Point3(self.buoys[buoy_id, 0],
                                                                       self.buoys[buoy_id, 1],
                                                                       self.buoys[buoy_id, 2]))

    def otptimize_graph(self):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        self.slam_result = self.optimizer.optimize()


# %% Main

if __name__ == "__main__":
    process = process_3d_data()

    process.correct_coord_problem()

    process.cluster_data()

    process.cluster_to_landmark()
    process.convert_poses_to_Pose3()
    process.Bearing_range_from_detection()
    process.construct_graph()
    process.otptimize_graph()

    if True:
        process.visualize_clustering()
        process.visualize_raw()
        process.visualize_otimized()
    if False:
        show_graph_3d(process.graph, process.initial_estimate, process.x, 'Initial')
        show_graph_3d(process.graph, process.current_estimate, process.x, 'Posterior')
    # plot.plot_trajectory(2, process.initial_estimate)

# %% testing junk
#
# process = process_3d_data()
#
# dr_pose_0 = process.dr_poses_graph[0]
# dr_pose_1 = process.dr_poses_graph[1]
#
# gt_pose_0 = process.gt_poses_graph[0]
# gt_pose_1 = process.gt_poses_graph[1]
#
# dr_P_0 = process_3d_data.create_Pose3(dr_pose_0)
# dr_P_1 = process_3d_data.create_Pose3(dr_pose_1)
#
# test_P = process_3d_data.create_Pose3([0, 0, 0, 1, 0, 0, 0])
