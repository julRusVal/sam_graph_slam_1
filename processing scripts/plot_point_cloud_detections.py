#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from sam_slam_utils import process_pointcloud2


def plot_points_lazy(points_3d, pipelines=None):
    # Create a 3D plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    # Plot all points
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
    #            c='blue', marker='o', label='Points')

    ax.scatter(points_3d[:, 0], points_3d[:, 1],
               c='blue', marker='o', label='Points')

    # Plot all points
    if pipelines is not None:
        ax.scatter(pipelines[0, 0, :], pipelines[0, 1, :], pipelines[0, 2, :],
                   c='red', marker='o', label='Centers')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.legend()

    # Show the plot
    plt.show()


script_directory = os.path.dirname(os.path.abspath(__file__))
# data path
data_path = script_directory + "/data/pc_transformed.npy"

# transforms paths
w2l_transform_path = script_directory + "/data/world_to_local.npy"
l2w_transform_path = script_directory + "/data/local_to_world.npy"

# Pipelines
p_0 = np.array([[877.2, 0, -235.7],  # center
               [0, 1.756, 0],  # Orientation r, p, y
               [105, 1, 1]])  # Dimensions

p_1 = np.array([[1006.6, 0, -314.8],  # center
               [0, 42.691, 0],  # Orientation r, p, y
               [200, 1, 1]])  # Dimensions

p_2 = np.array([[1130.2, 0, -397],  # center
               [0, 0.402, 0],  # Orientation r, p, y
               [100, 1, 1]])  # Dimensions

pipelines = np.dstack([p_0, p_1, p_2])

# load data
pc_data = np.load(data_path)
w2l_transforms = np.load(w2l_transform_path)
l2w_transforms = np.load(l2w_transform_path)

frame_count = pc_data.shape[2]
w2l_transform_count = w2l_transforms.shape[2]
l2w_transform_count = l2w_transforms.shape[2]

#
world_detections = np.zeros([frame_count, 3])
for frame_i in range(frame_count):
    current_data = pc_data[:, :, frame_i]
    current_l2w_transform = l2w_transforms[:, :, frame_i]

    detector = process_pointcloud2.process_pointcloud_data(current_data, plot_process_results=False)
    #detector.plot_points_and_lines(k=10)

    detection_local = detector.detection_coords_world  # NOTE: these coords are in the local 3d coordinates

    # Convert the detection into world coordinates using the homogeneous transform
    detection_local_homo = np.vstack([detection_local.reshape(3, 1),
                                np.array([1])])

    detection_world = np.matmul(current_l2w_transform, detection_local_homo)

    world_detections[frame_i, :] = detection_world[:3, 0]

plot_points_lazy(world_detections)


