#!/usr/bin/env python3

import math
import numpy as np
import gtsam
import networkx as nx
import matplotlib.pyplot as plt
import csv


# ===== General stuff =====
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


def read_csv_to_list(file_path):
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        f.close()
        return data
    except OSError:
        return -1


def write_array_to_csv(file_path, data_array):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data_array:
            writer.writerow(row)


def angle_between_rads(target_angle, source_angle):
    # Bound the angle [-pi, pi]
    target_angle = math.remainder(target_angle, 2 * np.pi)
    source_angle = math.remainder(source_angle, 2 * np.pi)

    diff_angle = target_angle - source_angle

    if diff_angle > np.pi:
        diff_angle = diff_angle - 2 * np.pi
    elif diff_angle < -1 * np.pi:
        diff_angle = diff_angle + 2 * np.pi

    return diff_angle


def calc_pose_error(array_test: np.ndarray, array_true: np.ndarray):
    """
    Calculate the error between two arrays of equal size.
    x: [:,0]
    y: [:,1]
    theta: [:,2]
    """
    # Positional error
    pos_error = array_test[:, :2] - array_true[:, :2]

    theta_error = np.zeros((array_true.shape[0], 1))
    for i in range(array_true.shape[0]):
        theta_error[i] = angle_between_rads(array_test[i, 2], array_true[i, 2])

    return np.hstack((pos_error, theta_error))


# ===== GTSAM Stuff =====
def create_Pose2(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    rot3_yaw = rot3.yaw()
    # GTSAM Pose2: x, y, theta
    return gtsam.Pose2(input_pose[0], input_pose[1], rot3_yaw)

def create_Pose3(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    return gtsam.Pose3(r=rot3, t=np.array((input_pose[0], input_pose[1], input_pose[2]), dtype=np.float64))

def merge_into_Pose3(input_pose2, input_rpd):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """

    # Calculate rotation component of Pose3
    # The yaw is provided by the pose2 and is combined with roll and pitch
    q_from_rpy = get_quaternion_from_euler(input_rpd[0],
                                           input_rpd[1],
                                           input_pose2[2])

    rot3 = gtsam.Rot3.Quaternion(q_from_rpy[0], q_from_rpy[1], q_from_rpy[2], q_from_rpy[3])

    # Calculate translation component of Pose3
    trans = np.array((input_pose2[0], input_pose2[1], input_rpd[2]), dtype=np.float64)

    return gtsam.Pose3(r=rot3, t=trans)

def pose2_list_to_nparray(pose_list):
    out_array = np.zeros((len(pose_list), 3))

    for i, pose2 in enumerate(pose_list):
        out_array[i, :] = pose2.x(), pose2.y(), pose2.theta()

    return out_array


def show_simple_graph_2d(graph, x_keys, b_keys, values, label):
    """
    Show Graph of Pose2 and Point2 elements
    This function does not display data association colors

    """
    plot_limits = [-12.5, 12.5, -5, 20]

    # Initialize network
    G = nx.Graph()
    for i in range(graph.size()):
        factor = graph.at(i)
        for key_id, key in enumerate(factor.keys()):
            # Test if key corresponds to a pose
            if key in x_keys.values():
                pos = (values.atPose2(key).x(), values.atPose2(key).y())
                G.add_node(key, pos=pos, color='black')

            # Test if key corresponds to points
            elif key in b_keys.values():
                pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])
                G.add_node(key, pos=pos, color='yellow')
            else:
                print('There was a problem with a factor not corresponding to an available key')

            # Add edges that represent binary factor: Odometry or detection
            for key_2_id, key_2 in enumerate(factor.keys()):
                if key != key_2 and key_id < key_2_id:
                    # detections will have key corresponding to a landmark
                    if key in b_keys.values() or key_2 in b_keys.values():
                        G.add_edge(key, key_2, color='red')
                    else:
                        G.add_edge(key, key_2, color='blue')

    # ===== Plot the graph using matplotlib =====
    # Matplotlib options
    fig, ax = plt.subplots()
    plt.title(f'Factor Graph\n{label}')
    ax.set_aspect('equal', 'box')
    plt.axis(plot_limits)
    plt.grid(True)
    plt.xticks(np.arange(plot_limits[0], plot_limits[1] + 1, 2.5))

    # Networkx Options
    pos = nx.get_node_attributes(G, 'pos')
    e_colors = nx.get_edge_attributes(G, 'color').values()
    n_colors = nx.get_node_attributes(G, 'color').values()
    options = {'node_size': 25, 'width': 3, 'with_labels': False}

    # Plot
    nx.draw_networkx(G, pos, edge_color=e_colors, node_color=n_colors, **options)
    np.arange(plot_limits[0], plot_limits[1] + 1, 2.5)
    plt.show()


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    Author: AutomaticAddison.com

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qw, qx, qy, qz: The orientation in quaternion [w, x,y,z] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


class odometry_data:
    def __init__(self, x, y, z, q_w, q_x, q_y, q_z, roll, pitch, depth, image_id=-1):
        self.x = x
        self.y = y
        self.z = z
        self.q_w = q_w
        self.q_x = q_x
        self.q_y = q_y
        self.q_z = q_z
        self.roll = roll
        self.pitch = pitch
        self.depth = depth
        self.image_id = image_id
