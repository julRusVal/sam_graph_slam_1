#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math


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
    return x_center, y_center

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
    u = ((x3 - x1)*(x2 - x1) + (y3 - y1)*(y2 - y1))/line_mag_sqrd

    if 0 < u < 1.0:
        x_perpendicular = x1 + u * (x2 - x1)
        y_perpendicular = y1 + u * (y2 - y1)

        return math.sqrt((x_perpendicular - x3) ** 2 + (y_perpendicular - y3) ** 2)

    else:
        # Calculate the distance from the third point to each endpoint of the line segment
        distance_line_end_1 = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        distance_line_end_2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)

        # Find the minimum distance
        min_distance = min(distance_line_end_1, distance_line_end_2)

        return min_distance


# Set the random seed for reproducibility
np.random.seed(42)

# Define the parameters of the line
x_1, y_1 = -5, 10
x_2, y_2 = 5, 5
x_3, y_3 = 0, 5  # For testing distance calculations
var_along_scale = 1  # This scales the variance wrt to the separation distance
var_cross = 1

plot_dist_test = True

# Calculate relevant properties
x_center, y_center = calculate_center(x_1, y_1, x_2, y_2)
separation_dist = calculate_distance(x_1, y_1, x_2, y_2)
rotation_angle = calculate_angle(x_1, y_1, x_2, y_2)

# === Construct covariance matrix
# Example covariance matrix
cov_matrix = np.array([[(separation_dist/2.0)**2, 0.0],  # variance defines so that 1 sigma is from center to end
                       [0.0, var_cross]])

# Compute rotation matrix
rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)]])

rot_cov_matrix = rotation_matrix @ cov_matrix @ rotation_matrix.transpose()

# Generate random points from the covariance matrix
num_points = 100
# points = np.random.multivariate_normal(mean=[x_center, y_center], cov=cov_matrix, size=num_points)
rot_points = np.random.multivariate_normal(mean=[x_center, y_center], cov=rot_cov_matrix, size=num_points)

# Extract x and y coordinates
# x = points[:, 0]
# y = points[:, 1]

rot_x = rot_points[:, 0]
rot_y = rot_points[:, 1]

if plot_dist_test:
    radius = calculate_line_point_distance(x_1, y_1, x_2, y_2, x_3,y_3)

    # Generate data points for the circle
    theta = np.linspace(0, 2 * np.pi, 100)  # Angles from 0 to 2*pi
    circle_x = x_3 + radius * np.cos(theta)  # x-coordinates of the circle points
    circle_y = y_3 + radius * np.sin(theta)  # y-coordinates of the circle points

    # Plot the circle and center
    plt.scatter(x_3, y_3)
    plt.plot(circle_x, circle_y)

# Create scatter plot
# plt.scatter(x, y, c='k')
plt.scatter(rot_x, rot_y, c='r')
plt.scatter(x_center, y_center, c='g')
plt.scatter([x_1, x_2], [y_1, y_2], c='b')
plt.plot([x_1, x_2], [y_1, y_2], c='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points from Covariance Matrix\n'
          f'Original: {cov_matrix}\n'
          f'Rotated:{rot_cov_matrix}')
plt.grid(True)
plt.show()
