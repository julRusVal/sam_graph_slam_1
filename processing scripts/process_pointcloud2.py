import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def fit_best_plane(points):
    # Use PCA to find the best-fit plane
    pca = PCA(n_components=2)
    pca.fit(points)

    # WRONG
    # The normal vector of the best-fit plane is the first principal component
    # normal_vector = pca.components_[0]

    comp_0 = pca.components_[0]
    comp_1 = pca.components_[1]
    normal_vector = np.cross(comp_0, comp_1)

    return normal_vector

def project_onto_plane(points, normal_vector):
    # Project all points onto the plane defined by the normal vector
    projected_points = points - np.outer(points.dot(normal_vector), normal_vector)

    return projected_points

def plot_3d_points_and_plane(points, normal_vector, projected_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points', marker='o', c='b')

    # Plot the best-fit plane
    xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 10),
                         np.linspace(min(points[:, 1]), max(points[:, 1]), 10))
    zz = (-normal_vector[0] * xx - normal_vector[1] * yy) / normal_vector[2]
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.3, label='Best-fit Plane')

    # Scatter plot of projected points
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2],
               label='Projected Points', marker='x', c='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Best-fit Plane and Projected Points')
    # ax.legend()

    plt.show()

def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z columns
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show()

# data location
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = script_directory + "/data/point_cloud_data.npy"

data = np.load(data_path)
test_data = data[:, :, 15]

best_norm = fit_best_plane(test_data)
plane_points = project_onto_plane(test_data, best_norm)
plot_3d_points_and_plane(test_data, best_norm, plane_points)

# plot_3d_points(data[:, :, 15])
