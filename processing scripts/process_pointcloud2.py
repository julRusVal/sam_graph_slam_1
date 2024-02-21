import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def find_centroid(points):
    """
    Assumes points are given as Nx3 numpy array.
    :param points: Nx3 numpy array of 3d points
    :return: 1x3 numpy array of centroid coordinates
    """
    centroid = np.mean(points, axis=0, keepdims=True)
    return centroid


def fit_plane_pca(points):
    # Use PCA to find the best-fit plane
    pca = PCA(n_components=2)
    pca.fit(points)

    # WRONG
    # The normal vector of the best-fit plane is the first principal component
    normal_vector = pca.components_[0]

    # comp_0 = pca.components_[0]
    # comp_1 = pca.components_[1]
    # normal_vector = np.cross(comp_0, comp_1)

    return normal_vector


def fit_plane_pca_2(points):
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


def fit_plane_svd(points):
    """
    Returns the normal vector corresponding to the best fit plane of the provided points
    Source: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    :param points: Nx3 array of points
    :return:
    """

    centroid = find_centroid(points)

    svd = np.linalg.svd(np.transpose(np.subtract(points, centroid)))

    normal = svd[0][:, -1]

    return normal


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


def plot_points_and_lines(points, k):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', label='Points')

    # Plot lines from every kth point to the origin
    for i in range(0, points.shape[0], k):
        ax.plot([points[i, 0], 0], [points[i, 1], 0], [points[i, 2], 0], c='red', linestyle='--', linewidth=2)

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


# data location
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = script_directory + "/data/point_cloud_data.npy"

data = np.load(data_path)
test_data = data[:, :, 15]

norm_svd = fit_plane_svd(test_data)
norm_pca = fit_plane_pca(test_data)
norm_pca_2 = fit_plane_pca_2(test_data)

# plane_points = project_onto_plane(test_data, norm_svd)
# plot_3d_points_and_plane(test_data, norm_svd, plane_point

plot_points_and_lines(test_data, 5)
