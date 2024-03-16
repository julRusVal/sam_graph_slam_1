#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
from skimage import color
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks, hough_ellipse
from sklearn.decomposition import PCA


def find_centroid(points):
    """
    Assumes points are given as Nx3 numpy array.
    :param points: Nx3 numpy array of 3d points
    :return: 1x3 numpy array of centroid coordinates
    """
    centroid = np.mean(points.reshape(-1, 3), axis=0, keepdims=True)
    return centroid


def fit_plane_pca(points):
    # Use PCA to find the best-fit plane
    pca = PCA(n_components=2)
    pca.fit(points)

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


def project_3d_points_onto_plane(points, vector_a, vector_b=None, point_on_plane=None):
    # If point_on_plane is not provided, use the origin as a point on the plane
    # points = np.transpose(points)
    if point_on_plane is None:
        point_on_plane = np.zeros(3)

    if vector_b is None:
        # if only one vector is provided is it used as the normal vector
        normal_vector = vector_a
    else:
        # Calculate the normal vector to the plane
        normal_vector = np.cross(vector_a, vector_b)

    # Calculate the projection
    points_relative = points - point_on_plane
    projected_points = points_relative - np.outer(points_relative.dot(normal_vector), normal_vector)

    return projected_points


def plot_2d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D point plots')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()


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


def find_non_zero_indices(array):
    """
    Returns the indices of the min and max indices of non-zero elements for both axes
    :param array: np array
    :return: min_row, max_row, min_col, max_col
    """
    non_zero_indices = np.nonzero(array)
    min_row = np.min(non_zero_indices[0])
    max_row = np.max(non_zero_indices[0])
    min_col = np.min(non_zero_indices[1])
    max_col = np.max(non_zero_indices[1])
    return min_row, max_row, min_col, max_col


class process_pointcloud_data:
    def __init__(self, points_3d, plot_process_results=False):
        """
        Assumes points are given as Nx3 numpy array.
        """
        # Original 3d points
        self.points_3d = points_3d.reshape(-1, 3)

        # Projection variable and parameters
        self.centroid = find_centroid(self.points_3d)
        self.x_new_basis = np.array([0, 0, 1])  # this needs to be updated
        self.y_new_basis = np.array([0, 1, 0])
        self.define_smart_basis()

        # Points projected onto the 'optimal' plane
        self.projected_points_3d = project_3d_points_onto_plane(self.points_3d,
                                                                self.x_new_basis,
                                                                self.y_new_basis)

        self.projection_matrix_3_2 = np.hstack([self.x_new_basis.reshape(-1, 1),
                                                self.y_new_basis.reshape(-1, 1)])

        self.projected_points_2d = np.dot(self.points_3d, self.projection_matrix_3_2)

        # === Processing parameters ===
        # (1) Convert 2d points to image
        self.scale = 100
        self.y_offset = 0  # Set by process_2d_points
        # (2) Range clipping
        self.perform_range_clipping = False  # Removes the most distant returns
        self.range_clipping_range = int(0.5 * self.scale)  # provide in units of meters
        # (2a) truncating - The hope is to make hough run a little faster
        self.perform_truncation = True
        self.truncating_margin = 5
        self.truncated_ind = 0
        # (3) Dilation
        self.dilation_radius = 3  # The detections are a little sparce so to thicken stuff up
        # (4) Hough lines
        self.perform_h_lines = False
        self.angle_bins = 360
        self.max_line_count = 2
        # (5) Hough circles
        self.h_circle_min_rad = 90
        self.h_circle_max_rad = 110
        self.h_circle_radius_resolution = 5
        # (6) Hough ellipse
        self.perform_h_ellipse = False
        self.h_ellipse_accuracy = 2
        self.h_ellipse_threshold = 20
        self.h_ellipse_min_size = 20
        self.h_ellipse_max_size = 100
        # (7) Determine detection

        # (8) Plotting
        self.plot_process_results = plot_process_results

        # Output
        self.detection_coords_basis = np.zeros([0, 0])  # 2D coords wrt the internal basis
        self.detection_coords_world = np.zeros([0, 0])  # world, 3D, coords wrt the reference frame of the original data

        self.process_2d_points()

    def define_smart_basis(self):
        if self.centroid is None:
            return

        # y basis is unchanged
        self.y_new_basis = np.array([0, 1, 0])

        # New x basis is aligned to the centroid of the 3d points
        x_new = np.array([self.centroid[0, 0], 0, self.centroid[0, 2]])
        self.x_new_basis = x_new / np.linalg.norm(x_new)

    def process_2d_points(self):
        """

        """
        # === (1) Convert 2D points to image ===
        # ======================================
        x = self.projected_points_2d[:, 0] * self.scale
        y = self.projected_points_2d[:, 1] * self.scale

        # offset to account for negative y values
        if np.min(y) < 0:
            self.y_offset = abs(np.min(y))
        y = y + self.y_offset

        # create an image from list of points
        x_shape = int(np.max(x))  #
        y_shape = int(np.max(y) - np.min(y))

        image_float = np.zeros((x_shape + 1, y_shape + 1))
        indices = np.stack([x - 1, y - 1], axis=1).astype(int)
        try:
            image_float[indices[:, 0], indices[:, 1]] = 1
        except IndexError as e:
            # Catch the IndexError exception
            print("IndexError:", e)
            self.detection_coords_world = np.zeros([0, 0])
            return

        image_float[indices[:, 0], indices[:, 1]] = 1
        image_float_original = np.copy(image_float)

        # === (2) Range clipping ===
        # ==========================
        if self.perform_range_clipping:
            image_float[-self.range_clipping_range:, :] = 0

        # === (2a) truncating ===
        # ========================
        # Only truncate in the down range direction, dimension 0
        if self.perform_truncation:
            min_ind, _, _, _ = find_non_zero_indices(image_float)
            new_min_ind = min_ind - self.truncating_margin
            if new_min_ind > 0:
                self.truncated_ind = new_min_ind
                image_float = image_float[self.truncated_ind:, :]

        # === (3) Dilation ===
        # ====================
        if self.dilation_radius > 0:
            dilation_footprint = skimage.morphology.disk(int(self.dilation_radius))
            image_float = skimage.morphology.dilation(image_float, footprint=dilation_footprint)

        # TODO look into casting image as a bool
        image_bool = image_float.astype(bool)

        # === (4) Hough Line ===
        # ======================
        if self.perform_h_lines:
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, self.angle_bins, endpoint=False)  # Generate bins
            h, theta, d = hough_line(image_bool, theta=tested_angles)  # Perform Hough line transform
            h_line_peaks = hough_line_peaks(h, theta, d, num_peaks=self.max_line_count)  # Select

        # === (5) Hough circle ===
        # ========================
        tested_radii = np.arange(self.h_circle_min_rad, self.h_circle_max_rad, self.h_circle_radius_resolution)
        h_circle_results = hough_circle(image_bool, tested_radii)
        h_circle_peaks = hough_circle_peaks(h_circle_results, tested_radii, total_num_peaks=10)  # accum, cx, cy, rad

        # === (6) Hough ellipse ===
        # =========================
        if self.perform_h_ellipse:
            h_ellipse_results = hough_ellipse(image_bool,
                                              accuracy=self.h_ellipse_accuracy,
                                              threshold=self.h_ellipse_threshold,
                                              min_size=self.h_ellipse_min_size,
                                              max_size=self.h_ellipse_max_size)
            h_ellipse_results.sort(order='accumulator')
            h_ellipse_results = np.flipud(h_ellipse_results)  # get results in descending order
        else:
            h_ellipse_results = np.zeros(shape=[0, 0])  # empty array denotes no ellipses

        # Estimated parameters for the ellipse
        if h_ellipse_results.size != 0:
            best = list(h_ellipse_results[0])
            yc, xc, a, b = (int(round(x)) for x in best[1:5])
            orientation = best[5]
        else:
            yc, xc, a, b, orientation = 0, 0, 0, 0, 0

        # === (7) Determine ====
        # print("Finding center")
        if bool(h_circle_peaks):
            centers_y = h_circle_peaks[1].reshape(-1, 1)
            centers_x = h_circle_peaks[2].reshape(-1, 1)
            centers = np.hstack([centers_x, centers_y])
            avg_center = np.mean(centers, axis=0)

            # Return to the orignal basis - undo scaling and offsetting
            self.detection_coords_basis = np.copy(avg_center)
            self.detection_coords_basis[0] += self.truncated_ind
            self.detection_coords_basis[1] -= self.y_offset
            self.detection_coords_basis /= self.scale

            # Reproject
            self.detection_coords_world = np.dot(self.projection_matrix_3_2, self.detection_coords_basis.reshape(2, 1))
        else:
            self.detection_coords_world = np.zeros([0, 0])
            avg_center = np.zeros([0, 0])

        # === (8) Plotting results ===
        # ============================
        if self.plot_process_results:
            # === Convert original images to rgb ===
            # ======================================
            # images are stored as floats: [0,1]
            image_original = color.gray2rgb(image_float_original)  # original sized image, raw
            image_original_marked = np.copy(image_original)  # original sized image, marked
            image_marked = color.gray2rgb(image_float)  # truncated image

            # === Draw Hough lines ===
            # ========================
            if bool(h_line_peaks):  # Check if result tuple is empty
                for _, angle, dist in zip(*h_line_peaks):
                    # Convert Hough parameters to Cartesian coordinates
                    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                    y1 = (dist - image_marked.shape[1] * np.cos(angle)) / np.sin(angle)

                    # Clip the coordinates to be within the image bounds
                    y0 = max(0, min(y0, image_marked.shape[0] - 1))
                    y1 = max(0, min(y1, image_marked.shape[0] - 1))

                    # Use skimage.draw to draw the line on the image
                    rr, cc = skimage.draw.line(r0=(round(y0)), c0=0,
                                               r1=int(round(y1)), c1=image_marked.shape[1] - 1)

                    # draw onto the truncated image
                    image_marked[rr, cc] = (0, 0, 1.0)

            # === Draw Hough circles ===
            # ==========================
            if bool(h_circle_peaks):  # Check if result tuple is empty
                for _, cy, cx, rad in zip(*h_circle_peaks):
                    # Use skimage.draw to draw the line on the image
                    rr, cc = skimage.draw.circle_perimeter(r=cx, c=cy,
                                                           radius=int(rad), shape=image_marked.shape[:2])
                    image_marked[rr, cc] = (1.0, 0, 1.0)

                    # plot the avg center
                    rr, cc = skimage.draw.circle_perimeter(r=int(avg_center[0]), c=int(avg_center[1]),
                                                           radius=5, shape=image_marked.shape[:2])
                    image_marked[rr, cc] = (0.0, 1.0, 1.0)

            # === Draw the ellipse on the original image ===
            # ==============================================
            draw_count = 5
            if h_ellipse_results.size > 0:  # Ellipse results are stored in a np array, check if its empty
                for result_ind, result in enumerate(h_ellipse_results):
                    if result_ind >= draw_count:
                        break

                    print(f"plotting - {result_ind}")
                    result_yc, result_xc, result_a, result_b = (int(round(x)) for x in list(result)[1:5])
                    result_orientation = result[5]

                    cy, cx = ellipse_perimeter(result_yc, result_xc, result_a, result_b, result_orientation)
                    image_marked[cy, cx] = (1.0, 0, 0)

                    rr, cc = skimage.draw.disk((yc, xc), 10, shape=image_marked.shape[:2])
                    image_marked[rr, cc] = (0, 1.0, 0)

            # Apply the image_marked to its its original counterpart
            image_original_marked[self.truncated_ind:, :] = image_marked

            # === Plotting ===
            # ================
            fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharey=True)

            ax1.set_xlabel("Y")
            ax1.set_ylabel("X")
            ax1.set_title('Original picture')
            ax1.imshow(image_original)

            # for _, angle, dist in zip(*hough_peaks):
            #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            #     ax2.axline((x0, y0), slope=np.tan(angle + np.pi / 2))

            ax2.set_xlabel("Y")
            ax2.set_title(f'Hough Results center: {avg_center[0]}, {avg_center[1]}')
            ax2.imshow(image_original_marked)

            plt.show()

        return

    def plot_points_and_lines(self, k=10):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points
        ax.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2],
                   c='blue', marker='o', label='Points')

        # Plot detection
        if self.detection_coords_world.size == 3:
            coords = self.detection_coords_world.reshape(-1, )
            ax.scatter(coords[0], coords[1], coords[2],
                       c='magenta', marker='o', label='Detection')

        # Plot lines from every kth point to the origin
        for i in range(0, self.points_3d.shape[0], k):
            ax.plot([self.points_3d[i, 0], 0], [self.points_3d[i, 1], 0], [self.points_3d[i, 2], 0],
                    c='red', linestyle='--', linewidth=2)

        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()


if __name__ == '__main__':
    # data location
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = script_directory + "/data/pc_transformed.npy"

    data = np.load(data_path)
    test_data = data[:, :, 15]

    detector = process_pointcloud_data(test_data, plot_process_results=True)
    detector.plot_points_and_lines(k=10)

    # centroid = find_centroid(test_data)
    #
    # # define new plane
    # y_new = np.array([0, 1, 0])
    # x_new = np.array([centroid[0, 0], 0, centroid[0, 2]])
    # x_new = x_new / np.linalg.norm(x_new)
    #
    # # project onto 3d plane
    # projected_3d_points = project_3d_points_onto_plane(test_data, x_new, y_new)  # (test_data, y_new, x_new)
    #
    # # project
    # matrix_3_2 = np.hstack([x_new.reshape(-1, 1),
    #                         y_new.reshape(-1, 1)])
    #
    # projected_2d_points = np.dot(test_data, matrix_3_2)
    # # process_2d_points(projected_2d_points)  # THE MAGIC
    #
    # # norm_svd = fit_plane_svd(test_data)  # no longer using 'fancy' planes
    # # norm_pca = fit_plane_pca(test_data)
    # # norm_pca_2 = fit_plane_pca_2(test_data)
    #
    # # # plane_points = project_onto_plane(test_data, norm_svd)
    # # # plot_3d_points_and_plane(test_data, norm_svd, plane_point
    #
    # # plot_2d_points(projected_2d_points)
    # plot_points_and_lines(projected_3d_points, 20)
