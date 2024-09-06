import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
from sam_slam_utils.sam_slam_proc_classes import calculate_center, calculate_angle, calculate_distance
from scipy.stats import chi2

"""
Plot the locations of the pipeline junctions, sometimes called buoys as ropes, lines, are defined between buoys
"""

def clr_pck(index):
    """
    Returns the name of a color from the list.
    This is intended to be used for plotting purposes when the color itself is not too important.

    :param index: provide index of the color
    :return:
    """
    color_names = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',
        'gray', 'orange', 'purple', 'brown', 'pink', 'olive', 'navy', 'teal',
        'coral', 'lime', 'indigo', 'turquoise'
    ]

    # Wrap the index around if needed
    wrapped_index = int(index) % len(color_names)

    # Return the color name
    return color_names[wrapped_index]

def plot_ellipse(ax, covariance_matrix, center=(0, 0), n_std=3, **kwargs):
    """
    Plot a 3-sigma confidence ellipse for a given covariance matrix.

    Parameters:
    - ax: Matplotlib axis to plot on.
    - covariance_matrix: The 2x2 covariance matrix.
    - center: The center of the ellipse (default: (0, 0)).
    - n_std: Number of standard deviations to define the ellipse (default: 3).
    - **kwargs: Additional keyword arguments to pass to the ellipse plotting function.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # 3-sigma confidence interval corresponds to 99.73%
    chi_square_value = chi2.ppf(0.9973, 2)

    # Scaling factor for the ellipse size
    scale_factor = np.sqrt(chi_square_value)

    ellipse = Ellipse(center, 2 * n_std * np.sqrt(eigenvalues[0]) * scale_factor,
                      2 * n_std * np.sqrt(eigenvalues[1]) * scale_factor,
                      angle, **kwargs)
    ax.add_patch(ellipse)

def confidence_ellipse(ax, covariance_matrix, center =(0, 0), n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    cov = covariance_matrix
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(center[0], center[1])

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return

rope_info_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/pipeline/rope_info.csv"
buoy_info_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/pipeline/buoys.csv"

# Load
# Lines
# Fomrat: [[x, y, xx, xy, yx, yy], ...]
lines = np.genfromtxt(rope_info_path,
                       delimiter=',', dtype=float)
line_count = lines.shape[0]

# Format:[[x, y, z], ...]
buoys = np.genfromtxt(buoy_info_path,
                      delimiter=',', dtype=float)
buoy_count = buoys.shape[0]

# Calculate the centers local
line_inds = [[0, 1], [1, 2], [2, 3]]
local_centers = []
for line_ind, buoy_inds in enumerate(line_inds):
    start_buoy = buoy_inds[0]
    end_buoy = buoy_inds[1]
    x1, y1 = buoys[start_buoy, 0], buoys[start_buoy, 1]
    x2, y2 = buoys[end_buoy, 0], buoys[end_buoy, 1]

    center = calculate_center(x1, y1, x2, y2)
    local_centers.append(center)

local_centers = np.array(local_centers)

cov_mats = np.zeros((2, 2, line_count))
for line_ind in range(line_count):
    cov_mat = lines[line_ind, 2:].reshape(2,2)
    cov_mats[:, :, line_ind] = cov_mat

# Plot set up
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.title(f'Pipeline map')
plt.grid(True)

# Plot
# ax.scatter(lines[:, 0], lines[:, 1], s=50, marker="+", color='blue', label='Saved centers')
# ax.scatter(local_centers[:, 0], local_centers[:, 1], s=30, alpha=0.75, color='red', label='Local centers')
ax.scatter(buoys[:, 0], buoys[:, 1], color='black', label='Segment ends')

# Plot the 3-sigma confidence ellipse
# Testing
covariance_matrix = np.array([[1.0, 0.6], [0.6, 2.0]])
# test_scatter = np. array([[10, 0], [-10, 0], [0, 10], [0, -10]])
# ax.scatter(test_scatter[:, 0], test_scatter[:, 1], s=50, marker="+", color='blue', label='Saved centers')
# plot_ellipse(ax, covariance_matrix, color='blue', alpha=0.3, label='3-sigma Confidence Ellipse')
for line_ind in range(line_count):
    center_x, center_y = lines[line_ind, 0], lines[line_ind, 1]
    confidence_ellipse(ax, cov_mats[:, :, line_ind], center=(center_x, center_y), edgecolor='red', label="thing")

ax.legend()

plt.show()
