import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Script for generating a plot of the template used for buoy detection
"""
def construct_template_kernel(size, sigma, feature_width):
    gaussian_kernel = cv2.getGaussianKernel(size, sigma)

    kernel = np.outer(np.ones(size), gaussian_kernel)

    # apply feature width
    center = size // 2
    # Find half width
    if feature_width <= 0:
        return kernel
    elif int(feature_width) % 2 == 0:
        half_width = (feature_width - 1) // 2
    else:
        half_width = feature_width // 2

    if center - half_width > 0:
        kernel[0:center - half_width, :] = -1 * kernel[0:center - half_width, :]
        kernel[center + half_width + 1:, :] = -1 * kernel[center + half_width + 1:, :]

    return kernel

# template parameters
template_size = 21  # buoy
template_sigma = 2  # buoy
template_feature_width = 10  # buoy

M = 5  # axis tickmark spacing

template = construct_template_kernel(template_size, template_sigma, template_feature_width)

# Define the size of the kernel (N)
N = template.shape[0]  # You can adjust this as needed


# Create a colormap for visualization (positive values in green, negative in red)
cmap = plt.cm.coolwarm

# Create a figure and plot the kernel
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
plt.imshow(template, cmap=cmap, interpolation='none')
plt.colorbar(label='Value')
plt.title('Template')
plt.xlabel('Column')
plt.ylabel('Row')
plt.grid(False)

# Force the axis values (ticks) to be integers
ax.set_xticks(np.arange(0, N, M))
ax.set_yticks(np.arange(0, N, M))
ax.set_xticklabels(map(int, ax.get_xticks()))
ax.set_yticklabels(map(int, ax.get_yticks()))

# plt.show()

plt.savefig('data/figures/template_high_quality.png', dpi=300, bbox_inches='tight')