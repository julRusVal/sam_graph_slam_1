#!/usr/bin/env python3

# Script for testing out the sea thru implementation from https://github.com/hainh/sea-thru
# Data from https://www.kaggle.com/datasets/viseaonlab/flsea-vi
# Original paper from http://csms.haifa.ac.il/profiles/tTreibitz/webfiles/sea-thru_cvpr2019.pdf
# Data for original paper (not working) https://www.viseaon.haifa.ac.il/datasets
#
# A modified version of this is present in sam_slam_mapping

import numpy as np
import matplotlib.pyplot as plt
import time
import csv

import sea_thru

from PIL import Image

from skimage import data, segmentation, color
from skimage.color import label2rgb
from skimage import filters, morphology
from skimage import graph


def preprocess_depth_map(depths, min_depth, max_depth):
    """
    Truncate the depths such that depth values above and below thresholds are set to zero

    !!THIS WILL NEED TO BE CHANGED FOR MY DATA!!

    :param depths:
    :param min_depth:
    :param max_depth:
    :return:
    """

    new_depths = depths.copy()

    # Check original max range
    max_orig = np.max(new_depths)
    if max_depth > max_orig:
        max_depth = max_orig
    new_depths[new_depths == 0.0] = max_depth
    new_depths[new_depths > max_depth] = max_depth
    new_depths[new_depths < min_depth] = 0.0
    return new_depths


# TODO This is also found in the helpers script
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


# %% Parameters

# Data parameters
dataset_select = 2  # 0: Moorings (shallow and nearby), 1: coral farm (deeper and more distant), 2: Simulation data
dataset_subselect = 2  # 1: original sim data, 2: new sim data from current run

# Data states, determined by which data set is selected
tif_data = False
sim_data = False

# Pre process parameters
depth_min = 1.0
depth_max = 20.0

# Backscatter estimation parameters
bs_bins = 10
bs_fraction = 0.01

# Neighborhood 1 parameters
nh_1_refine = True
nh_1_epsilon = 0.01
nh_1_min_size = 25
nh_1_closing_rad = 1

# Neighborhood 2 parameters (currently unused)
nh_2_compactness = 0.01
nh_2_segments = 800

# Illumination parameters
ill_p = 0.01
ill_f = 2  # recommended by paper
ill_l = 0.5
ill_spread_data_fract = 0.01

# Plotting parameters
plot_depth = False
plot_bs_points = True
plot_bs_results = True
plot_nh_1 = True
plot_nh_2 = True
plot_ill = True
plot_final = True

# %% Import images
if dataset_select == 0:
    image_name = 'test_data/16315300515654855.tiff'
    depth_name = 'test_data/16315300515654855_SeaErra_abs_depth.tif'
    image_proc_name = 'test_data/16315300515654855_SeaErra.tiff'
    tif_data = True

elif dataset_select == 1:
    image_name = 'test_data/16315955526365306.tiff'
    depth_name = 'test_data/16315955526365306_SeaErra_abs_depth.tif'
    image_proc_name = 'test_data/16315955526365306_SeaErra.tiff'
    tif_data = True

elif dataset_select == 2:
    if dataset_subselect == 1:
        img_name = 'sea_thru_data/img_left_8_88_0.jpg'
        mask_name = 'sea_thru_data/mask_left_8_88_0.jpg'
        ranges_name = 'sea_thru_data/left_8_88_0.csv'
        extra_mask = None
    else:
        img_name = 'sea_thru_data/img_left_9_96_0.jpg'
        mask_name = 'sea_thru_data/mask_left_9_96_0.jpg'
        ranges_name = 'sea_thru_data/left_9_96_0.csv'
        extra_mask_name = None  # Specify extra mask will replace refining by depth
        # extra_mask_name = 'sea_thru_data/extra_mask_left_9_96_0.jpg'

    sim_data = True
    start_depth_i = 200  # This will exclude the region proceeding the index, geometrically the image above the rope
    end_depth_i = 625  # This will exclude the region following the index, geometrically the image below the algae

else:
    print("invalid dataset selected")
    exit()

# %% Pre process RGB images and depth maps
if tif_data:
    image_tif = Image.open(image_name)
    depth_tif = Image.open(depth_name)
    image_proc_tif = Image.open(image_proc_name)

    image = np.asarray(image_tif) / 255.0
    image_proc = np.asarray(image_proc_tif) / 255.0
    depths = np.asarray(depth_tif)

    depths = preprocess_depth_map(depths, depth_min, depth_max)
    depths_valid = np.ones((depths.shape[0], depths.shape[1], 1))
    depths_valid[depths == 0] = 0.0

if sim_data:
    image = np.asarray(Image.open(img_name))
    mask = np.asarray(Image.open(mask_name))
    depths = read_csv_to_array(ranges_name)
    if extra_mask_name is not None:
        extra_mask = np.asarray(Image.open(extra_mask_name))
    else:
        extra_mask = None

    image = image / 255.0
    image_proc = image  # this is just for plotting later
    mask = mask / 255.0

    # make sure the mask has a depth of 3
    if len(mask.shape) == 2:
        mask = np.dstack((mask, mask, mask))
    elif len(mask.shape) == 3 and mask.shape[2] != 3:
        mask = np.dstack((mask[:, :, 0], mask[:, :, 0], mask[:, :, 0]))

    # the extra mask is optional but need some processing and checks
    if extra_mask is not None:
        extra_mask = extra_mask / 255.0

        if len(extra_mask.shape) == 2:
            extra_mask = np.dstack((extra_mask, extra_mask, extra_mask))
        elif len(extra_mask.shape) == 3 and extra_mask.shape[2] != 3:
            extra_mask = np.dstack((extra_mask[:, :, 0], extra_mask[:, :, 0], extra_mask[:, :, 0]))

    # TESTING
    # Plot the original image and high gradient regions

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    # find gradiant of original image
    img_gray = color.rgb2gray(image)

    grad_rgb = filters.sobel(image)
    grad_gray = filters.sobel(img_gray)

    # Apply median filter to gradient
    med_size = 11
    grad_med_gray = filters.median(grad_gray, morphology.square(med_size))

    magnitude = np.linalg.norm(grad_rgb, axis=2)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display the high gradient regions as a mask
    ax[1].imshow(grad_gray)
    ax[1].set_title('Gradient Gray')
    ax[1].axis('off')

    ax[2].imshow(grad_med_gray)
    ax[2].set_title(f'Gradient median: {med_size}')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Contruct a mask for the depths map
    depths_valid = np.ones((mask.shape[0], mask.shape[1], 1))
    depths_valid[mask[:, :, 0] < 0.5] = 0.0

    # filter depth mask base mask
    # this is needed because the edges of the mask are not well defined
    # applying erosion will cause these regions to be be excluded
    footprint = morphology.disk(3)[:, :, np.newaxis]
    depths_valid = morphology.erosion(depths_valid, footprint)

    # The depth mask can be refined in one of two way
    # 1) manually define mask: this is called extra mask
    # 2) By the depth of the image: this use knowledge of the farm geometry
    # Note: these methods could be used together but there would be some conflict so for now they are separate

    if extra_mask is not None:
        extra_mask[extra_mask <= 0.5] = 0
        extra_mask[extra_mask > 0.5] = 1

        depths_valid = np.logical_and(depths_valid, extra_mask)

    else:
        # Limit valid depths based on water depth, row index is used to specify
        if 0 <= start_depth_i < depths_valid.shape[0]:
            depths_valid[:start_depth_i, :, 0] = 0.0

        if 0 <= end_depth_i < depths_valid.shape[0]:
            depths_valid[end_depth_i:, :, 0] = 0.0

    # apply the depths mask to the depth array
    depths[depths_valid[:, :, 0] == 0.0] = 0.0

    fig_sim_mask, (ax1, ax2) = plt.subplots(1, 2)
    fig_sim_mask.suptitle('depth and mask')

    ax1.title.set_text('depth')
    ax1.imshow(depths)

    ax2.title.set_text('mask')
    ax2.imshow(mask)

    fig_sim_mask.show()

# %% Backscatter estimation points
ptsR, ptsG, ptsB, bs_points = sea_thru.find_backscatter_estimation_points(image, depths,
                                                                          num_bins=bs_bins,
                                                                          fraction=bs_fraction,
                                                                          z_min=depth_min,
                                                                          z_max=depth_max)

bs_points = bs_points.astype(int)

# %% Backscatter
Br, coefsR = sea_thru.find_backscatter_values(ptsR, depths, restarts=25)
Bg, coefsG = sea_thru.find_backscatter_values(ptsG, depths, restarts=25)
Bb, coefsB = sea_thru.find_backscatter_values(ptsB, depths, restarts=25)

B_rgb = np.dstack((Br, Bg, Bb))

image_backscatter_comp = np.add(image, np.multiply(-B_rgb, depths_valid))

# %% neighborhood method 1
nh_start_time = time.time()
nmap_initial, n_initial = sea_thru.construct_neighborhood_map(depths, nh_1_epsilon)
nh_initial_time = time.time()
if nh_1_refine:
    nmap, n = sea_thru.refine_neighborhood_map(nmap_initial, nh_1_min_size, nh_1_closing_rad)
    nh_refine_time = time.time()
    print('Neighborhood method 1| ' +
          f'Initial: {nh_initial_time - nh_start_time} Refine:{nh_refine_time - nh_initial_time}')
else:
    nmap = nmap_initial
    n = n_initial
    print(f'Neighborhood method 1| Initial: {nh_initial_time - nh_start_time}')

# %% Neighborhood method 2
depths_norm = depths / np.max(depths)
labels1 = segmentation.slic(depths_norm, compactness=nh_2_compactness, n_segments=nh_2_segments, start_label=1,
                            channel_axis=None, enforce_connectivity=True)

# %% Illumination
illR = sea_thru.estimate_illumination(image[:, :, 0], Br, nmap, n, p=ill_p, f=ill_f, max_iters=100, tol=1E-5)
illG = sea_thru.estimate_illumination(image[:, :, 1], Bg, nmap, n, p=ill_p, f=ill_f, max_iters=100, tol=1E-5)
illB = sea_thru.estimate_illumination(image[:, :, 2], Bb, nmap, n, p=ill_p, f=ill_f, max_iters=100, tol=1E-5)
ill = np.stack([illR, illG, illB], axis=2)

# %% Attenuation
beta_D_r, _ = sea_thru.estimate_wideband_attenuation(depths, illR)
refined_beta_D_r, coefsR = sea_thru.refine_wideband_attentuation(depths, illR, beta_D_r,
                                                                 radius_fraction=ill_spread_data_fract, l=ill_l)
beta_D_g, _ = sea_thru.estimate_wideband_attenuation(depths, illG)
refined_beta_D_g, coefsG = sea_thru.refine_wideband_attentuation(depths, illG, beta_D_g,
                                                                 radius_fraction=ill_spread_data_fract, l=ill_l)
beta_D_b, _ = sea_thru.estimate_wideband_attenuation(depths, illB)
refined_beta_D_b, coefsB = sea_thru.refine_wideband_attentuation(depths, illB, beta_D_b,
                                                                 radius_fraction=ill_spread_data_fract, l=ill_l)

# %% Reconstruction
B = np.stack([Br, Bg, Bb], axis=2)
beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
recovered = sea_thru.recover_image_clipped(image, depths, B, beta_D, nmap)
# recovered = sea_thru.recover_image_balanced(image, depths, B, beta_D, nmap)

# %% Plot Depth map
if plot_depth:
    plt.imshow(depths)
    plt.title(f'Depth map\n'
              f'Min/max settings: {depth_min} / {depth_max}\n'
              f'Max value: {np.max(depths)}')
    plt.show()

# %% Plot depth map with back scatter points
if plot_bs_points:
    fig_bs_points, (ax1, ax2) = plt.subplots(1, 2)
    fig_bs_points.suptitle('Backscatter Points')

    ax1.title.set_text('RGB')
    ax1.imshow(image)
    ax1.scatter(bs_points[:, 1], bs_points[:, 0], s=5, c='r')

    ax2.title.set_text('Depth')
    ax2.imshow(depths)
    ax2.scatter(bs_points[:, 1], bs_points[:, 0], s=5, c='r')

    fig_bs_points.show()

# %% Plot backscatter
if plot_bs_results:
    fig_bs, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3)
    fig_bs.suptitle("Backscatter")

    # Plot the individual channels
    ax1.title.set_text('Red')
    img1 = ax1.imshow(np.multiply(Br[:, :, np.newaxis], depths_valid))
    ax2.title.set_text('Green')
    img2 = ax2.imshow(np.multiply(Bg[:, :, np.newaxis], depths_valid))
    ax3.title.set_text('Blue')
    img3 = ax3.imshow(np.multiply(Bb[:, :, np.newaxis], depths_valid))

    fig_bs.colorbar(img1, ax=ax1)
    fig_bs.colorbar(img2, ax=ax2)
    fig_bs.colorbar(img3, ax=ax3)

    # Plot the original and the backscatter compensation image
    ax4.title.set_text('Original Image')
    ax4.imshow(image)

    ax5.axis('off')

    ax6.title.set_text('Backscatter Compensated')
    ax6.imshow(image_backscatter_comp)

    fig_bs.show()

# %% Plot neighborhood method 1
fig_nh_1, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig_nh_1.suptitle(f'neighborhood Map, Method 1\n'
                  f'Local Space Average Color\n'
                  f'epsilon: {nh_1_epsilon}  min size: {nh_1_min_size}  radius: {nh_1_closing_rad}')

ax1.title.set_text('Depths')
ax1.imshow(depths)

ax2.title.set_text('Initial Neighborhoods\n'
                   f'Count: {n_initial}')
ax2.imshow(nmap_initial)

ax3.title.set_text('Refined Neighborhoods\n'
                   f'Count: {n}')
ax3.imshow(nmap)

plt.show()

# %% Plot neighborhood method 2
if plot_nh_2:
    fig_nh_2, ax1 = plt.subplots(1, 1)
    fig_nh_2.suptitle('Neighborhood Map, Method 2\n'
                      'SLIC labels')
    ax1.imshow(labels1)
    plt.show()

# %% Plot Illumination
if plot_ill:
    fig_ill, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3)
    fig_ill.suptitle("Illumination")

    # Top row: R, G, B
    ax1.title.set_text('Red')
    img1 = ax1.imshow(illR)
    ax2.title.set_text('Green')
    img2 = ax2.imshow(illG)
    ax3.title.set_text('Blue')
    img3 = ax3.imshow(illB)

    fig_ill.colorbar(img1, ax=ax1)
    fig_ill.colorbar(img2, ax=ax2)
    fig_ill.colorbar(img3, ax=ax3)

    # Bottom row: neighborhood, empty, combined
    ax4.title.set_text('Neighborhood map')
    img4 = ax4.imshow(nmap)
    # fig_ill.colorbar(img4, ax=ax4)

    ax5.axis('off')

    ax6.title.set_text('RGB illumination')
    img6 = ax6.imshow(ill)
    fig_ill.colorbar(img6, ax=ax6)

    plt.show()

# %% Plot Final results
if plot_final:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Final results')

    ax1.title.set_text('Original')
    ax1.imshow(image)

    ax2.title.set_text('Locally processed')
    ax2.imshow(recovered)

    # ax3.title.set_text('FLSea processed')
    # ax3.imshow(image_proc)
    plt.show()

    # ax4.title.set_text('Valid depths')
    # ax4.imshow(depths_valid)
