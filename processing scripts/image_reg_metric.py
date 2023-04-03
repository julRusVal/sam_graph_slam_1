#!/usr/bin/env python3
"""
This script is for testing different metrics for image registrations
"""
import cv2
import numpy as np

# %% Parameters
"""
Select two images and their associated masks to compare. These image should be of the same plane 
"""
# ===== Select the images for analysis =====
# Mac
path_name = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/image_registration/"
# linux
# path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/image_registration/'

# Set 0 - generic left pair
# img_0_name = "Warping_left_8_96_2"
# img_1_name = "Warping_left_12_100_2"

# Set 1 - generic down pair
# img_0_name = "Warping_down_14_102_2"
# img_1_name = "Warping_down_15_103_2"

# Set 2 -
img_0_name = "Warping_down_14_102_2"
img_1_name = "Warping_left_13_101_2"

# Other parameters
verbose_output = True

# %% Load data to process
img_0 = cv2.imread(f"{path_name}images/{img_0_name}.jpg")
img_1 = cv2.imread(f"{path_name}images/{img_1_name}.jpg")

mask_0 = cv2.imread(f"{path_name}masks/{img_0_name}.jpg")
mask_1 = cv2.imread(f"{path_name}masks/{img_1_name}.jpg")

# Record shape information
img_0_shape = img_0.shape
img_1_shape = img_1.shape

mask_0_shape = mask_0.shape
mask_1_shape = mask_1.shape

# %% Check for shape agreement
if img_0_shape[0] != img_1_shape[0] or img_0_shape[1] != img_1_shape[1]:
    print("Image shape mismatch!")

if mask_0_shape[0] != mask_1_shape[0] or mask_0_shape[1] != mask_1_shape[1]:
    print("mask shape mismatch!")

if img_0_shape[0] != mask_0_shape[0] or img_0_shape[1] != mask_0_shape[1]:
    print("Image/mask shape mismatch!")

height = img_0_shape[0]
width = img_0_shape[1]

# %% Use masks to determine region of overlap
"""
The similarity analysis will only be performed in regions with overlap. An alternative approach would be to look for
discontinuities across the seams. This, for the time being, will be left to the reader as an exercise.
"""

mask_overlap = np.full((height, width), False, dtype=bool)
mask_overlap[np.logical_and(mask_0[:, :, 0] >= 255 / 2, mask_1[:, :, 0] >= 255 / 2)] = True

img_0_overlap = np.zeros_like(img_0)
img_0_overlap[mask_overlap] = img_0[mask_overlap]

img_1_overlap = np.zeros_like(img_1)
img_1_overlap[mask_overlap] = img_1[mask_overlap]

if verbose_output:
    cv2.imwrite(f"{path_name}output/img_0_overlap.jpg", img_0_overlap)
    cv2.imwrite(f"{path_name}output/img_1_overlap.jpg", img_1_overlap)

# %% Perform comparison
result_sqdiff = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_SQDIFF_NORMED)[0][0]
result_ccorr = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_CCORR_NORMED)[0][0]
result_ccoeff = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_CCOEFF_NORMED)[0][0]
