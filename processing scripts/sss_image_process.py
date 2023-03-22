#!/usr/bin/env python3

"""
Apply image processing techniques to assist in the detection of relevant features
"""

# %% Imports
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Parameters
do_median_blur = True

# Load image
file_name = 'sss_data_2'
img = cv.imread(f'data/{file_name}.png', cv.IMREAD_GRAYSCALE)
height_tot, width_tot = img.shape
width_side = width_tot//2

img_port = np.flip(img[:, :width_side], axis=1)
img_starboard = img[:, width_side:]

if do_median_blur:
    img_port = cv.medianBlur(img_port, 5)
    img_starboard = cv.medianBlur(img_starboard, 5)

# X gradient
grad_port = cv.Sobel(img_port, cv.CV_8U, 1, 0, ksize=5)
grad_starboard = cv.Sobel(img_starboard, cv.CV_8U, 1, 0, ksize=5)

plt.subplot(1, 2, 1)
plt.imshow(np.flip(grad_port, axis=1), cmap='gray')
plt.title('Port')
plt.subplot(1, 2, 2)
plt.imshow(grad_starboard, cmap='gray')
plt.title('Starboard')

complete_img = np.hstack((np.flip(grad_port), grad_starboard))
cv.imwrite(f'data/{file_name}_grad.png', complete_img)

