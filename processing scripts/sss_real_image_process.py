#!/usr/bin/env python3

"""
Apply image processing techniques to assist in the detection of relevant features

This script is intended to process the real data collected at the algae farm.
"""

# %% Imports
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import signal
import sllib
from PIL import Image
import math

# Parameters
do_median_blur = False
process_sam_sss = False
process_boat_sss = False
sam_file_name = 'sss_data_7608.jpg'
seq_file_name = 'sss_seqs_7608.csv'
boat_file_name = 'Sonar_2023-05-03_20.51.26.sl2'

detections = [[7106, 1092], [6456, 1064],
              [5570, 956], [4894, 943],
              [4176, 956], [3506, 924],
              [2356, 911], [1753, 949],
              [1037, 941], [384, 943]]


class process_sss:
    def __init__(self, data_file_name, seq_file_name, start_ind=None, end_ind=None, max_range_ind=None):
        # Parameters
        self.canny_l_threshold = 100
        self.canny_h_threshold = 175
        self.canny_kernel_size = 5

        #
        self.data_file_name = data_file_name
        self.data_label = self.data_file_name.split(sep='.')[0]
        self.seq_file_name = seq_file_name
        self.seq_ids = np.genfromtxt(f'data/{self.seq_file_name}', delimiter=',')
        self.buoy_seq_ids = None

        # Load data
        self.img_original = cv.imread(f'data/{self.data_file_name}', cv.IMREAD_GRAYSCALE)
        # Determine the shape of the original data
        self.original_height, self.original_width = self.img_original.shape[0:2]
        self.channel_size = self.original_width // 2

        # Separate the channels
        # The port side is stored flipped so that all distances increase to the right
        self.img_port_original = np.flip(self.img_original[:, :self.channel_size], axis=1)
        self.img_starboard_original = self.img_original[:, self.channel_size:]

        # Set the region of interest
        if start_ind is None:
            self.start_ind = 0
        else:
            self.start_ind = int(start_ind)

        if end_ind is None:
            self.end_ind = self.original_height
        else:
            self.end_ind = int(end_ind)

        if self.end_ind <= self.start_ind or self.start_ind >= self.end_ind:
            self.start_ind = 0
            self.end_ind = self.original_height

        if max_range_ind is None or max_range_ind <= 0 or max_range_ind >= self.channel_size:
            self.max_range_ind = self.channel_size
        else:
            self.max_range_ind = max_range_ind

        # Set working data - Extract area of interest
        # self.img = np.copy(self.img_original)[self.start_ind:self.end_ind, :]
        self.img_port = np.copy(self.img_port_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img_starboard = np.copy(self.img_starboard_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img = np.hstack((np.fliplr(self.img_port), self.img_starboard))
        self.img_height, self.img_width = self.img.shape[0:2]

        # Perform some processing
        self.img_canny = cv.Canny(self.img, self.canny_l_threshold, self.canny_h_threshold)

    def set_working_to_original(self):
        # Extract area of interest
        # self.img = np.copy(self.img_original)[self.start_ind:self.end_ind, :]
        self.img_port = np.copy(self.img_port_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img_starboard = np.copy(self.img_starboard_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img = np.hstack((np.fliplr(self.img_port), self.img_starboard))

    def row_fft(self, channel='PORT'):
        if channel.upper() == 'PORT':
            fft_input_data = self.img_port_original[self.start_ind:self.end_ind, :]
        else:
            fft_input_data = self.img_starboard_original[self.start_ind:self.end_ind, :]

        # %% fourier Analysis of image
        # fft_img = np.fft.fft2(fft_input_data)

        # Compute the Fourier transform along the rows using NumPy
        fft_rows = np.fft.fft(fft_input_data, axis=1)

        # Shift the zero-frequency component to the center of the spectrum
        fft_rows = np.fft.fftshift(fft_rows, axes=1)

        # Compute the magnitude spectrum (in dB) and plot the results
        power_spectrum = 20 * np.log10(np.abs(fft_rows))

        fft_fig, (ax1, ax2) = plt.subplots(1, 2)
        fft_fig.suptitle('FFT, row-wise')

        ax1.title.set_text('Input image')
        ax1.imshow(self.img_original[self.start_ind:self.end_ind, :])

        ax2.title.set_text('Frequency Domain')
        ax2.imshow(power_spectrum)

        fft_fig.show()

    def filter_median(self, kernel_size=5, show=False):
        if kernel_size not in [3, 5, 7, 9]:
            return

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # Perform median filter
        self.img_port = cv.medianBlur(self.img_port, kernel_size)
        self.img_starboard = cv.medianBlur(self.img_starboard, kernel_size)

        # update complete image
        img_filtered = np.hstack((np.fliplr(self.img_port), self.img_starboard))
        self.img = np.copy(img_filtered)

        if show:
            med_fig, (ax1, ax2) = plt.subplots(1, 2)
            med_fig.suptitle(f'Median filter, Kernel: {kernel_size}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(img_filtered)
            med_fig.show()

        cv.imwrite(f'data/{self.data_label}_med.png', img_filtered)

    def filter_gaussian(self, kernel_size=5, show=False):
        if kernel_size not in [3, 5, 7, 9]:
            return

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # Perform median filter
        self.img_port = cv.GaussianBlur(self.img_port, (kernel_size,kernel_size), 0)
        self.img_starboard = cv.GaussianBlur(self.img_starboard, (kernel_size,kernel_size), 0)
        img_filtered = np.hstack((np.fliplr(self.img_port), self.img_starboard))
        self.img = np.copy(img_filtered)

        if show:
            med_fig, (ax1, ax2) = plt.subplots(1, 2)
            med_fig.suptitle(f'Gaussian filter, Kernel: {kernel_size}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(img_filtered)
            med_fig.show()

        cv.imwrite(f'data/{self.data_label}_gauss.png', img_filtered)

    def gradient_cross_track(self, kernel_size=5, show=False):

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # X gradient
        grad_port = cv.Sobel(self.img_port, cv.CV_8U, 1, 0, ksize=kernel_size)
        grad_starboard = cv.Sobel(self.img_starboard, cv.CV_8U, 1, 0, ksize=kernel_size)
        complete_img = np.hstack((np.fliplr(grad_port), grad_starboard))

        if show:
            grad_fig, (ax1, ax2) = plt.subplots(1, 2)
            grad_fig.suptitle(f'Gradients, kernel: {kernel_size}')

            ax1.title.set_text('Image input')
            ax1.imshow(img_before)

            ax2.title.set_text('Image gradient')
            ax2.imshow(complete_img)

        cv.imwrite(f'data/{self.data_label}_grad.png', complete_img)

        return complete_img

    def canny_custom(self, m_size=5, g_size=5, s_size=5, l_threshold=100, h_threshold=200, show=True):
        """
        The

        :param g_size: size of gaussian filter, [3, 5, 7, 9]
        :param m_size: size of median filter, [3, 5, 7, 9]
        :param l_threshold: lower canny threshold
        :param h_threshold: upper canny threshold
        :return:
        """


        self.set_working_to_original()
        if m_size in [3, 5, 7, 9]:
            self.filter_median(m_size)

        if g_size in [3, 5, 7, 9]:
            self.filter_gaussian(g_size)

        if s_size in [3, 5, 7, 9]:
            dx_port = cv.Sobel(self.img_port, cv.CV_8U, 1, 0, ksize=s_size)
            dx_star = cv.Sobel(self.img_starboard, cv.CV_8U, 1, 0, ksize=s_size)

            dx_port[dx_port < 0] = 0
            dx_star[dx_star < 0] = 0

            dx = np.hstack((np.fliplr(dx_port), dx_star)).astype(np.int16)

            dy = cv.Sobel(self.img, cv.CV_16S, 0, 1, ksize=s_size)
            dy = np.zeros_like(dx)
            custom_canny = cv.Canny(dx=dx, dy=dy, threshold1=l_threshold, threshold2=h_threshold, L2gradient=True)
            cv.imwrite(f'data/canny_custom.png', custom_canny)

            if show:
                med_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                med_fig.suptitle(f'Custom canny, m_size: {m_size}  g_size: {g_size}, s_size: {s_size}')

                ax1.title.set_text('Input image')
                ax1.imshow(self.img)

                ax2.title.set_text('Custom canny results')
                img_color = np.dstack((self.img, self.img, self.img))
                img_color[custom_canny > 0] = [255, 0, 0]
                ax2.imshow(img_color)

                ax3.title.set_text('Gradient, dx, of input image')
                ax3.imshow(dx)
                med_fig.show()

            return custom_canny



    def combined_points(self):
        # Reset current image
        self.set_working_to_original()
        sss_analysis.filter_median(5, show=False)
        sss_analysis.filter_gaussian(5, show=False)

        gradient = sss_analysis.gradient_cross_track(5, show=False)

        img_combined = np.multiply(sss_analysis.img_canny, gradient)
        cv.imwrite(f'data/combined_canny_grad.png', img_combined)

    def mark_detections(self):
        if self.detections is None:
            return
        grey_img = np.copy(self.img_original)
        color_img = np.dstack((grey_img, grey_img, grey_img))
        for detection in detections:
            cv.circle(color_img, (detection[1], detection[0]), 30, (255, 0, 0), 10)

        marked_fig, ax1 = plt.subplots(1, 1)
        marked_fig.suptitle('Manually marked detections')
        ax1.imshow(color_img)
        marked_fig.show()

    def extract_seq_ids(self, detections):
        if self.detections is None or self.seq_ids is None:
            return
        self.buoy_seq_ids = []
        for detection in detections:
            self.buoy_seq_ids.append(self.seq_ids[detection[0]])


# %% Process SAM SSS
sss_analysis = process_sss(sam_file_name, seq_file_name, start_ind=0, end_ind=2500, max_range_ind=225)
sss_analysis.filter_median(0, show=True)
sss_analysis.filter_gaussian(0, show=True)
sss_analysis.gradient_cross_track(5, show=True)

canny_custom = sss_analysis.canny_custom(5, 5, 5, 175, 225)
sss_analysis.set_working_to_original()
image_raw = sss_analysis.img

# %%
lines = cv.HoughLines(canny_custom, rho=1, theta=np.pi / 180, threshold=25)
linesP = cv.HoughLinesP(canny_custom, rho=1, theta=np.pi / 180, threshold=25, minLineLength=25, maxLineGap=10)

image_raw = np.copy(sss_analysis.img)
image_color = cv.cvtColor(image_raw, cv.COLOR_GRAY2BGR)
# Draw the detected lines on the original image
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(image_color, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

# Display the image with detected lines
lines_fig = plt.figure()
plt.imshow(cv.cvtColor(image_color, cv.COLOR_BGR2RGB))

#sss_analysis.combined_points()
# sss_analysis.row_fft()

# %% Process boat sss
if process_boat_sss:
    boat_data = []
    with open(f'data/{boat_file_name}', 'rb') as f:
        reader = sllib.Reader(f)
        header = reader.header
        print(header.format)
        for frame in reader:
            raw_sonar = np.frombuffer(frame.packet, dtype=np.uint8)
            boat_data.append(raw_sonar)

    data_array = np.flipud(np.asarray(boat_data))
    data_len = data_array.shape[0]

    # Save sonar as jpg
    data_image = Image.fromarray(data_array)
    data_image.save(f'data/sss_boat_data_{data_len}.jpg')
