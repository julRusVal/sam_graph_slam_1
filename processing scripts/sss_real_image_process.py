#!/usr/bin/env python3

"""
Apply image processing techniques to assist in the detection of relevant features

This script is intended to process the real data collected at the algae farm.
"""

# %% Imports
import os

import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import signal
import sllib
from PIL import Image
from scipy.signal import medfilt
from skimage import measure

import math

# Choose detector as of 5/30 the detectors are the same but this could change!!!
# originally I just copied the detector script but later made a fork the smarc_perception
# from cp_detector_local import CPDetector, ObjectID  # My old version
from sss_object_detection.consts import ObjectID
from sss_object_detection.cpd_detector import CPDetector

# Import for basic time
import time

class process_sss:
    def __init__(self, data_file_name, seq_file_name, start_ind=None, end_ind=None, max_range_ind=None,
                 cpd_max_depth=None, cpd_ratio=None, flipping_regions=None, flip_original=False):

        """
        Currently processing is divided into pre- and post-processing. This division is kind of arbitrary.
        Roughly the pre-processing is concerned with traditional image processing to filter the image.
        Post-processing follows and attempts to employ some knowledge of the geometry to categorize detections.
        """

        # ===== Pre-processing Parameters =====
        self.canny_l_threshold = 100
        self.canny_h_threshold = 175
        self.canny_kernel_size = 5

        # Data file names
        self.data_file_name = data_file_name
        self.data_label = self.data_file_name.split(sep='.')[0]
        self.seq_file_name = seq_file_name
        self.seq_ids = np.genfromtxt(f'data/{self.seq_file_name}', delimiter=',')
        self.buoy_seq_ids = None

        # Data manipulations and manual detections
        self.flip_original = flip_original
        self.flipped_region = flipping_regions
        self.detections = None
        self.detections_mask = None

        # ===== post-processing Parameters =====
        self.post_rope_original = None  # These are set during post initialization
        self.post_buoy_original = None

        # These are modified by the post methods
        self.post_rope = None  # 2d representation of the rope detection, values > 0 indicate rope
        self.post_rope_inds_port = None
        self.post_rope_inds_star = None

        self.post_buoy = None
        self.post_buoy_centers = None

        self.post_use_port = None  # These will only modify post_rope and post_buoy not the originals
        self.post_use_star = None

        self.post_height, self.post_width = None, None

        # Final results
        # these are stored [ orig index | seq ID | orig cross index ]
        self.final_bouys = None
        self.final_ropes = None
        self.final_ropes_port = None
        self.final_ropes_star = None

        # ===== Start
        # Load data
        self.img_original = cv.imread(f'data/{self.data_file_name}', cv.IMREAD_GRAYSCALE)

        if flip_original:
            self.flip_data(self.flipped_region, show=False, overwrite_orig=True)

        # Determine the shape of the original data
        self.original_height, self.original_width = self.img_original.shape[0:2]
        self.channel_size = self.original_width // 2

        # Separate the channels
        # The port side is stored flipped so that all distances increase to the right
        self.img_port_original = np.fliplr(self.img_original[:, :self.channel_size])
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

        # Change point detector
        self.cpd_max_depth = cpd_max_depth
        self.cpd_ratio = cpd_ratio
        if cpd_ratio is None:
            self.detector = CPDetector()
        else:
            self.detector = CPDetector(min_mean_diff_ratio=self.cpd_ratio)

        self.nadir_color = np.array([255, 0, 0], dtype=np.uint8)
        self.rope_color = np.array([0, 255, 0], dtype=np.uint8)
        self.buoy_color = np.array([0, 0, 255], dtype=np.uint8)
        self.port_detections = None  # np.zeros(self.img_port_original.shape, dtype=np.uint8)
        self.starboard_detections = None  # np.zeros(self.img_starboard_original.shape, dtype=np.uint8)

        # Perform some processing
        self.img_canny = cv.Canny(self.img, self.canny_l_threshold, self.canny_h_threshold)

        # List of operations
        self.operations_list = []

        # Manual

    def set_working_to_original(self):
        # Extract area of interest
        # self.img = np.copy(self.img_original)[self.start_ind:self.end_ind, :]
        self.img_port = np.copy(self.img_port_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img_starboard = np.copy(self.img_starboard_original)[self.start_ind:self.end_ind, :self.max_range_ind]
        self.img = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # Clear operations list
        self.operations_list = []

    def flip_data(self, flipped_sections=None, show=False, overwrite_orig=False):
        '''
        Flips the sss data based on the specified flip regions, does not check for overlapping regions so be careful!

        :param show:
        :param flipped_sections: [[start_ind, end_ind],[...]...]
        :return:
        '''
        if flipped_sections is None:
            return

        if len(flipped_sections) <= 0:
            return

        local_img = np.copy(self.img_original)
        local_height = local_img.shape[0]

        seq_id_starts = []
        seq_id_ends = []

        for section in flipped_sections:
            if len(section) != 2:
                print('malformed flip request')
                continue
            # Check the bound of the flipping inds
            if section[0] < 0:
                flip_start_ind = 0
            else:
                flip_start_ind = section[0]

            if section[1] > local_height or section[1] <= flip_start_ind:
                flip_end_ind = local_height
            else:
                flip_end_ind = section[1]

            # Perform flip
            local_img[flip_start_ind:flip_end_ind, :] = local_img[flip_start_ind:flip_end_ind, ::-1]

            # Sequence ids
            # recording here to add to the manual detector, will flip seq ids in the ranges
            seq_id_starts.append(int(self.seq_ids[flip_start_ind]))
            seq_id_ends.append(int(self.seq_ids[flip_end_ind - 1]))

            if show:
                plt.imshow(local_img)
                plt.title('Flipped data results\n'
                          f'Starts: {seq_id_starts}\n'
                          f'Ends: {seq_id_ends}')
                plt.show()

            if overwrite_orig:
                self.img_original = np.copy(local_img)

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

    def filter_threshold(self, threshold=128, show=False):
        if threshold <= 0:
            return
        elif threshold > 255:
            threshold = 255

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # Perform median filter
        self.img_port[self.img_port >= threshold] = 255
        self.img_port[self.img_port < threshold] = 0
        self.img_starboard[self.img_starboard >= threshold] = 255
        self.img_starboard[self.img_starboard < threshold] = 0

        # update complete image
        img_filtered = np.hstack((np.fliplr(self.img_port), self.img_starboard))
        self.img = np.copy(img_filtered)

        if show:
            thresh_fig, (ax1, ax2) = plt.subplots(1, 2)
            thresh_fig.suptitle(f'Threshold filter, Threshold: {threshold}\n'
                                f'Previous Operations: {self.operations_list}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(img_filtered)
            thresh_fig.show()

        # Record the operation and save the output
        self.operations_list.append(f't_{threshold}')
        cv.imwrite(f'data/{self.data_label}_thresh.png', img_filtered)

        return img_filtered

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
            med_fig.suptitle(f'Median filter, Kernel: {kernel_size}\n'
                             f'Previous Operations: {self.operations_list}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(img_filtered)
            med_fig.show()

        # Record the operation and save the output
        self.operations_list.append(f'm_{kernel_size}')
        cv.imwrite(f'data/{self.data_label}_med.png', img_filtered)

    def filter_gaussian(self, kernel_size=5, show=False):
        if kernel_size not in [3, 5, 7, 9]:
            return

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # Perform median filter
        self.img_port = cv.GaussianBlur(self.img_port, (kernel_size, kernel_size), 0)
        self.img_starboard = cv.GaussianBlur(self.img_starboard, (kernel_size, kernel_size), 0)
        img_filtered = np.hstack((np.fliplr(self.img_port), self.img_starboard))
        self.img = np.copy(img_filtered)

        if show:
            med_fig, (ax1, ax2) = plt.subplots(1, 2)
            med_fig.suptitle(f'Gaussian filter, Kernel: {kernel_size}\n'
                             f'Previous Operations: {self.operations_list}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(img_filtered)
            med_fig.show()

        # Record the operation and save the output
        self.operations_list.append(f'gauss_{kernel_size}')
        cv.imwrite(f'data/{self.data_label}_gauss.png', img_filtered)

    def gradient_cross_track(self, kernel_size=5, show=False):

        # Before filter image
        img_before = np.hstack((np.fliplr(self.img_port), self.img_starboard))

        # X gradient
        grad_port = cv.Sobel(self.img_port, cv.CV_8U, 1, 0, ksize=kernel_size)
        grad_starboard = cv.Sobel(self.img_starboard, cv.CV_8U, 1, 0, ksize=kernel_size)
        complete_img = np.hstack((np.fliplr(grad_port), grad_starboard))

        self.img_port = np.copy(grad_port)
        self.img_starboard = np.copy(grad_starboard)
        self.img = np.copy(complete_img)

        if show:
            grad_fig, (ax1, ax2) = plt.subplots(1, 2)
            grad_fig.suptitle(f'Gradients, kernel: {kernel_size}\n'
                              f'Previous operations: {self.operations_list}')

            ax1.title.set_text('Image input')
            ax1.imshow(img_before)

            ax2.title.set_text('Image gradient')
            ax2.imshow(self.img)

        # Record the operation and save the output
        self.operations_list.append(f'grad_{kernel_size}')
        cv.imwrite(f'data/{self.data_label}_grad.png', complete_img)

        return complete_img

    def canny_custom(self, m_size=5, g_size=5, s_size=5, l_threshold=100, h_threshold=200, show=True):
        """
        The

        :param show:
        :param s_size: size of sobel kernel, [3, 5, 7, 9]
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
            dx_port = cv.Sobel(self.img_port, cv.CV_16S, 1, 0, ksize=s_size)
            dx_star = cv.Sobel(self.img_starboard, cv.CV_16S, 1, 0, ksize=s_size)

            # gradients along the each ping
            # Negative gradient is used for visualizing
            dx_port_neg = np.copy(dx_port)
            dx_star_neg = np.copy(dx_star)

            dx_port_neg[dx_port_neg > 0] = 0
            dx_star_neg[dx_star_neg > 0] = 0

            dx_port_neg = np.abs(dx_port_neg)
            dx_star_neg = np.abs(dx_star_neg)

            dx_neg = np.hstack((np.fliplr(dx_port_neg), dx_star_neg)).astype(np.int16)

            # Only the positive gradient is used for edge detection
            dx_port[dx_port < 0] = 0
            dx_star[dx_star < 0] = 0

            dx = np.hstack((np.fliplr(dx_port), dx_star)).astype(np.int16)

            # dy = cv.Sobel(self.img, cv.CV_16S, 0, 1, ksize=s_size)
            dy = np.zeros_like(dx)
            custom_canny = cv.Canny(dx=dx, dy=dy, threshold1=l_threshold, threshold2=h_threshold, L2gradient=True)
            cv.imwrite(f'data/canny_custom.png', custom_canny)

            if show:
                custom_canny_fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                custom_canny_fig.suptitle(f'Custom canny, m_size: {m_size}  g_size: {g_size}, s_size: {s_size}')

                ax1.title.set_text('Input image')
                ax1.imshow(self.img)

                ax2.title.set_text('Custom canny results')
                img_color = np.dstack((self.img, self.img, self.img))
                img_color[custom_canny > 0] = [255, 0, 0]
                ax2.imshow(img_color)

                ax3.title.set_text('Gradient, dx, of input image')
                ax3.imshow(dx)

                ax4.title.set_text('Gradient, dy, of input image')
                ax4.imshow(dx_neg)

                # custom_canny_fig.show()
                plt.show()

            return custom_canny, dx, dx_neg

    def canny_standard(self, m_size=5, l_threshold=100, h_threshold=200, show=True):
        """
        The

        :param show:
        :param m_size: size of median filter, [3, 5, 7, 9]
        :param l_threshold: lower canny threshold
        :param h_threshold: upper canny threshold
        :return:
        """

        self.set_working_to_original()

        if m_size in [3, 5, 7, 9]:
            self.filter_median(m_size)

        standard_canny = cv.Canny(self.img, threshold1=l_threshold, threshold2=h_threshold, L2gradient=True)
        cv.imwrite(f'data/canny_standard.png', standard_canny)

        if show:
            standard_canny_fig, (ax1, ax2) = plt.subplots(1, 2)
            standard_canny_fig.suptitle(f'standard canny, m_size: {m_size}')

            ax1.title.set_text('Input image')
            ax1.imshow(self.img)

            ax2.title.set_text('Custom canny results')
            img_color = np.dstack((self.img, self.img, self.img))
            img_color[standard_canny > 0] = [255, 0, 0]
            ax2.imshow(img_color)

            plt.show()

        return standard_canny

    def combined_points(self):
        # Reset current image
        self.set_working_to_original()
        sss_analysis.filter_median(5, show=False)
        sss_analysis.filter_gaussian(5, show=False)

        gradient = sss_analysis.gradient_cross_track(5, show=False)

        img_combined = np.multiply(sss_analysis.img_canny, gradient)
        cv.imwrite(f'data/combined_canny_grad.png', img_combined)

    def cpd_perform_detection(self, side=0):
        """
        Note: this has the ability to use multiple detection methods, defined in cp_detector_local.py
        """
        # Select which side to perform detection on
        if side == 0:
            img_side = self.img_port
        else:
            img_side = self.img_starboard

        # Allocate array for detections
        img_detections = np.zeros((img_side.shape[0], img_side.shape[1], 3), dtype=np.uint8)
        # img_detections = np.copy(img_side).astype(np.uint8)

        for i, ping in enumerate(img_side):

            if self.cpd_max_depth > 0:
                ping_results = self.detector.detect_rope_buoy(ping, self.cpd_max_depth)
            else:
                ping_results = self.detector.detect(ping)

            if ObjectID.NADIR in ping_results.keys():
                img_detections[i, ping_results[ObjectID.NADIR]['pos'], :] = self.nadir_color

            if ObjectID.ROPE in ping_results.keys():
                img_detections[i, ping_results[ObjectID.ROPE]['pos'], :] = self.rope_color

            if ObjectID.BUOY in ping_results.keys():
                img_detections[i, ping_results[ObjectID.BUOY]['pos'], :] = self.buoy_color

        # Save results
        if side == 0:
            self.port_detections = np.copy(img_detections[:, :, :])
        else:
            self.starboard_detections = np.copy(img_detections[:, :, :])

    def cpd_plot_detections(self):
        """
        Plot the detections
        """
        # Form yellow center band, helps to separate the port and starboard returns
        band = np.ones((self.img_height, 5, 3), dtype=np.uint8) * 255
        band[:, :, 2] = 0

        final = np.hstack((np.fliplr(self.port_detections),
                           band,
                           self.starboard_detections))

        plt.figure()
        plt.title('CPD detections')
        plt.imshow(final)
        plt.show()

    def show_detections(self, grad_results=None, canny_results=None):

        # convert cpd detections
        cpd_port_detections_mono = np.copy(self.port_detections.max(axis=2))
        cpd_star_detections_mono = np.copy(self.starboard_detections.max(axis=2))
        cpd_detections_mono = np.hstack((np.fliplr(cpd_port_detections_mono), cpd_star_detections_mono))

        if grad_results is None:
            grad_mono = np.zeros_like(self.img)
        else:
            grad_mono = np.copy(grad_results)

        if canny_results is None:
            canny_mono = np.zeros_like(self.img)
        else:
            canny_mono = np.copy(canny_results)

        combined_detections = np.dstack((cpd_detections_mono, grad_mono, canny_mono))

        plt.figure()
        plt.title('Combined detections\n'
                  'CPD: Red  -  Gradient: Green  -  Canny: Blue')
        plt.imshow(combined_detections)
        plt.show()

    def show_detections_overlay(self, grad_results=None, canny_results=None, show_manual=False):

        grey_img = np.copy(self.img_original)
        color_img = np.dstack((grey_img, grey_img, grey_img))

        # convert cpd detections
        cpd_port_detections_mono = np.copy(self.port_detections.max(axis=2))
        cpd_star_detections_mono = np.copy(self.starboard_detections.max(axis=2))
        cpd_detections_mono = np.hstack((np.fliplr(cpd_port_detections_mono), cpd_star_detections_mono))

        if grad_results is not None:
            grad_mono = np.zeros_like(self.img)
        else:
            grad_mono = np.copy(grad_results)

        if canny_results is None:
            canny_mono = np.zeros_like(self.img)
        else:
            canny_mono = np.copy(canny_results)

        combined_detections = np.dstack((cpd_detections_mono, grad_mono, canny_mono))

        plt.figure()
        plt.title('Combined detections\n'
                  'CPD: Red  -  Gradient: Green  -  Canny: Blue')
        plt.imshow(combined_detections)
        plt.show()

    def mark_manual_detections(self, manual_detections=None):
        """
        Mark manual detections
        :return:
        """
        self.detections = manual_detections

        if self.detections is None:
            return
        grey_img = np.copy(self.img_original)
        color_img = np.dstack((grey_img, grey_img, grey_img))
        for detection in detections:
            cv.circle(color_img, (detection[1], detection[0]), 15, (0, 255, 255), 5)  # Yellow

        marked_fig, ax1 = plt.subplots(1, 1)
        marked_fig.suptitle('Manually marked detections')
        ax1.imshow(color_img)
        # marked_fig.show()
        plt.show()

        cv.imwrite(f'data/{self.data_label}_marked.jpg', color_img)

    def extract_seq_ids(self, detections):
        if self.detections is None or self.seq_ids is None:
            return
        self.buoy_seq_ids = []
        for detection in detections:
            self.buoy_seq_ids.append(self.seq_ids[detection[0]])

    def show_thresholds(self, data, l_threshold, h_threshold, data_label='NO LABEL', reference_data=False):
        """
        plots original data, data less than l_threshold, data between l and h threshold, and greater than h_threshold

        :param data:
        :param l_threshold:
        :param h_threshold:
        :return:
        """

        low_data = np.zeros_like(data, np.uint8)
        mid_data = np.zeros_like(data, np.uint8)
        high_data = np.zeros_like(data, np.uint8)

        low_data[data < l_threshold] = 255
        mid_data[data >= l_threshold] = 255
        mid_data[data >= h_threshold] = 0
        high_data[data >= h_threshold] = 255

        thresholding_fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        thresholding_fig.suptitle(f'Thresholding of {data_label}\n'
                                  f'Low threshold: {l_threshold}  High threshold: {h_threshold}')

        ax1.title.set_text('Input image')
        if reference_data is True:
            ax1.imshow(self.img)
        else:
            ax1.imshow(data)

        ax2.title.set_text(f'data < {l_threshold}')
        ax2.imshow(low_data)

        ax3.title.set_text(f'{l_threshold} < data < {h_threshold}')
        ax3.imshow(mid_data)

        ax4.title.set_text(f'data > {h_threshold}')
        ax4.imshow(high_data)

        plt.show()

    def find_rising_edges(self, data, threshold, max_count=2, show=False, save_output=False):

        height, width = data.shape[:2]

        data_local = np.copy(data)
        data_threshold = np.zeros_like(data_local)

        # Threshold the data
        data_threshold[data_local > threshold] = 1

        data_port = np.fliplr(data_threshold[:, 0: width // 2])
        data_star = data_threshold[:, width // 2:]

        detections_port = np.zeros_like(data_port)
        detections_star = np.zeros_like(data_star)

        # this is mostly here to save the data and look at it in matlab
        port_detection_inds = np.zeros((height, max_count), dtype=np.int16)
        star_detection_inds = np.zeros((height, max_count), dtype=np.int16)

        for row in range(height):
            # === Port ===
            where_port = np.where((data_port[row, :-1] == 0) & (data_port[row, 1:] == 1))
            if len(where_port[0] > 0):
                max_where_ind = min(len(where_port[0]), max_count)
                for detect_num, index in enumerate(where_port[0][:max_where_ind]):
                    detections_port[row, index] = 255  # detect_num + 1

                    if detect_num < port_detection_inds.shape[1]:
                        port_detection_inds[row, detect_num] = index

            # Starboard
            where_star = np.where((data_star[row, :-1] == 0) & (data_star[row, 1:] == 1))
            if len(where_star[0] > 0):
                max_where_ind = min(len(where_star[0]), max_count)
                for detect_num, index in enumerate(where_star[0][:max_where_ind]):
                    detections_star[row, index] = 255  # detect_num + 1

                    if detect_num < star_detection_inds.shape[1]:
                        star_detection_inds[row, detect_num] = index

        if save_output:
            np.savetxt("data/port_detection_inds.csv", port_detection_inds, delimiter=",")
            np.savetxt("data/star_detection_inds.csv", star_detection_inds, delimiter=",")

        data_detections = np.hstack((np.fliplr(detections_port), detections_star))

        if show:
            grad_detect_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            grad_detect_fig.suptitle(f'Threshold Detections\n'
                                     f'Threshold: {threshold}  max: {max_count}')

            ax1.title.set_text('Input image')
            ax1.imshow(data_local)

            ax2.title.set_text(f'Image thresholded')
            ax2.imshow(data_threshold)

            ax3.title.set_text(f'Detections')
            ax3.imshow(data_detections)

            plt.show()

        return port_detection_inds, star_detection_inds

    def construct_template_kernel(self, size, sigma, feature_width):

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

    def post_initialize(self, rope_detections, buoy_detections, use_port=True, use_starboard=True):
        # Save originals
        self.post_rope_original = rope_detections
        self.post_buoy_original = buoy_detections

        self.post_height, self.post_width = self.post_rope_original.shape[:2]

        # Set which channels are used
        self.post_use_port = use_port
        self.post_use_star = use_starboard

        # Make copies to be process further
        self.post_rope = np.copy(self.post_rope_original)
        self.post_buoy = np.copy(self.post_buoy_original)

        # Remove channels if specified
        # Currently rope and buoy channels are ignored/used together
        if not self.post_use_port:
            self.post_rope[:, 0:self.post_width // 2] = 0
            self.post_buoy[:, 0:self.post_width // 2] = 0

        if not self.post_use_star:
            self.post_rope[:, self.post_width // 2:] = 0
            self.post_buoy[:, self.post_width // 2:] = 0

    def post_reset(self):
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        # Make copies to be process further
        self.post_rope = np.copy(self.post_rope_original)
        self.post_buoy = np.copy(self.post_buoy_original)

        # Remove channels if specified
        # Currently rope and buoy channels are ignored/used together
        if not self.post_use_port:
            self.post_rope[:, 0:self.post_width // 2] = 0
            self.post_buoy[:, 0:self.post_width // 2] = 0

        if not self.post_use_star:
            self.post_rope[:, self.post_width // 2:] = 0
            self.post_buoy[:, self.post_width // 2:] = 0

    def post_remove_ringing_rope(self, max_count=2, show_results=False):
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        # Threshold the data
        # 1 indicates a detection
        data_threshold = np.zeros_like(self.post_rope)
        data_threshold[self.post_rope > 0] = 1

        data_port = np.fliplr(data_threshold[:, 0: self.post_width // 2])
        data_star = data_threshold[:, self.post_width // 2:]

        filtered_port = np.copy(data_port)
        filtered_star = np.copy(data_star)

        # Previously detections are indicated by a value of 1
        # The sum of a row indicates the number of detections. This value is used to forma mask to apply to the data.
        # The mask is 1 if the detection count is <= max_count and 0 everywhere else.
        detection_count_port = np.sum(data_port, axis=1)
        detection_count_star = np.sum(data_port, axis=1)

        # This mask will need to be reshaped before it can be applied to the data
        mask_port = np.zeros_like(detection_count_port)
        mask_star = np.zeros_like(detection_count_star)

        mask_port[detection_count_port <= max_count] = 1
        mask_star[detection_count_star <= max_count] = 1

        mask_port = mask_port.reshape((-1, 1))
        mask_star = mask_star.reshape((-1, 1))

        filtered_port = np.multiply(filtered_port, mask_port)
        filtered_star = np.multiply(filtered_star, mask_star)

        # Form output array
        filtered_data = np.hstack((np.fliplr(filtered_port), filtered_star))

        if show_results:
            grad_detect_fig, (ax1, ax2) = plt.subplots(1, 2)
            grad_detect_fig.suptitle(f'Ringing Removal\n'
                                     f'Max detections: {max_count}')

            ax1.title.set_text('Input Image')
            ax1.imshow(self.post_rope)

            ax2.title.set_text(f'Output Image')
            ax2.imshow(filtered_data)

            plt.show()

        # Overwrite input array with output array
        self.post_rope = np.copy(filtered_data)

    def post_limit_range(self, min_index, max_index, show_results):
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        # From mask for single channel
        # Handle max range
        if max_index + 1 >= self.post_width // 2:
            half_range_mask = np.ones((1, self.post_width))
            max_index = (self.post_width // 2) - 1
        else:
            half_range_mask = np.ones((1, self.post_width // 2))
            half_range_mask[0, max_index:] = 0

        # handle min range
        min_skip_conditions = [min_index <= 0,
                               min_index >= max_index,
                               min_index > self.post_width // 2]
        if True in min_skip_conditions:
            min_index = 0
        else:
            half_range_mask[0, 0:min_index] = 0

        range_mask = np.hstack((np.fliplr(half_range_mask), half_range_mask))

        filtered_data = np.multiply(self.post_rope, range_mask)

        if show_results:
            # Boundary mask is just for visualization
            half_boundary_mask = np.zeros((1, self.post_width // 2))
            half_boundary_mask[0, min_index] = 255
            half_boundary_mask[0, max_index] = 255
            boundary_mask = np.hstack((np.fliplr(half_boundary_mask), half_boundary_mask))
            boundary_mask = np.repeat(boundary_mask, self.post_height, axis=0)

            # Raw image converted to RGB
            img_color = np.dstack((self.img, self.img, self.img))

            # Input
            img_rope_input = np.copy(img_color)
            img_rope_input[self.post_rope > 0] = self.rope_color

            # Output
            img_rope_output = np.copy(img_color)
            img_rope_output[boundary_mask > 0] = np.array([255, 255, 0], dtype=np.uint8)
            img_rope_output[filtered_data > 0] = self.rope_color

            # Form plot
            grad_detect_fig, (ax1, ax2) = plt.subplots(1, 2)
            grad_detect_fig.suptitle(f'Post: Range Limiting\n'
                                     f'Min index: {min_index}  Max index: {max_index}  (Yellow)')

            ax1.title.set_text('Input Image')
            ax1.imshow(img_rope_input)

            ax2.title.set_text(f'Output Image')
            ax2.imshow(img_rope_output)

            plt.show()

        # Overwrite input array with output array
        self.post_rope = np.copy(filtered_data)

    def post_exclude_rope_in_buoy_area(self, radius=0, show_results=False):
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        if radius == 0:
            print("Select non-zero radius!")
            return

        if self.post_buoy_centers is None:
            print("Find buoy centers first!")
            return

        buoy_mask = np.ones_like(self.post_rope)

        for center in self.post_buoy_centers:
            cv.circle(buoy_mask, (center[1], center[0]), radius, 0, -1)

        filtered_data = np.multiply(self.post_rope, buoy_mask)

        if show_results:
            # Raw image converted to RGB
            img_color = np.dstack((self.img, self.img, self.img))

            # Input
            img_rope_input = np.copy(img_color)
            img_rope_input[self.post_rope > 0] = self.rope_color

            # Output
            img_rope_output = np.copy(img_color)
            img_rope_output[filtered_data > 0] = self.rope_color

            # Form plot
            grad_detect_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            grad_detect_fig.suptitle(f'Post: Buoy masking\n'
                                     f'Radius: {radius}')

            ax1.title.set_text('Input Image')
            ax1.imshow(img_rope_input)

            ax2.title.set_text(f'Output Image')
            ax2.imshow(img_rope_output)

            ax3.title.set_text(f'Mask')
            ax3.imshow(buoy_mask * 255)

            plt.show()

        # Overwrite input array with output array
        self.post_rope = np.copy(filtered_data)

    def post_overlay_detections(self, ):
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        detection_value = 255  # 255

        # Raw image converted to RGB
        img_color = np.dstack((self.img, self.img, self.img))

        # Rope
        img_rope = np.copy(img_color)
        img_rope[self.post_rope > 0] = self.rope_color

        # Buoy
        img_buoy = np.copy(img_color)
        img_buoy[self.post_buoy > 0] = self.buoy_color

        # Combined
        img_combined = np.copy(img_color)
        img_combined[self.post_rope > 0] = self.rope_color
        img_combined[self.post_buoy > 0] = self.buoy_color

        detection_value = 255  # 255
        detections_combined = np.add(self.post_rope, self.post_buoy)
        combined_color = np.add(self.rope_color, self.buoy_color)

        img_combined[detections_combined > detection_value] = combined_color

        grad_detect_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        grad_detect_fig.suptitle(f'Post: Detection overlay')

        ax1.title.set_text('Rope detections')
        ax1.imshow(img_rope)

        ax2.title.set_text('Buoy detections')
        ax2.imshow(img_buoy)

        ax3.title.set_text('Combined detections')
        ax3.imshow(img_combined)
        if self.post_buoy_centers is not None:
            ax3.plot(self.post_buoy_centers[:, 1], self.post_buoy_centers[:, 0],
                     'ro')  # Note: Y coordinates come before X coordinates

        plt.show()

    def post_overlay_detections_pretty(self, ):
        """
        This method is for generating images for icra paper

        Note: only port are shown due to the survey path taken

        :return:
        """
        circ_rad = 10
        circ_thick = 2

        # start_ind = 4800  # values for image in icra submission
        # end_ind = 5650

        start_ind = 0
        end_ind = int(self.img.shape[0] - 1)

        end_ind_column = int(self.img.shape[1]//2)

        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        # Raw image converted to RGB
        img_color = np.dstack((self.img, self.img, self.img))

        # Combined
        img_combined = np.copy(img_color)
        img_combined[self.post_rope > 0] = self.rope_color

        for center in self.post_buoy_centers:
            # Color is reversed because cv assumes bgr??
            color = self.buoy_color[::-1]  # (color[0], color[1], color[2])
            color_tup = (255, 0, 0)
            cv.circle(img_combined, (center[1], center[0]),
                      radius=circ_rad, color=color_tup, thickness=circ_thick)

        grad_detect_fig, (ax1, ax2) = plt.subplots(1, 2)
        grad_detect_fig.suptitle(f'Post: Final Detection overlay')

        ax1.title.set_text('Original')
        ax1.imshow(img_color)

        ax2.title.set_text('Combined detections')
        ax2.imshow(img_combined)

        plt.show()

        # Save
        print("Saving detection for ICRA paper")
        image_path = f"data/detector_output_original_{start_ind}_{end_ind}.png"
        cv2.imwrite(image_path, img_color[start_ind:end_ind, :end_ind_column, ::-1])

        image_path = f"data/detector_output_marked_{start_ind}_{end_ind}.png"
        cv2.imwrite(image_path, img_combined[start_ind:end_ind, :end_ind_column, ::-1])

    def post_interleave_detection_inds(self, channel_data, channel_id, med_filter_kernel=0, show_results=False):
        """
        column 0 is the first detection and column 1 is the 2nd detection.
        The columns will be interleaved together for further filtering.

        Also can perform median filtering

        N is height of sonar data
        M is the number of rising edges that are used, most likely 2

        :param channel_data: NxM array of detection inds
        :return:
        """
        if self.post_rope_original is None:
            print("Post processing was not initialized!")
            return

        data_local = np.copy(channel_data)
        m = data_local.shape[1]

        # When only a single detection is returned, it's value is copied to the 2nd detection.
        # Otherwise, a bunch of false negatives will be introduced
        indices_to_update = (data_local[:, 0] > 0) & (data_local[:, 1] == 0)
        data_local[indices_to_update, 1] = data_local[indices_to_update, 0]

        # Reshape the array to interleave
        data_interleaved = data_local.reshape(-1, 1)
        interleaved_height, interleaved_width = data_interleaved.shape[0:2]

        x_values = np.arange(0, interleaved_height)

        # Perform median filtering
        if med_filter_kernel > 3 and not med_filter_kernel % 2 == 0:
            data_interleaved_med = medfilt(data_interleaved.squeeze(), med_filter_kernel)
        else:
            data_interleaved_med = data_interleaved

        # Store
        # inds are stored de-interleaved so the length should match the original height
        if channel_id.casefold() == 'port':
            self.post_rope_inds_port = data_interleaved_med[::m]
        elif 'star' in channel_id.casefold():
            self.post_rope_inds_star = data_interleaved_med[::m]
        else:
            channel_id = 'INVALID'
            print("Improper channel ID given!")
        if show_results:
            # Generate raw sonar image of correct size
            img_color = np.dstack((self.img, self.img, self.img))
            img_color = np.repeat(img_color, 2, axis=0)

            # Form plot
            interleaved_fig, (ax1, ax2) = plt.subplots(1, 2)
            interleaved_fig.suptitle(f'Post: Interleaved results - {channel_id.upper()}\n'
                                     f'Median Filter size: {med_filter_kernel}')

            ax1.title.set_text('Raw Sensor Data')

            ax1.imshow(img_color, aspect=.05)

            ax2.title.set_text(f'Interleaved Indices')
            ax2.plot(data_interleaved, x_values, label="original")
            ax2.plot(data_interleaved_med, x_values, label="Median Filtered")

            plt.legend()
            ax2.invert_yaxis()
            ax2.invert_xaxis()

            plt.show()

        return data_interleaved, data_interleaved_med

    def post_plot_inds(self, channel_id='port'):
        #
        filtered_height = self.post_rope.shape[0]
        if channel_id.casefold() == 'port':
            filtered = self.post_rope_inds_port
        elif 'star' in channel_id.casefold():
            filtered = self.post_rope_inds_star
        else:
            print("Improper channel ID given!")
            return

        # Test 1
        # Find the array distance from the previous non zero
        non_zero_results = np.zeros((filtered_height, 1))
        last_non_zero = -1

        for h_i in range(filtered_height):
            current_value = filtered[h_i]
            #
            if last_non_zero == -1:  # no previous value is non zero
                non_zero_results[h_i] = 0

            else:
                non_zero_results[h_i] = h_i - last_non_zero

            #
            current_value = filtered[h_i]
            if current_value != 0:
                last_non_zero = h_i

        # Test 2
        # consecutive zeros
        zero_results = np.ones((filtered_height, 1)) * -1  # Initialized to -1

        for h_i in range(filtered_height):
            if zero_results[h_i] != -1:
                continue

            if filtered[h_i] != 0:
                zero_results[h_i] = 0

            else:
                if h_i == filtered_height - 1:  # check if we're at the end
                    zero_results[h_i] = 1
                else:
                    # Look for next non zero
                    count = 0
                    start_i = h_i
                    end_i = -1
                    for i in range(start_i, filtered_height):
                        if filtered[i] == 0:
                            count += 1
                        else:
                            end_i = i
                            break

                    # check if loop reached the end of the data set
                    if end_i == -1:
                        end_i = filtered_height

                    zero_results[start_i:end_i] = count

        # X values for plotting
        x_values = np.arange(0, filtered_height)

        # Generate raw sonar image of correct size
        img_color = np.dstack((self.img, self.img, self.img))
        # img_color = np.repeat(img_color, 2, axis=0)

        # Form plot
        interleaved_fig, (ax1, ax2) = plt.subplots(1, 2)
        interleaved_fig.suptitle(f'Post: De-interleaved results - {channel_id.upper()}')

        ax1.title.set_text('Raw Sensor Data')

        ax1.imshow(img_color, aspect=.05)

        ax2.title.set_text(f'Interleaved Indices')
        ax2.plot(filtered, x_values, label="Median Filtered")
        ax2.plot(zero_results, x_values, label="zero lengths")

        plt.legend()
        ax2.invert_yaxis()
        ax2.invert_xaxis()

        plt.show()

    def post_interleaved_to_2d(self, interleaved_port=None, interleaved_star=None):
        new_port = np.zeros((self.post_height, self.post_width // 2))
        new_star = np.zeros((self.post_height, self.post_width // 2))

        # Port
        if interleaved_port is not None:
            port_interleaved = interleaved_port[::2]

            # Check dimensions
            if port_interleaved.shape[0] != self.post_height:
                print("post_interleaved_to_2d(): dimension mismatch")
                return

            for i in range(self.post_height):
                detection_ind = port_interleaved[i]

                if detection_ind != 0 and detection_ind < self.post_width // 2:
                    new_port[i, detection_ind] = 255

        # Starboard
        if interleaved_star is not None:
            star_interleaved = interleaved_star[::2]

            # Check dimensions
            if star_interleaved.shape[0] != self.post_height:
                print("post_interleaved_to_2d(): dimension mismatch")
                return

            for i in range(self.post_height):
                detection_ind = star_interleaved[i]

                if detection_ind != 0 and detection_ind < self.post_width // 2:
                    new_star[i, detection_ind] = 255

        # New
        new = np.hstack((np.fliplr(new_port), new_star))
        self.post_rope = np.copy(new)

    def post_find_buoy_centers(self, min_region_size, exclusion_zone=None):
        labeled_array, num_labels = measure.label(self.post_buoy, connectivity=2, return_num=True)
        region_properties = measure.regionprops(labeled_array)

        valid_regions = [region for region in region_properties if region.area >= min_region_size]

        region_centers = np.array([region.centroid for region in valid_regions]).astype(int)

        self.post_buoy_centers = region_centers
        # Filter to remove buoys that are too close, favor closer detections
        # This introduces some assumptions on the depth of the auv
        if exclusion_zone is None:
            self.post_buoy_centers = region_centers
            return region_centers
        valid = []
        skip = []
        for i in range(region_centers.shape[0]):
            if i in skip:
                continue
            y_coord = region_centers[i, 0]
            x_coord = region_centers[i, 1]

            if x_coord < self.post_width // 2:
                port = True
            else:
                port = False

            if port:
                test_1 = np.abs(y_coord - region_centers[:, 0]) < exclusion_zone
                test_2 = region_centers[:, 1] < self.post_width // 2
                inds = np.where(test_1 & test_2)[0]

                if len(inds) == 1:
                    valid.append(i)
                    continue

                max_inds_ind = np.argmax(region_centers[inds, 1])
                # select the max for port
                valid.append(inds[max_inds_ind])
                # there is no need to consider centers twice
                for ind in inds:
                    skip.append(ind)

            else:
                inds = np.where(abs(y_coord - region_centers[:, 0]) < exclusion_zone &
                                region_centers[:, 1] >= self.post_width // 2)[0]

                if len(inds) == 1:
                    valid.append(i)
                    continue

                min_inds_ind = np.argmin(region_centers[inds, 1])
                # select the min for starboard
                valid.append(inds[min_inds_ind])
                # there is no need to consider centers twice
                for ind in inds:
                    skip.append(ind)

        new_region_centers = region_centers[valid, :]
        self.post_buoy_centers = new_region_centers
        return new_region_centers

    def post_find_buoy_offsets(self, window_size=55, plot=False):
        """

        :param window_size:
        :return:
        """

        if self.post_buoy_centers is None:
            print("Find buoys centers first!")
            return

        # Port stuff
        port_detections_indices = np.where(self.post_buoy_centers[:, 1] < self.post_width // 2)[0]
        port_detections = self.post_buoy_centers[port_detections_indices, :]
        port_detections_count = port_detections.shape[0]
        port_leading_windows = np.zeros((port_detections_count, 1))
        port_trailing_windows = np.zeros((port_detections_count, 1))

        # Convert rope detections to absolute coords
        # here absolute means a 2d array with width self.post_width
        port_rope_inds_abs = (self.post_width // 2 - 1) - self.post_rope_inds_port[:]

        # This is the raw seq IDs, being used as the independent variable
        x_values = self.post_buoy_centers[port_detections_indices, 1]

        for i in range(port_detections_count):
            # Find indices for current window
            center = int(port_detections[i, 0])
            leading_stop = int(center + 1)
            leading_start = int(max(0, leading_stop - window_size))

            trailing_start = center
            trailing_stop = int(min(self.post_height, trailing_start + window_size))

            current_leading_window = port_rope_inds_abs[leading_start: leading_stop]
            current_trailing_window = port_rope_inds_abs[trailing_start: trailing_stop]

            leading_median = np.median(current_leading_window)
            trailing_median = np.median(current_trailing_window)

            port_leading_windows[i] = leading_median
            port_trailing_windows[i] = trailing_median

        # TODO More work to use the rope detections to determine the buoy offset

        if plot:
            # Form plot
            interleaved_fig, (ax1) = plt.subplots(1, 1)
            interleaved_fig.suptitle(f'Post: Buoy Offsets - PORT\n'
                                     'WORK IN PROGRESS')

            # ax1.title.set_text('Original detections')
            #
            # ax1.plot(port_detections[:, 0],
            #          port_detections[:, 1],
            #          label="Detection")
            #
            # ax1.plot(port_detections[:, 0],
            #          port_leading_windows[:],
            #          label="leading")

            labels = [str(int(num)) for num in port_detections[:, 0]]
            x = np.arange(len(labels))  # Generate x-axis ticks

            ax1.bar(x - 0.2, port_leading_windows[:, 0], width=0.2, label='Leading')
            ax1.bar(x, port_detections[:, 1], width=0.2, label='Detection')
            plt.bar(x + 0.2, port_trailing_windows[:, 0], width=0.2, label='Trailing 3')

            plt.xlabel('Detection')
            plt.ylabel('Values')
            # plt.title('Bar Graph of Three Arrays')
            plt.xticks(x, labels)
            plt.legend()
            plt.show()

    def post_final_buoys(self, plot=False):
        # The input is stored as [ orig index | truncated cross index ]
        # The output is stored as [ orig index | seq ID | orig cross index ]

        buoy_count = self.post_buoy_centers.shape[0]
        self.final_bouys = np.zeros((buoy_count, 3), int)

        # convert to int
        buoy_centers_int = self.post_buoy_centers.astype(int)
        self.final_bouys[:, 0] = buoy_centers_int[:, 0]
        self.final_bouys[:, 2] = buoy_centers_int[:, 1]

        # offset to account for post being truncated
        if self.post_width != self.original_width:
            size_dif = self.original_width - self.post_width
            offset = size_dif // 2
            # Apply
            self.final_bouys[:, 2] = self.final_bouys[:, 2] + offset

        # find seq IDs
        relevant_seq_ids = self.seq_ids[self.final_bouys[:, 0]]

        self.final_bouys[:, 1] = relevant_seq_ids.astype(int)

        np.savetxt("data/image_process_buoys.csv", self.final_bouys, delimiter=",")
        print("Buoy detections saved")

        if plot:
            img_color = np.dstack((self.img_original, self.img_original, self.img_original))

            final_buoy_fig, (ax1, ax2) = plt.subplots(1, 2)
            final_buoy_fig.suptitle(f'Post: Final buoy detections')

            ax1.title.set_text('Original Sonar')
            ax1.imshow(img_color)

            ax2.title.set_text('Buoy detections')
            ax2.imshow(img_color)
            ax2.plot(self.final_bouys[:, 2], self.post_buoy_centers[:, 0],
                     'ro')  # Note: Y coordinates come before X coordinates

            plt.show()

    def post_final_ropes(self, plot=False):
        # The output is stored as [ orig index | seq ID | range index ]

        # Port
        port_valid_inds = np.nonzero(self.post_rope_inds_port[:])[0]
        if len(port_valid_inds) > 0:
            port_seq_ids = self.seq_ids[port_valid_inds]
            port_valid = self.post_rope_inds_port[port_valid_inds]

            port_length = port_valid_inds.shape[0]
            self.final_ropes_port = np.zeros((port_length, 3))

            self.final_ropes_port[:, 0] = port_valid_inds[:]
            self.final_ropes_port[:, 1] = port_seq_ids[:]
            self.final_ropes_port[:, 2] = port_valid[:]

            np.savetxt("data/image_process_ropes_port.csv", self.final_ropes_port, delimiter=",")

        # Star
        star_valid_inds = np.nonzero(self.post_rope_inds_star[:])[0]
        if len(star_valid_inds) > 0:
            star_seq_ids = self.seq_ids[star_valid_inds]
            star_valid = self.post_rope_inds_star[star_valid_inds]

            star_length = star_valid_inds.shape[0]
            self.final_ropes_star = np.zeros((star_length, 3))

            self.final_ropes_star[:, 0] = star_valid_inds[:]
            self.final_ropes_star[:, 1] = star_seq_ids[:]
            self.final_ropes_star[:, 2] = star_valid[:]

            np.savetxt("data/image_process_ropes_star.csv", self.final_ropes_star, delimiter=",")

        print("Rope detections saved")


if __name__ == '__main__':
    # Parameters
    # process_sam_sss = False
    # ===== Data preprocessing and manual detections =====
    show_flipping = False
    show_manual_detection = False
    perform_flipping = True
    # ===== Gradient method =====
    perform_grad_method = True
    grad_show_intermediate = False
    grad_show = False
    grad_med_size = 7
    grad_gauss_size = 5
    grad_grad_size = 5

    # ===== Canny edge detector =====
    perform_custom_canny = True
    custom_canny_show = False
    canny_med_size = 7  # both
    canny_gauss_size = 5  # both
    canny_sobel_size = 5  # both
    canny_l_thresh = 175  # rope
    canny_h_thresh = 225  # rope
    do_blob = False  # buoy
    do_template = True  # buoy
    show_template = False  # buoy
    show_template_raw = False  # buoy
    template_size = 21  # buoy
    template_sigma = 2  # buoy
    template_feature_width = 10  # buoy
    template_l_threshold = 1000  # buoy
    template_h_threshold = 3000  # buoy

    # ===== Standard canny detector =====
    perform_standard_canny = False
    standard_canny_show = False
    standard_canny_med_size = 7
    standard_canny_l_thresh = 175
    standard_canny_h_thresh = 225

    # ===== CPD method =====
    perform_cpd = False
    cpd_show = False
    cpd_max_depth = 100
    cpd_ratio = 1.1  # 1.55 was default
    cpd_med_size = 0

    # ===== Combined detector output =====
    perform_combined_detector = False

    # ===== Post ====
    perform_post = True
    ringing_max_count = 2
    ringing_show = False
    limiting_min = 30
    limiting_max = 100
    limiting_show = False
    inds_med_size = 55  # 45 worked well
    show_final_post = False

    buoy_center_size_threshold = 5
    buoy_center_exclusion_zone = 25

    rope_exclusion_size = 25

    show_final_inds_port = False

    start_time = time.time()

    # ===== Boat sonar data =====
    process_boat_sss = False

    sam_file_name = 'sss_data_7608.jpg'
    seq_file_name = 'sss_seqs_7608.csv'  # this data is produced by the sss_raw_saver.py
    boat_file_name = 'Sonar_2023-05-03_20.51.26.sl2'

    start_ind = 0  # 2000  # 3400  # 3400  # 6300
    end_ind = 0  # 7608# 6000  # 5700  # 4600  # 7200
    max_range_ing = 175

    # detections = [[7106, 1092], [6456, 1064],
    #               [5570, 956], [4894, 943],
    #               [4176, 956], [3506, 924],
    #               [2356, 911], [1753, 949],
    #               [1037, 941], [384, 943]]

    detections = [[7096, 907], [6452, 937],
                  [5570, 956], [4894, 943],
                  [4176, 956], [3506, 924],
                  [2356, 911], [1753, 949],
                  [1037, 941], [384, 943]]

    flipped_regions = [[5828, -1]]

    # %% Process SAM SSS
    sss_analysis = process_sss(sam_file_name, seq_file_name,
                               start_ind=start_ind, end_ind=end_ind,
                               max_range_ind=max_range_ing,
                               cpd_max_depth=cpd_max_depth, cpd_ratio=cpd_ratio,
                               flipping_regions=flipped_regions,
                               flip_original=perform_flipping)

    # if show_flipping:
    #     sss_analysis.flip_data(flipped_sections=flipped_regions)

    if show_manual_detection:
        sss_analysis.mark_manual_detections(detections)

    if perform_grad_method:
        sss_analysis.filter_median(grad_med_size, show=grad_show_intermediate)
        sss_analysis.filter_gaussian(grad_gauss_size, show=grad_show_intermediate)
        grad_output = sss_analysis.gradient_cross_track(grad_grad_size, show=grad_show_intermediate)
        grad_method_results = sss_analysis.filter_threshold(threshold=200, show=grad_show)

    else:
        grad_method_results = None

    if perform_custom_canny:
        canny_custom, custom_dx, custom_dx_neg = sss_analysis.canny_custom(canny_med_size,
                                                                           canny_gauss_size,
                                                                           canny_sobel_size,
                                                                           canny_l_thresh, canny_h_thresh,
                                                                           show=custom_canny_show)
        # sss_analysis.show_thresholds(custom_dx, 100, 1000, 'Custom Dx Positive')
        # sss_analysis.show_thresholds(custom_dx_neg, 100, 1000, 'Custom Dx Negative')

        # ==== find the first and second rising edges
        # sss_analysis.find_rising_edges(canny_custom, 150, 2, True)

        # ===== New =====
        if do_blob:
            # Create blob detector
            params = cv.SimpleBlobDetector_Params()

            # Set parameters
            params.minThreshold = 140
            params.maxThreshold = 500
            params.filterByArea = True
            params.minArea = 15
            params.filterByCircularity = False
            params.minCircularity = 0.7

            # Create detector with parameters
            detector = cv.SimpleBlobDetector_create(params)

            # Detect blobs
            keypoints = detector.detect(custom_dx)

            # Draw blobs on the image
            image_with_keypoints = cv.drawKeypoints(custom_dx, keypoints, np.array([]), (0, 0, 255),
                                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Plot the original image and the image with keypoints
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(custom_dx)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(cv.cvtColor(image_with_keypoints, cv.COLOR_BGR2RGB))
            axs[1].set_title('Image with Keypoints')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

        if do_template:
            matching_image = np.copy(custom_dx).astype(np.float32)
            if template_size % 2 == 0:
                template_size += 1
            template = sss_analysis.construct_template_kernel(template_size, template_sigma, template_feature_width)
            template = template.astype(np.float32)
            template_result = cv2.matchTemplate(matching_image, template, cv2.TM_CCOEFF)

            # matchTemplate results will need padding
            # W' = W - w + 1, where W': final width, W: initial width, w: template width
            # Above holds for the height as well
            pad_size = template_size // 2
            template_result = np.pad(template_result, pad_size)

            if show_template:
                sss_analysis.show_thresholds(template_result,
                                             template_l_threshold,
                                             template_h_threshold,
                                             'Gradient Template',
                                             reference_data=not show_template_raw)

                plt.imshow(template)
        else:
            template_result = None

    else:
        canny_custom = None
        template_result = None

    if perform_standard_canny:
        standard_canny = sss_analysis.canny_standard(m_size=standard_canny_med_size,
                                                     l_threshold=standard_canny_l_thresh,
                                                     h_threshold=standard_canny_h_thresh,
                                                     show=standard_canny_show)
    else:
        standard_canny = None

    if perform_cpd:
        sss_analysis.set_working_to_original()
        sss_analysis.filter_median(cpd_med_size, show=cpd_show)
        sss_analysis.cpd_perform_detection(0)
        sss_analysis.cpd_perform_detection(1)
        if cpd_show:
            sss_analysis.cpd_plot_detections()

    if perform_combined_detector:
        sss_analysis.show_detections(grad_results=grad_method_results,
                                     canny_results=canny_custom)

    pre_end_time = time.time()
    post_end_time = None

    while perform_post:
        # check if the needed pre-processing has been saved and
        if canny_custom is None:
            print("Post failed: canny_custom not found!")
            break

        if template_result is None:
            print("Post failed: tem not found!")
            break

        # Thresholding of 'raw' detections
        # ROPE: Thresholding is applied during canny
        # Buoy: template_results requires thresholding
        template_result_threshold = np.zeros_like(template_result, np.uint8)
        template_result_threshold[template_result >= template_h_threshold] = 255

        sss_analysis.post_initialize(rope_detections=canny_custom,
                                     buoy_detections=template_result_threshold,
                                     use_port=True,
                                     use_starboard=False)

        # Process buoy detections
        sss_analysis.post_find_buoy_centers(min_region_size=buoy_center_size_threshold,  # 5
                                            exclusion_zone=buoy_center_exclusion_zone)

        # Work in progress,
        # sss_analysis.post_find_buoy_offsets(window_size=55, plot=True)

        # Process rope detections
        """
        Rope detection is carried out in multiple steps
        - Ringing removal
        - Range limiting
        
        """
        # Remove ringing
        # Useful to perform before the limiting the range
        sss_analysis.post_remove_ringing_rope(max_count=ringing_max_count,
                                              show_results=ringing_show)

        # Enforce max detection range
        sss_analysis.post_limit_range(min_index=limiting_min,
                                      max_index=limiting_max,
                                      show_results=limiting_show)

        sss_analysis.post_exclude_rope_in_buoy_area(radius=rope_exclusion_size, show_results=False)

        post_port_detection_inds, post_star_detection_inds = sss_analysis.find_rising_edges(data=sss_analysis.post_rope,
                                                                                            threshold=0,
                                                                                            max_count=2,
                                                                                            show=False,
                                                                                            save_output=False)

        port_inter_raw, port_inter_med = sss_analysis.post_interleave_detection_inds(
            channel_data=post_port_detection_inds,
            channel_id='port',
            med_filter_kernel=inds_med_size,
            show_results=False)

        star_inter_raw, star_inter_med = sss_analysis.post_interleave_detection_inds(
            channel_data=post_star_detection_inds,
            channel_id='star',
            med_filter_kernel=inds_med_size,
            show_results=False)
        # sss_analysis.post_de_interleave(interleaved_med)

        sss_analysis.post_interleaved_to_2d(interleaved_port=port_inter_med,
                                            interleaved_star=None)

        if show_final_inds_port:
            sss_analysis.post_plot_inds(channel_id='port')

        # sss_analysis.post_find_buoy_offsets(window_size=55, plot=True)

        sss_analysis.post_final_buoys(plot=False)
        sss_analysis.post_final_ropes(plot=False)

        post_end_time = time.time()

        if show_final_post:
            sss_analysis.post_overlay_detections()
            sss_analysis.post_overlay_detections_pretty()

        # Generate

        break

    # Show timings
    pre_time = pre_end_time - start_time

    if post_end_time is not None:
        post_time = post_end_time - pre_end_time
        complete_time = post_end_time - start_time

    else:
        post_time = 0
        complete_time = pre_time

    size = sss_analysis.end_ind - sss_analysis.start_ind - 1
    print(f"Processed count : {size}")
    print(f"Pre-processing time: {pre_time}")
    print(f"Post-processing time: {post_time}")
    print(f"Complete time: {complete_time}")
    # %%
    # lines = cv.HoughLines(canny_custom, rho=1, theta=np.pi / 180, threshold=25)
    # linesP = cv.HoughLinesP(canny_custom, rho=1, theta=np.pi / 180, threshold=25, minLineLength=25, maxLineGap=10)
    #
    # image_raw = np.copy(sss_analysis.img)
    # image_color = cv.cvtColor(image_raw, cv.COLOR_GRAY2BGR)
    # # Draw the detected lines on the original image
    # if linesP is not None:
    #         for i in range(0, len(linesP)):
    #             l = linesP[i][0]
    #             cv.line(image_color, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    #
    # # Display the image with detected lines
    # lines_fig = plt.figure()
    # plt.imshow(cv.cvtColor(image_color, cv.COLOR_BGR2RGB))

    # sss_analysis.combined_points()
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
