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
from cp_detector_local import CPDetector, ObjectID
import math


class process_sss:
    def __init__(self, data_file_name, seq_file_name, start_ind=None, end_ind=None, max_range_ind=None,
                 cpd_max_depth=None, cpd_ratio=None, flipping_regions=None, flip_original=False):
        # Parameters
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

        # Manul

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
                ping_results = self.detector.detect_rope(ping, self.cpd_max_depth)
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


if __name__ == '__main__':
    # Parameters
    # process_sam_sss = False
    # ===== Data preprocessing and manual detections =====
    show_flipping = False
    show_manual_detection = False
    perform_flipping = True
    # ===== Gradient method =====
    perform_grad_method = False
    grad_med_size = 7
    grad_gauss_size = 5
    grad_grad_size = 5
    grad_show = True
    # ===== Canny edge detector =====
    perform_canny = True
    canny_med_size = 7
    canny_gauss_size = 5
    canny_sobel_size = 5
    canny_l_thresh = 175
    canny_h_thresh = 225
    canny_show = True
    # ===== CPD method =====
    perform_cpd = True
    cpd_max_depth = 100
    cpd_ratio = 1.25  # 1.55 was default
    cpd_med_size = 0
    cpd_show = True
    # ===== Combined detector output =====
    perform_combined_detector = True
    # ===== Boat sonar data =====
    process_boat_sss = False

    sam_file_name = 'sss_data_7608.jpg'
    seq_file_name = 'sss_seqs_7608.csv'  # this data is produced by the sss_raw_saver.py
    boat_file_name = 'Sonar_2023-05-03_20.51.26.sl2'

    start_ind = 3400  # 3400  # 6300
    end_ind = 5700  # 4600  # 7200
    max_range_ing = 225

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
        sss_analysis.filter_median(grad_med_size, show=grad_show)
        sss_analysis.filter_gaussian(grad_gauss_size, show=grad_show)
        sss_analysis.gradient_cross_track(grad_grad_size, show=grad_show)
        grad_method_results = sss_analysis.filter_threshold(threshold=200, show=grad_show)
    else:
        grad_method_results = None

    if perform_canny:
        canny_custom = sss_analysis.canny_custom(canny_med_size,
                                                 canny_gauss_size,
                                                 canny_sobel_size,
                                                 canny_l_thresh, canny_h_thresh,
                                                 show=canny_show)
    else:
        canny_custom = None

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
