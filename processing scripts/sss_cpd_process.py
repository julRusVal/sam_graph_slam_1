#!/usr/bin/env python3

"""
Apply Change point detection to saved side scans
"""

# %% Imports
from cp_detector_local import CPDetector, ObjectID
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# %% Classes
class sss_detector:
    def __init__(self, file_name):
        # SSS parameters
        self.resolution = 0.5

        # Img info
        self.file_name = file_name
        self.img = cv.imread(file_name)  # cv.IMREAD_GRAYSCALE)
        self.height = self.img.shape[0]
        self.scan_width = self.img.shape[1] // 2
        self.port = np.flip(self.img[:, :self.scan_width], axis=1)
        self.starboard = self.img[:, self.scan_width:]
        self.port_detections = np.zeros(self.port.shape, dtype=np.uint8)
        self.starboard_detections = np.zeros(self.port.shape, dtype=np.uint8)

        # Detector
        self.detector = CPDetector()
        self.nadir_color = np.array([255, 0, 0], dtype=np.uint8)
        self.rope_color = np.array([0, 255, 0], dtype=np.uint8)
        self.buoy_color = np.array([0, 0, 255], dtype=np.uint8)

        # Plotting parameters
        self.plot_width = 400

    def perform_detection(self, side=0):
        """

        """
        # Select which side to perform detection on
        if side == 0:
            img_side = self.port
        else:
            img_side = self.starboard

        # Allocate array for detections
        # img_detections = np.zeros((img_side.shape[0], img_side.shape[1], 3), dtype=np.uint8)
        img_detections = np.copy(img_side).astype(np.uint8)

        for i, ping in enumerate(img_side):
            ping_results = self.detector.detect(ping)

            if ObjectID.NADIR in ping_results.keys():
                img_detections[i, ping_results[ObjectID.NADIR]['pos'], :] = self.nadir_color

            if ObjectID.ROPE in ping_results.keys():
                img_detections[i, ping_results[ObjectID.ROPE]['pos'], :] = self.rope_color

            if ObjectID.BUOY in ping_results.keys():
                img_detections[i, ping_results[ObjectID.BUOY]['pos'], :] = self.buoy_color

        # Save results
        if side == 0:
            self.port_detections[:, :, :] = img_detections[:, :, :]
        else:
            self.starboard_detections[:, :, :] = img_detections[:, :, :]

    def plot_detections(self):
        """
        Plot the detections
        """
        # Form yellow center band, helps to separate the port and starboard returns
        band = np.ones((self.height, 5, 3), dtype=np.uint8) * 255
        band[:, :, 2] = 0

        final = np.hstack((np.flip(self.port_detections[:, :self.plot_width, :], axis=1),
                           band,
                           self.starboard_detections[:, :self.plot_width, :]))

        plt.imshow(final)
        plt.show()


# %% Process
detector_test = sss_detector('data/sss_data_1.png')

detector_test.perform_detection(0)
detector_test.perform_detection(1)

detector_test.plot_detections()

