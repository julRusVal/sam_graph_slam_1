#!/usr/bin/env python3

import math
import numpy as np

def angle_between_rads(target_angle, source_angle):
    # Bound the angle [-pi, pi]
    target_angle = math.remainder(target_angle, 2 * np.pi)
    source_angle = math.remainder(source_angle, 2 * np.pi)

    diff_angle = target_angle - source_angle

    if diff_angle > np.pi:
        diff_angle = diff_angle - 2 * np.pi
    elif diff_angle < -1 * np.pi:
        diff_angle = diff_angle + 2 * np.pi

    return diff_angle