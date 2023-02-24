#!/usr/bin/env python3
import gtsam

from sam_slam_utils.sam_slam_helper_funcs import create_Pose2
# %% Testing Junk #2
p_0 = create_Pose2([0, 0, 0, 1, 0, 0, 0])
p_1 = create_Pose2([1, 0, 0, 1, 0, 0, 0])
p_2 = create_Pose2([1, 1, 0, .707, 0, 0, 0.707])

b_01 = p_0.between(p_1)
b_12 = p_1.between(p_2)