#!/usr/bin/env python3
import gtsam
import numpy as np

from sam_slam_utils.sam_slam_helpers import create_Pose2

# %%
# %% Testing Junk #2
p_0 = create_Pose2([0, 0, 0, 1, 0, 0, 0])
p_1 = create_Pose2([1, 0, 0, 1, 0, 0, 0])
p_2 = create_Pose2([1, 1, 0, .707, 0, 0, 0.707])

b_01 = p_0.between(p_1)
b_12 = p_1.between(p_2)

graph = gtsam.NonlinearFactorGraph()

point = np.array((3, 5), dtype=np.float64)

noise_model = gtsam.noiseModel.Diagonal.Sigmas((1, 1))

graph.addPriorPoint2(1, point, noise_model)
