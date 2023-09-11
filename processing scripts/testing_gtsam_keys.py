import gtsam
import numpy as np
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

prior_model = gtsam.noiseModel.Diagonal.Sigmas((0.3, 0.3, 0.1))
odometry_model = gtsam.noiseModel.Diagonal.Sigmas((0.2, 0.2, 0.1))
Between = gtsam.BetweenFactorPose2

slam_graph = gtsam.NonlinearFactorGraph()
slam_graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), prior_model))
slam_graph.add(Between(1, 2, gtsam.Pose2(2.0, 0.0, 0.0), odometry_model))
slam_graph.add(Between(2, 3, gtsam.Pose2(2.0, 0.0, 0.0), odometry_model))

# Add Range-Bearing measurements to two different landmarks L1 and L2
measurement_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
BR = gtsam.BearingRangeFactor2D
l = {k: gtsam.symbol('l', k) for k in [1, 2]}  # name landmark variables
slam_graph.add(BR(1, l[1], gtsam.Rot2.fromDegrees(45), np.sqrt(4.0 + 4.0), measurement_model))  # pose 1 -*- landmark 1
slam_graph.add(BR(2, l[1], gtsam.Rot2.fromDegrees(90), 2.0, measurement_model))  # pose 2 -*- landmark 1
slam_graph.add(BR(3, l[2], gtsam.Rot2.fromDegrees(90), 2.0, measurement_model))  # pose 3 -*- landmark 2

slam_initial = gtsam.Values()
slam_initial.insert(1, gtsam.Pose2(-0.25, 0.20, 0.15))
slam_initial.insert(2, gtsam.Pose2(2.30, 0.10, -0.20))
slam_initial.insert(3, gtsam.Pose2(4.10, 0.10, 0.10))
slam_initial.insert(l[1], gtsam.Point2(1.80, 2.10))
slam_initial.insert(l[2], gtsam.Point2(4.10, 1.80))

optimizer = gtsam.LevenbergMarquardtOptimizer(slam_graph, slam_initial)
slam_result = optimizer.optimize()

# covariance_matrix = slam_result.marginalCovariance(1)
# This is the 'basic' covariance, it needs to be rotated to align with the rope
along_sigma = 25
cross_sigma = 1
cov_matrix = np.array([[along_sigma ** 2, 0.0],
                       [0.0, cross_sigma ** 2]])

# Rotate the cov_matrix to align with the rope
rope_angle = np.pi/3
rotation_matrix = np.array([[np.cos(rope_angle), -np.sin(rope_angle)],
                            [np.sin(rope_angle), np.cos(rope_angle)]])
rot_cov_matrix = rotation_matrix @ cov_matrix @ rotation_matrix.transpose()
gtsam_covar = gtsam.noiseModel.Gaussian.Covariance(rot_cov_matrix)

# marginals = gtsam.Marginals(slam_graph, slam_result)
# for k in [1, 2, 3]:
#     gtsam_plot.plot_pose2(0, slam_result.atPose2(k), 0.5, marginals.marginalCovariance(k))
# for j in [1, 2]:
#     gtsam_plot.plot_point2(0, slam_result.atPoint2(l[j]), 'g', P=marginals.marginalCovariance(l[j]))
#
# plt.axis('equal')
# plt.show()

# print factors that relate to key of interest
# key_of_interest = l[1]
# factor_keys = list(slam_graph.keyVector())
# for factor_key in factor_keys:
#     factor = slam_graph.at(factor_key)
#     if key_of_interest in factor.keys():
#         print(factor)

# Do modification of prior factor
key_of_interest = 1
new_prior = gtsam.PriorFactorPose2(key_of_interest, gtsam.Pose2(1.0, 1.0, 1.0), prior_model)
#factor_keys = slam_graph.keys()

print("Original Graph")
print(slam_graph)

replaced = False
for factor_inx in range(slam_graph.size()):
    factor = slam_graph.at(factor_inx)
    if key_of_interest in factor.keys() and isinstance(factor, gtsam.PriorFactorPose2):
        print("Replaced")
        slam_graph.replace(factor_inx, new_prior)
        replaced = True
        break

if not replaced:
    print("Added")
    slam_graph.add(new_prior)

print("New Graph")
print(slam_graph)

slam_result_2 = optimizer.optimize()
# plotting
marginals = gtsam.Marginals(slam_graph, slam_result)
for k in [1, 2, 3]:
    gtsam_plot.plot_pose2(0, slam_result.atPose2(k), 0.5, marginals.marginalCovariance(k))
for j in [1, 2]:
    gtsam_plot.plot_point2(0, slam_result.atPoint2(l[j]), 'g', P=marginals.marginalCovariance(l[j]))

marginals_2 = gtsam.Marginals(slam_graph, slam_result_2)
for k in [1, 2, 3]:
    gtsam_plot.plot_pose2(1, slam_result_2.atPose2(k), 0.5, marginals_2.marginalCovariance(k))
for j in [1, 2]:
    gtsam_plot.plot_point2(1, slam_result_2.atPoint2(l[j]), 'g', P=marginals_2.marginalCovariance(l[j]))

plt.axis('equal')
plt.show()