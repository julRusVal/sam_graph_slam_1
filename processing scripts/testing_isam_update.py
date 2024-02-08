import gtsam
import numpy as np

# Create a simple 2D pose factor graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# Add initial pose estimate
initial_pose = gtsam.Pose2(0.0, 0.0, 0.0)
initial_estimate.insert(0, initial_pose)

# Define noise models for odometry and measurement factors
odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)

# Add odometry factors to the graph
odom_factor1 = gtsam.BetweenFactorPose2(0, 1, gtsam.Pose2(1.0, 0.0, 0.0), odometry_noise)
odom_factor2 = gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(1.0, 0.0, 0.0), odometry_noise)

graph.add(odom_factor1)
graph.add(odom_factor2)

# Add a custom measurement factor (e.g., range measurement)
def custom_measurement_factor(pose_key, landmark_key, measurement, noise):
    return gtsam.RangeFactor(pose_key, landmark_key, measurement, noise)

# Add a landmark at position (2, 1)
landmark_key = 3
initial_estimate.insert(landmark_key, gtsam.Point2(2.0, 1.0))

# Add a custom measurement factor to the graph
measurement_factor = custom_measurement_factor(2, landmark_key, 1.5, measurement_noise)
graph.add(measurement_factor)

# Use ISAM to optimize the factor graph
params = gtsam.ISAM2Params()
isam = gtsam.ISAM2(params)
result = isam.update(graph, initial_estimate)

# Retrieve the optimized poses and landmarks
optimized_poses = result.calculateEstimate().filter(gtsam.Pose2)
optimized_landmarks = result.calculateEstimate().filter(gtsam.Point2)

# Print the optimized poses and landmarks
print("Optimized Poses:")
for key, pose in optimized_poses.items():
    print(f"Pose {key}: {pose}")

print("\nOptimized Landmarks:")
for key, landmark in optimized_landmarks.items():
    print(f"Landmark {key}: {landmark}")
