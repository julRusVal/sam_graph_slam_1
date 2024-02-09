import numpy as np
import gtsam

def interpolated_pose2(pose2_list, time_list, interpolate_time):
    index = np.searchsorted(time_list, interpolate_time, side='right') - 1

    # Interpolate the position of the detection
    if index < 0:
        interpolated_pose = pose2_list[0]
    elif index >= len(pose2_list) - 1:
        interpolated_pose = pose2_list[-1]
    else:
        # Get the two poses and their times
        pose1, pose2 = pose2_list[index], pose2_list[index + 1]
        time1, time2 = time_list[index], time_list[index + 1]

        # Interpolate the x, y, and theta values separately
        x = np.interp(interpolate_time, [time1, time2], [pose1.x(), pose2.x()])
        y = np.interp(interpolate_time, [time1, time2], [pose1.y(), pose2.y()])
        theta = np.interp(interpolate_time, [time1, time2], [pose1.theta(), pose2.theta()])

        # Create a new Pose2 object with the interpolated values
        interpolated_pose = gtsam.Pose2(x, y, theta)

    return interpolated_pose

# Parameters
initial_pose = gtsam.Pose2(0.0, 0.0, 0.0)  # initial pose
delta_pose = gtsam.Pose2(1.0, 0.0, 0)  # change in pose per step
num_steps = 10  # number of steps

# Time parameters
initial_time = 0
time_step = 1

# Noise parameters
# Initial
init_pos_sigma = 0.1  # standard deviation of Gaussian noise
init_ang_sigma = 0.001  # standard deviation of Gaussian noise

# step
step_dist_sigma = 0.1  # standard deviation of Gaussian noise
step_ang_sigma = 0.001  # standard deviation of Gaussian noise

# Create the measured, actual, and time trajectory
measured_trajectory = [initial_pose]

# The actual initial is corrupt
initial_noise_vector = np.array([np.random.normal(0, init_pos_sigma),
                                 np.random.normal(0, init_pos_sigma),
                                 np.random.normal(0, init_ang_sigma)])

actual_trajectory = [initial_pose.retract(initial_noise_vector)]

# time of trajectory values
time_trajectory = [initial_time]

for _ in range(num_steps):
    # Add measured pose
    next_measured_pose = measured_trajectory[-1].compose(delta_pose)
    measured_trajectory.append(next_measured_pose)

    # Add actual pose
    # Add Gaussian noise measured step, delta_pose
    step_noise_vector = np.array([np.random.normal(0, step_dist_sigma),
                                 np.random.normal(0, step_dist_sigma),
                                 np.random.normal(0, step_ang_sigma)])

    next_actual_pose = actual_trajectory[-1].compose(delta_pose.retract(step_noise_vector))
    actual_trajectory.append(next_actual_pose)

    # Add time step
    time_trajectory.append(time_trajectory[-1] + time_step)

# Create two point landmarks
landmark = gtsam.Point2(10.0, 10.0)
detection_times = [1, 1.5, 3.25]  # times of detection
detection_input_time = [1, 2, 6]
detection_pose2s = []  # this is based on the detection time and the actual_trajectory

for detection_time in detection_times:
    detection_pose2 = interpolated_pose2(actual_trajectory, time_trajectory, detection_time)
    detection_pose2s.append(detection_pose2)

    detection_bearing = detection_pose2.bearing(landmark)
    detection_range = detection_pose2.range(landmark)

    print(f"Detection pose: {detection_pose2}")
    print(f"Detection bearing: {detection_bearing}")
    print(f"Detection range: {detection_range}")


# print("Actual trajectory:", actual_trajectory)
# print("Measured trajectory:", measured_trajectory)
