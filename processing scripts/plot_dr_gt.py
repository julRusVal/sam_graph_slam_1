"""
Plot figure of paper of the dr and gt.

most plotting is carried out at runtime this plots based on the saved output
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Settings
dr_color = 'r'
gt_color = 'b'
post_color = 'g'
online_color = 'm'
rope_color = 'b'
buoy_color = 'k'
title_size = 16
legend_size = 12
label_size = 14

# Pick one root path
# Method 1
root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_new"
# Method 2
root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope"
# Method 3
# root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_no_rope"

# IROS
script_directory = os.path.dirname(os.path.abspath(__file__))

# method 1
root_path = script_directory + "/data/iros_method_1"

online_path = root_path + "/analysis_online.csv"
buoy_path = root_path + "/buoys.csv"

dr_path = root_path + "/dr_poses_graph.csv"
gt_path = root_path + "/gt_poses_graph.csv"

ropes = [[0, 5],
         [1, 4],
         [2, 3]]

last_ind = 30
start_ind = 20  # 150
stop_ind = 40  # 180

min_x, max_x = 0, 80
min_y, max_y = -20, 60

plot_limits = [min_x, max_x, min_y, max_y]

# Define the arrow interval
arrow_interval = 3

# Load
online = np.genfromtxt(online_path,
                       delimiter=',', dtype=float)

buoys = np.genfromtxt(buoy_path,
                      delimiter=',', dtype=float)

dr_poses = np.genfromtxt(dr_path,
                         delimiter=',', dtype=float)
dr_poses = dr_poses[:, :2]

gt_poses = np.genfromtxt(gt_path,
                         delimiter=',', dtype=float)
gt_poses = gt_poses[:, :2]

fig, ax = plt.subplots(figsize=(10, 8))
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)

# ===== Plot dead reckoning =====
dr_len = dr_poses.shape[0]
plt.plot(dr_poses[:, 0],
         dr_poses[:, 1],
         color=dr_color,
         label='Dead reckoning')

# # Add arrows to indicate direction with custom color
# for i in range(1, dr_len - 1, arrow_interval):
#     thing = dr_poses[i + 1, :]
#     plt.annotate('', xy=dr_poses[i + 1, :], xytext=dr_poses[i, :],
#                  arrowprops=dict(arrowstyle='->', color='r'),
#                  size=20)

# ==== Plot ground truth =====
plt.plot(gt_poses[:, 0],
         gt_poses[:, 1],
         color=gt_color,
         label='Auxiliary vehilce')

# Add arrows to indicate direction with custom color
for i in range(1, dr_len - 1, arrow_interval):
    plt.annotate('', xy=gt_poses[i + 1, :], xytext=gt_poses[i, :],
                 arrowprops=dict(arrowstyle='->', color='b'),
                 size=20)

# ==== Plot buoy priors ====
plt.scatter(buoys[:, 0],
            buoys[:, 1],
            color=buoy_color,
            label='Prior buoys')

# ==== Plot ropes, based on buoy priors ====
rope_labeled = False
for rope_i in range(len(ropes)):
    rope = ropes[rope_i]
    x_s = [buoys[k, 0] for k in rope]
    y_s = [buoys[k, 1] for k in rope]
    if not rope_labeled:
        plt.plot(x_s, y_s, label='ropes', color='k', linestyle='-')
        rope_labeled = True
    else:
        plt.plot(x_s, y_s, color='k', linestyle='-')

# Add labels and a legend
plt.xlabel('x [m]', fontsize=label_size)
plt.ylabel('y [m]', fontsize=label_size)

plt.xlim(min_y, max_x)
plt.ylim(min_y, max_y)
# plt.axis(plot_limits)

# plt.title(f'count {count}')
plt.legend(fontsize=legend_size, loc='upper right')

plt.axis('equal')
# plt.tight_layout()

# Show the plot
plt.grid(True)

plt.savefig(root_path + "/gt_dr_map.png", dpi=300, bbox_inches='tight')
plt.show()
