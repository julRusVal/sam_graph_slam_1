"""
Plot figure of paper of the dr vs estimated error ot the three different methods
compared to their own final estimates.
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

iros_data = True

# Paths
if iros_data:
    # This will load the IROS data
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # method 1
    method_1_path = script_directory + "/data/iros_method_1"
    method_1_error = np.genfromtxt(method_1_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_1_detections = np.genfromtxt(method_1_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 2
    method_2_path = script_directory + "/data/iros_method_2"
    method_2_error = np.genfromtxt(method_2_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_2_detections = np.genfromtxt(method_2_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 1
    method_3_path = script_directory + "/data/iros_method_3"
    method_3_error = np.genfromtxt(method_3_path + "/dr_online_error.csv",
                                   delimiter=',', dtype=float)
    method_3_detections = np.genfromtxt(method_3_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)
else:
    # This will load the ICRA data
    # Note that the ICRA data is organized a little bit differently

    # method 1
    method_1_error = np.genfromtxt('/home/julian/Documents/thesis_figs/dr_online_error_new.csv',
                                   delimiter=',', dtype=float)

    method_1_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_new"
    method_1_detections = np.genfromtxt(method_1_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 2
    method_2_error = np.genfromtxt('/home/julian/Documents/thesis_figs/dr_online_error_no_rope.csv',
                                   delimiter=',', dtype=float)

    method_2_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope"
    method_2_detections = np.genfromtxt(method_2_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

    # method 3
    method_3_error = np.genfromtxt('/home/julian/Documents/thesis_figs/dr_online_error_with_rope.csv',
                                   delimiter=',', dtype=float)

    method_3_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_no_rope"
    method_3_detections = np.genfromtxt(method_3_path + '/detections_graph.csv',
                                        delimiter=',', dtype=float)

# data = [new, nr, r]
data = [method_1_error, method_2_error, method_3_error]
detections = [method_1_detections, method_2_detections, method_3_detections]
# data_colors = [post_color, online_color, gt_color]
data_colors = [post_color, gt_color, online_color]
titles = ["Method 1", "Method 2", "Method 3"]

crossings = [[25, 45, 100, 145, 155, 210],
             [155, 160],
             [30, 160, 175]]

# buoy_detections = [11, 33, 55, 84,
#                  100, 127, 179, 196, 218]

detections = [method_1_detections, method_2_detections, method_3_detections]
detections_ignore = [1, 9]  # Select which detections were ignored

# === Setting ===
plot_crossings = True
plot_buoy_detections = True

# Create a plot
n_plots = len(data)
fig, ax = plt.subplots(n_plots, 1, sharex=True, figsize=(9, 6), dpi=150)

for i in range(0, n_plots):
    dr_error = data[i][0, :]
    online_error = data[i][1, :]
    detections_inds = detections[i][:, -1].astype(int)

    dr_rmse = np.sqrt(np.mean(dr_error ** 2))
    online_rmse = np.sqrt(np.mean(online_error ** 2))

    # Plot squared error
    # ax[i].plot(dr_error, label=f'DR error, RMSE: {dr_rmse:.2f}', color=dr_color)
    # ax[i].plot(online_error, label=f'Online error, RMSE: {online_rmse:.2f}', color=data_colors[i])

    ax[i].plot(dr_error, label=f'DR error', color=dr_color)
    ax[i].plot(online_error, label=f'Online error', color=data_colors[i])

    if plot_crossings:
        crossing_labeled = False
        for crossing in crossings[i]:
            ax[i].axvline(crossing, color='magenta', linestyle='--')

    if plot_buoy_detections:
        for detection_ignore_ind, detection_ind in enumerate(detections_inds):
            if detection_ignore_ind not in detections_ignore:
                ax[i].axvline(detection_ind, color='cyan', linestyle='--')

    # Add labels and title
    ax[i].legend(fontsize=legend_size, loc='upper left')
    ax[i].set_title(titles[i], fontsize=label_size)
    ax[i].grid(True)

    if i == n_plots // 2:
        ax[i].set_ylabel('Error [m]', fontsize=label_size)

# Show the plot
# fig.suptitle('Error Comparison', fontsize=title_size)
# plt.title('Error Comparison', fontsize=title_size)
plt.xlabel('m_t', fontsize=label_size)
# plt.ylabel('Error [m]', fontsize=label_size)
plt.grid(True)
plt.tight_layout()
plt.show()
