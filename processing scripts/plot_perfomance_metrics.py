"""
Plot figure for IROS
performance metrics

expects .csv files with the following format: [update time, factor count, detection type]
update time: in units of seconds
factor count: number of factors in the graph
detection type: 0 for no detection, 1 for buoy detection, 2 for rope detection
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
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
figure_width = 8
figure_height = 6

# Paths
script_directory = os.path.dirname(os.path.abspath(__file__))
# Method 1
method_1_path = script_directory + "/data/iros_method_1"
# Method 2
method_2_path = script_directory + "/data/iros_method_2"
# Method 3
method_3_path = script_directory + "/data/iros_method_3"

# Load
# method 1
method_1_metrics = np.genfromtxt(method_1_path + '/performance_metrics.csv',
                                 delimiter=',', dtype=float)

# method 2
method_2_metrics = np.genfromtxt(method_2_path + '/performance_metrics.csv',
                                 delimiter=',', dtype=float)

# method 3
method_3_metrics = np.genfromtxt(method_3_path + '/performance_metrics.csv',
                                 delimiter=',', dtype=float)

# Data to process
data_colors = [post_color, gt_color, online_color]
titles = ["Method 1", "Method 2", "Method 3"]
data = [method_1_metrics, method_2_metrics, method_3_metrics]

# === Setting ===


# Create a plot
n_plots = len(data)
fig, ax = plt.subplots(n_plots, 1, sharex=True, figsize=(9, 6), dpi=150)
ax_2 = []

fig.suptitle("Performance Metrics", fontsize=title_size)

for i in range(0, n_plots):
    # Data
    update_times = data[i][:, 0]
    factor_counts = data[i][:, 1]
    update_type = data[i][:, 2]  # 0: odometry, 1: buoy, 2: rope

    max_time = np.max(update_times)
    max_factors = np.max(factor_counts)

    # Create the first subplot with the left y-axis
    # ax1 = fig.add_subplot(111)
    ax[i].plot(update_times, color=dr_color, label=f'update_times')
    # ax[i].set_xlabel('X-axis')
    if i == n_plots // 2:
        ax[i].set_ylabel('Update times [s]', fontsize=label_size, color='k')
    ax[i].tick_params('y', color='k')

    # Create the second subplot with the right y-axis
    # ax2 = ax1.twinx()
    # x = [i for i in range(len(update_type))]
    # ax2.bar(x, update_type, alpha=0.5, color=data_colors[i], label='update_type')
    # ax2.set_ylabel('Update type', color=data_colors[i])
    # ax2.tick_params('y', colors='k')

    ax_2.append(ax[i].twinx())
    ax_2[i].plot(factor_counts, color=data_colors[i], label='factor_counts')
    if i == n_plots // 2:
        ax_2[i].set_ylabel('Factor counts', fontsize=label_size, color='k')
    ax_2[i].tick_params('y', colors='k')

    if i == n_plots - 1:
        ax[i].set_xlabel('m_t', fontsize=label_size, color='k')

    # Add a legend
    lines, labels = ax[i].get_legend_handles_labels()
    lines2, labels2 = ax_2[i].get_legend_handles_labels()
    ax_2[i].legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_size)

    title = titles[i] + f" - Max Update time: {max_time:.3g}s" + f" Max factors: {int(max_factors)}"
    ax[i].set_title(title, fontsize=label_size)

    ax[i].grid(True)

# plt.xlabel('m_t', fontsize=label_size)
# plt.ylabel('Error [m]', fontsize=label_size)
# plt.grid(True)
# plt.tight_layout()
fig.set_size_inches(figure_width, figure_height)
plt.show()
