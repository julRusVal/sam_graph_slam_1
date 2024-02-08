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

# Paths

# Method 1
method_1_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/iros_method_1"
# Method 2
method_2_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/iros_method_2"
# Method 3
method_3_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/iros_method_3"

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

for i in range(0, n_plots):
    # Data
    update_times = data[i][:, 0]
    factor_counts = data[i][:, 1]
    update_type = data[i][:, 2]  # 0: odometry, 1: buoy, 2: rope

    # plt.plot(update_times, label=f'update_times', color=dr_color)
    # plt.plot(factor_counts, label=f'factor_counts', color=data_colors[i])
    # plt.plot(update_type, label=f'update_type', color=data_colors[i])

    fig = plt.figure(i, figsize=(9, 3), dpi=150)

    # Create the first subplot with the left y-axis
    ax1 = fig.add_subplot(111)
    ax1.plot(update_times, color=dr_color, label=f'update_times')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('time [s]', color=dr_color)
    ax1.tick_params('y', color='k')

    # Create the second subplot with the right y-axis
    ax2 = ax1.twinx()
    x = [i for i in range(len(update_type))]
    ax2.bar(x, update_type, alpha=0.5, color=data_colors[i], label='update_type')
    ax2.set_ylabel('Update type', color=data_colors[i])
    ax2.tick_params('y', colors='k')

    # Add a legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=legend_size)

    plt.title(titles[i], fontsize=title_size )
    # plt.xlabel('m_t', fontsize=label_size)
    # plt.ylabel('Error [m]', fontsize=label_size)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
