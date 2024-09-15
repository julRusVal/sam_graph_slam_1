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
"""
Data set naming convention
baseline 1: buoy only
baseline 2: buoy and single landmark per rope
proposed: buoy and single landmark per rope detection
"""
use_icra_2024 = True

script_directory = os.path.dirname(os.path.abspath(__file__))

if use_icra_2024:
    baseline_1_path = script_directory + "/data/icra_2024_baseline_1"
    baseline_2_path = script_directory + "/data/icra_2024_baseline_2"
    proposed_path = script_directory + "/data/icra_2024_proposed"
else:
    baseline_1_path = script_directory + "/data/iros_method_3"
    baseline_2_path = script_directory + "/data/iros_method_1"
    proposed_path = script_directory + "/data/iros_method_2"

# Load
# method 1
baseline_1_metrics = np.genfromtxt(baseline_1_path + '/performance_metrics.csv',
                                   delimiter=',', dtype=float)

# method 2
baseline_2_metrics = np.genfromtxt(baseline_2_path + '/performance_metrics.csv',
                                 delimiter=',', dtype=float)

# method 3
proposed_metrics = np.genfromtxt(proposed_path + '/performance_metrics.csv',
                                 delimiter=',', dtype=float)

# Data to process
data_colors = [post_color, gt_color, online_color]
titles = ["Baseline 1", "Baseline 2", "Proposed"]
data = [baseline_1_metrics, baseline_2_metrics, proposed_metrics]

# === Setting ===


# Create a plot
n_plots = len(data)
fig, ax = plt.subplots(n_plots, 1, sharex=True, figsize=(9, 6), dpi=150)
ax_2 = []

fig.suptitle("Performance Metrics", fontsize=title_size)

for i in range(0, n_plots):
    # Data
    update_times = data[i][:, 0]
    update_time_total = np.sum(update_times)
    factor_counts = data[i][:, 1]
    update_type = data[i][:, 2]  # 0: odometry, 1: buoy, 2: rope

    max_time = np.max(update_times)
    max_factors = np.max(factor_counts)

    # factor type
    type_0 = np.sum(data[i][:, 2] == 0.0)
    type_1 = np.sum(data[i][:, 2] == 1.0)
    type_2 = np.sum(data[i][:, 2] == 2.0)
    type_tot = type_0 + type_1 + 2 * type_2

    # Create the first subplot with the left y-axis
    # ax1 = fig.add_subplot(111)
    ax[i].plot(update_times, color=dr_color, label=f'update_times')
    # ax[i].set_xlabel('X-axis')
    if i == n_plots // 2:
        ax[i].set_ylabel('Update times [s]', fontsize=label_size, color='k')
    ax[i].tick_params('y', color='k')

    # Manual factor counting
    manual_factor_count = None
    ax_2.append(ax[i].twinx())
    try:
        factor_counter = data[i][:, 3]
        prioe_counter = data[i][:, 4]
        manual_factor_count = np.sum(factor_counter) + np.sum(prioe_counter)
        # DEBUGGING
        ax_2[i].plot(factor_counter, color='g', label='factor counter')
        ax_2[i].plot(prioe_counter, color='m', label='prior counter')
    except IndexError:
        print(f"Incomplete performance {titles[i]}")
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

    title = (titles[i] + f" - Max time: {max_time:.3g}s Total time: {update_time_total:.3g}s   " +
             f" Max factors: {int(max_factors)} ({type_0}, {type_1}, {type_2}: {type_tot}) man_count: {manual_factor_count}")
    ax[i].set_title(title, fontsize=int(label_size * .8))

    ax[i].grid(True)
    ax[i].margins(x=0)

# plt.xlabel('m_t', fontsize=label_size)
# plt.ylabel('Error [m]', fontsize=label_size)
# plt.grid(True)
# plt.tight_layout()
fig.set_size_inches(figure_width, figure_height)

if use_icra_2024:
    file_path = script_directory + "/data/icra_2024_results"
else:
    file_path = script_directory + "/data/iros_results"
plt.savefig(file_path + "/performance_metrics.png", dpi=300)

plt.show()
