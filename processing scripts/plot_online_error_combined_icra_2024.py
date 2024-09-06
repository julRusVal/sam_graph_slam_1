"""
Plot figure of paper of the dr vs estimated error ot the three different methods
compared to their own final estimates.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# = Colors =
dr_color = 'r'
proposed_color = 'b'
base_1_color = 'g'
base_2_color = 'r'
rope_color = 'b'
buoy_color = 'k'
title_size = 16
legend_size = 12
label_size = 12

# = Names =
x_axis_label = "Poses"
y_axis_label = "Error [m]"

# = visulualizations =
plot_crossings = False
plot_buoy_detections = False
plot_dr_error = False  # w.r.t. proposed method final estimate
show_rmse = True

# = Data to include =
"""
Check that the data provided matches the newest nomenclature. We have changed what the methods are reffered to as
is the different revisions... 

Proposed: real_testing_rope <-- IROS method 2
baseline 1 (buoy only): real_testing_no_rope <-- IROS method 3
baseline 2 (buoy and single rope prior per rope): real_testing_new <-- IROS method 1

This script uses the IROS data but uses the nameing convention of the ICRA 2024 paper
(see above)
"""

script_directory = os.path.dirname(os.path.abspath(__file__))

# Baseline 1
base_1_path = script_directory + "/data/iros_method_3"
base_1_error = np.genfromtxt(base_1_path + "/dr_online_error.csv",
                               delimiter=',', dtype=float)
base_1_detections = np.genfromtxt(base_1_path + '/detections_graph.csv',
                                    delimiter=',', dtype=float)

# Baseline 2
base_2_path = script_directory + "/data/iros_method_1"
base_2_error = np.genfromtxt(base_2_path + "/dr_online_error.csv",
                               delimiter=',', dtype=float)
base_2_detections = np.genfromtxt(base_2_path + '/detections_graph.csv',
                                    delimiter=',', dtype=float)

# Proposed
proposed_path = script_directory + "/data/iros_method_2"
proposed_error = np.genfromtxt(proposed_path + "/dr_online_error.csv",
                               delimiter=',', dtype=float)
proposed_detections = np.genfromtxt(proposed_path + '/detections_graph.csv',
                                    delimiter=',', dtype=float)



# data = [new, nr, r]
data = [base_1_error, base_2_error, proposed_error]
detections = [base_1_detections, base_2_detections, proposed_detections]
data_colors = [base_1_color, base_2_color, proposed_color]
titles = ["Baseline 1", "Baseline 2", "Proposed method"]

crossings = [[25, 45, 100, 145, 155, 210],
             [155, 160],
             [30, 160, 175]]

# buoy_detections = [11, 33, 55, 84,
#                  100, 127, 179, 196, 218]

detections = [base_2_detections, proposed_detections, base_1_detections]
detections_ignore = [1, 9]  # Select which detections were ignored


# Create a plot
n_data = len(data)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9, 6), dpi=150)

for i in range(0, n_data):
    dr_error = data[i][0, :]
    online_error = data[i][1, :]
    detections_inds = detections[i][:, -1].astype(int)

    dr_rmse = np.sqrt(np.mean(dr_error ** 2))
    online_rmse = np.sqrt(np.mean(online_error ** 2))

    # Plot squared error
    # ax[i].plot(dr_error, label=f'DR error, RMSE: {dr_rmse:.2f}', color=dr_color)
    # ax[i].plot(online_error, label=f'Online error, RMSE: {online_rmse:.2f}', color=data_colors[i])

    # if plot_dr_error:
    #     ax.plot(dr_error, label=f'DR error: {}', color=dr_color)
    ax.plot(online_error, label=f'{titles[i]}', color=data_colors[i])

    if plot_crossings:
        crossing_labeled = False
        for crossing in crossings[i]:
            ax[i].axvline(crossing, color='magenta', linestyle='--')

    if plot_buoy_detections:
        for detection_ignore_ind, detection_ind in enumerate(detections_inds):
            if detection_ignore_ind not in detections_ignore:
                ax.axvline(detection_ind, color='cyan', linestyle='--')

    # Add labels and title
    ax.legend(fontsize=legend_size, loc='upper left')
    ax.grid(True)


# Show the plot
# fig.suptitle('Error Comparison', fontsize=title_size)
# plt.title('Error Comparison', fontsize=title_size)
plt.xlabel(x_axis_label, fontsize=label_size)
plt.ylabel(y_axis_label, fontsize=label_size)
plt.grid(True)
plt.tight_layout()

file_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/icra_2024"
plt.savefig(file_path + "/online_error.png", dpi=300)

plt.show()
