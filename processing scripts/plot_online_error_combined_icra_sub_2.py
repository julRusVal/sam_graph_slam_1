"""
NEW:
Plot the online errors of the proposed method and the base line methods

This plot is intended for the second ICRA submission and addresses some of the following:
- Removal of dr error

OLD:
Plot figure of paper of the dr vs estimated error ot the three different methods.
Compared to the final estimate of the method 2 (using buoy and ropes).
"""
import matplotlib.pyplot as plt
import numpy as np

# === Settings ===

# = Colors =
dr_color = 'r'
proposed_color = 'b'
base_1_color = 'g'
base_2_color = 'm'
rope_color = 'b'
buoy_color = 'k'
title_size = 16
legend_size = 12
label_size = 12

# = Names =
x_axis_label = "Poses"
y_axis_label = "Error [m]"

# = Data to include =
"""
Check that the data provided matches the newest nomenclature. We have changed what the methods are reffered to as
is the different revisions... 

Proposed: real_testing_rope
baseline 1 (buoy only): real_testing_no_rope
baseline 2 (buoy and single rope prior per rope): real_testing_new
"""
plot_dr_error = False  # w.r.t. proposed method final estimate
show_rmse = True

proposed_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_online.csv'
base_1_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_no_rope/analysis_online.csv'
base_2_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_new/analysis_online.csv'

final_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_est.csv'
dr_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_dr.csv'

# === Plotting

# Load
proposed_online = np.genfromtxt(proposed_online_path,
                                delimiter=',', dtype=float)

base_1_online = np.genfromtxt(base_1_online_path,
                   delimiter=',', dtype=float)

base_2_online = np.genfromtxt(base_2_online_path,
                    delimiter=',', dtype=float)

dr = np.genfromtxt(dr_path,
                    delimiter=',', dtype=float)

final = np.genfromtxt(final_path,
                    delimiter=',', dtype=float)

# Organize data, colors, titles
data = [dr, base_1_online, base_2_online, proposed_online]
colors = [dr_color, base_1_color, base_2_color, proposed_color]
titles = ["DR", "Baseline 1", "baseline 2", "Proposed Method"]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

errors = []
rms_errors = []
for i in range(len(data)):
    # Check whether plotting DR is desired
    if titles[i] == 'DR' and not plot_dr_error:
        continue

    # Plot with rmse
    if show_rmse:
        # Calculate RMSE of method, w.r.t final estimate of proposed method
        max_ind = min(final.shape[0], data[i].shape[0])
        error = np.sqrt(np.sum((final[0:max_ind, 0:2] - data[i][0:max_ind, 0:2]) ** 2, axis=1))
        rms_e = np.sqrt(np.mean(error ** 2))

        # Plot online error of method
        ax.plot(error, label=f'{titles[i]}, RMSE: {rms_e:.2f}', color=colors[i])

    # Plot without rmse
    else:
        ax.plot(data[i], label=titles[i], color=colors[i])

# Show the plot
ax.grid(True)
ax.legend(fontsize=legend_size, loc='upper left')

# plt.title('Error Comparison', fontsize=title_size)
plt.xlabel(x_axis_label, fontsize=label_size)
plt.ylabel(y_axis_label, fontsize=label_size)
plt.grid(True)

# testing
file_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/icra_2024"
plt.savefig(file_path + "/online_error.png", dpi=300)

plt.show()