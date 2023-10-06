"""
Plot figure of paper of the dr vs estimated error ot the three different methods.
Compared to the final estimate of the method 2 (using buoy and ropes).
"""
import matplotlib.pyplot as plt
import numpy as np

# Settings
dr_color = 'r'
gt_color = 'b'
post_color = 'g'
online_color = 'm'
rope_color = 'b'
buoy_color = 'k'
title_size = 16
legend_size = 12
label_size = 12

r_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_online.csv'
nr_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_no_rope/analysis_online.csv'
new_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_new/analysis_online.csv'
final_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_est.csv'
dr_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_dr.csv'

# r_online_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_online.csv'
# nr_online_path = r_online_path
# new_online_path = r_online_path
# final_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_est.csv'
# dr_path = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope/analysis_dr.csv'

# Load
r_online = np.genfromtxt(r_online_path,
                  delimiter=',', dtype=float)

nr_online = np.genfromtxt(nr_online_path,
                   delimiter=',', dtype=float)

new_online = np.genfromtxt(new_online_path,
                    delimiter=',', dtype=float)

dr = np.genfromtxt(dr_path,
                    delimiter=',', dtype=float)

final = np.genfromtxt(final_path,
                    delimiter=',', dtype=float)

# data = [r_online, nr_online, new_online, dr]
# colors = [gt_color, online_color, post_color, dr_color]
# titles = ["Method 2: Ropes and Buoys", "Method 2: Buoys", "Method 1", "DR"]

data = [dr, new_online, nr_online, r_online]
colors = [dr_color, post_color, online_color, gt_color]
titles = ["DR", "Method 1", "Method 2: Buoys", "Method 2: Ropes and Buoys"]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

errors = []
rms_errors = []
for i in range(len(data)):
    max_ind = min(final.shape[0], data[i].shape[0])
    error = np.sqrt(np.sum((final[0:max_ind, 0:2] - data[i][0:max_ind, 0:2]) ** 2, axis=1))
    rms_e = np.sqrt(np.mean(error ** 2))

    # Plot  error
    ax.plot(error, label=f'{titles[i]}, RMSE: {rms_e:.2f}', color=colors[i])

# Show the plot
ax.grid(True)
ax.legend(fontsize=legend_size, loc='upper left')

# plt.title('Error Comparison', fontsize=title_size)
plt.xlabel('Poses', fontsize=label_size)
plt.ylabel('Error [m]', fontsize=label_size)
plt.grid(True)


# testing
file_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing"
plt.savefig(file_path + "/final.png", dpi=300)

# fig.suptitle('Error Comparison', fontsize=title_size)
plt.show()