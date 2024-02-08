"""
Plot figure of paper of the dr vs estimated error ot the three different methods
compared to their own final estimates.
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
label_size = 14

# Pick one root path
# Method 1
root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_new"
# Method 2
root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_rope"
# Method 3
# root_path = "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/real_testing_no_rope"

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

# Load
online = np.genfromtxt(online_path,
                  delimiter=',', dtype=float)

buoys = np.genfromtxt(buoy_path,
                   delimiter=',', dtype=float)

dr = np.genfromtxt(dr_path,
                   delimiter=',', dtype=float)

gt = np.genfromtxt(gt_path,
                   delimiter=',', dtype=float)

for count in range(start_ind, online.shape[0]):
    last_ind = count
    for rope_i in range(len(ropes)):
        rope = ropes[rope_i]
        x_s = [buoys[k, 0] for k in rope]
        y_s = [buoys[k, 1] for k in rope]
        plt.plot(x_s, y_s, label='rope', color='r', linestyle='-')

    # Create the line plot
    plt.plot(online[0:last_ind,0], online[0:last_ind, 1], color='b', label='Data Points', marker='o', linestyle='-')

    # Add labels and a legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'count {count}')
    plt.axis('equal')

    # Show the plot
    plt.grid(True)

    if count == stop_ind:
        plt.show()
    else:
        plt.pause(1)  # Pause for 1 second (adjust as needed)

