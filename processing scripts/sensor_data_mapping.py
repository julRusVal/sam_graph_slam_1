#!/usr/bin/env python3
import os

from sam_slam_utils.sam_slam_helpers import read_csv_to_array, read_csv_to_list
from sam_slam_utils.sam_slam_mapping import sss_mapping, image_mapping

# %% Load and process data
paths = {"mac": "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/online_testing/",
         "linux": "/home/julian/catkin_ws/src/sam_slam/processing scripts/data/online_testing/"}

path_name = ''
for path in paths.values():
    if os.path.isdir(path):
        path_name = path
        break

if len(path_name) == 0:
    print("path_name was not assigned")

# === Camera data ===
gt_base = read_csv_to_array(path_name + 'camera_gt.csv')
base = read_csv_to_array(path_name + 'camera_est.csv')  # change to camera_gt.csv, camera_dr.csv, or camera_est.csv

left_info = read_csv_to_list(path_name + 'left_info.csv')
right_info = read_csv_to_list(path_name + 'right_info.csv')
down_info = read_csv_to_list(path_name + 'down_info.csv')

# === SSS data ===
sss_base_gt = read_csv_to_array(path_name + 'sss_gt.csv')
sss_base_est = read_csv_to_array(path_name + 'sss_est.csv')
sss_path = path_name + "sss/"

# === Map information ===
# Select buoy position to use
# buoys.csv contains the ground truth locations of the buoys
# camera_buoys_est.csv contains the estimated locations of the buoys
buoy_info = read_csv_to_array(path_name + 'buoys.csv')
# buoy_info = read_csv_to_array(path_name + 'buoys_est.csv')

# Define structure of the farm
# Define the connections between buoys
# list of start and stop indices of buoys
# TODO hard coded rope structure
ropes = [[0, 4], [4, 2],
         [1, 5], [5, 3]]

# Define which connections above form a row
# Note: each rope section forms two planes for example [0, 4] and [4, 0], so that both sides can be imaged separately
# TODO hard coded rows
rows = [[0, 2], [3, 1], [4, 6], [7, 5]]

img_map = image_mapping(gt_base_link_poses=gt_base,
                        base_link_poses=base,
                        l_r_camera_info=[left_info, right_info, down_info],
                        buoy_info=buoy_info,
                        ropes=ropes,
                        rows=rows,
                        path_name=path_name)

sss_map = sss_mapping(sss_base_gt=sss_base_gt,
                      sss_base_est=sss_base_est,
                      sss_data_path=sss_path,
                      buoys=buoy_info,
                      ropes=ropes,
                      rows=rows)

# %% sonar plotting
sss_map.generate_3d_plot(farm=True,
                         sss_pos=True,
                         sss_data=True)

# %%Plot base and camera poses
# img_map.plot_fancy(other_name=None)
# img_map.plot_fancy(other_name="left")
# img_map.plot_fancy(other_name="down")
# img_map.plot_fancy(img_map.gt_camera_pose3s)  # plot the ground ruth as other
# plot_fancy(base_gt_pose3s, left_gt_pose3s, buoy_info, points)

# %% Perform processing on images
img_map.process_images(ignore_first=8, verbose=True)  #
# img_map.process_ground_plane_images(path_name, ignore_first=8, verbose=True)  # ground plane processing incomplete
img_map.simple_stitch_planes_images(max_dist=12)
img_map.combine_row_images()

# %% Produce 3d map
# img_map.plot_3d_map(show_base=True)
img_map.plot_3d_map_mayavi()

# %% Quantify quality of registration
img_map.quantify_registration(method="ccorr",  # "ccorr"
                              min_overlap_threshold=0.05,
                              verbose_output=True)
img_map.report_registration_quality()
