#!/usr/bin/env python3

# Imports
import rospy
from sam_slam_utils.sam_slam_ros_classes import sam_slam_listener
from sam_slam_utils.sam_slam_proc_classes import online_slam_2d


def main():
    rospy.init_node('slam_listener', anonymous=True)
    rate = rospy.Rate(5)

    # Topic parameters
    robot_name = rospy.get_param('robot_name', 'sam')
    frame_name = rospy.get_param('frame', 'map')
    simulated_data = rospy.get_param('simulated_data', False)
    record_gt = rospy.get_param('record_ground_truth', False)

    # This needs to match the structure defined in algae_map_markers.py
    ropes_by_buoy_ind = [[0, 5],
                         [1, 4],
                         [2, 3]]

    # Output parameters
    path_name = rospy.get_param('path_name',
                                '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/online_testing')
    data_processed = False

    # ===== Start =====
    print('SAM SLAM starting\n'
          f'Robot name: {robot_name}\n'
          f'Frame name: {frame_name}\n'
          f'Simulated data: {simulated_data}')

    print('Initializing online graph')
    online_graph = online_slam_2d(ropes_by_buoy_ind=ropes_by_buoy_ind)

    print('initializing listener')
    listener = sam_slam_listener(robot_name=robot_name,
                                 frame_name=frame_name,
                                 simulated_data=simulated_data,
                                 record_gt=record_gt,
                                 path_name=path_name,
                                 online_graph=online_graph)

    while not rospy.is_shutdown():
        # Add the end of the run the listener will save its data, at this point we perform the offline slam
        if listener.data_written and not data_processed:
            print("Processing data")
            # listener.analysis.calculate_corresponding_points(debug=True)
            listener.analysis.visualize_posterior(plot_gt=False,
                                                  plot_dr=False)
            data_processed = True
            # rospy.signal_shutdown("Shutting down gracefully")
        rate.sleep()


if __name__ == '__main__':
    main()
