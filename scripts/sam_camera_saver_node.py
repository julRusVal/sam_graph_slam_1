#!/usr/bin/env python3

# Imports
import rospy
from sam_slam_utils.sam_slam_ros_classes import sam_image_saver


def main():
    rospy.init_node('slam_listener', anonymous=True)
    rospy.Rate(5)

    camera_down_topic = '/sam/perception/csi_cam_0/camera'
    camera_left_topic = '/sam/perception/csi_cam_1/camera'
    camera_right_topic = '/sam/perception/csi_cam_2/camera'

    path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data'

    print('initializing listener')
    listener = sam_image_saver(camera_down_top_name=camera_down_topic,
                               camera_left_top_name=camera_left_topic,
                               camera_right_top_name=camera_right_topic,
                               file_path=path_name)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
