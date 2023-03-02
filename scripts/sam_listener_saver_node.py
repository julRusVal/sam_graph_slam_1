#!/usr/bin/env python3

# Imports
import rospy
from sam_slam_utils.sam_slam_ros_classes import sam_slam_listener


def main():
    rospy.init_node('slam_listener', anonymous=True)
    rospy.Rate(5)

    ground_truth_topic = '/sam/sim/odom'
    dead_reckon_topic = '/sam/dr/odom'
    detection_topic = '/sam/payload/sidescan/detection_hypothesis'
    buoy_topic = '/sam/sim/marked_positions'
    frame = 'map'

    print('initializing listener')
    listener = sam_slam_listener(ground_truth_topic, dead_reckon_topic, detection_topic, buoy_topic, frame)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
