#!/usr/bin/env python3

import rospy
from sam_slam_utils.pipeline_slam_gt_dr_publisher import pipeline_sim_dr_gt_publisher

try:
    dr_publisher_node = pipeline_sim_dr_gt_publisher()
    rospy.spin()
except rospy.ROSInterruptException:
    pass