#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

def odometry_callback(odom):
    # Extract the position and orientation from the received Odometry message
    position = odom.pose.pose.position
    orientation = odom.pose.pose.orientation

    # Create a new PoseStamped message
    pose_stamped = PoseStamped()
    pose_stamped.header = odom.header
    pose_stamped.pose.position = position
    pose_stamped.pose.orientation = orientation

    # Publish the republished PoseStamped message
    pose_pub.publish(pose_stamped)

if __name__ == '__main__':
    rospy.init_node('odom_to_pose_republisher')

    # Create a publisher for the republished PoseStamped message
    pose_pub = rospy.Publisher('/republished_pose', PoseStamped, queue_size=10)

    # Subscribe to the Odometry topic
    rospy.Subscriber('/odom', Odometry, odometry_callback)

    # Spin the node to receive and process messages
    rospy.spin()
