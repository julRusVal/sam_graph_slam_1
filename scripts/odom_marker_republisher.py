#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

def dr_odom_callback(msg):
    # Create an arrow marker for the first odometry
    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = msg.header.frame_id
    marker.id = 0
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose = msg.pose.pose
    marker.scale.x = 3
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    dr_marker_pub.publish(marker)

def gt_odom_callback(msg):
    # Create a sphere marker for the second odometry
    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = msg.header.frame_id
    marker.id = 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = msg.pose.pose.position.x
    marker.pose.position.y = msg.pose.pose.position.y
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = 2
    marker.scale.y = 2
    marker.scale.z = 2
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    gt_marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('odometry_marker_publisher')

    dr_odom_topic = "/sam/dr/odom"
    gt_odom_topic = "/sam/dr/gps_odom"

    dr_marker_pub = rospy.Publisher('/sam_slam/dr_marker', Marker, queue_size=10)
    gt_marker_pub = rospy.Publisher('/sam_slam/gt_marker', Marker, queue_size=10)

    rospy.Subscriber(dr_odom_topic, Odometry, dr_odom_callback)
    rospy.Subscriber(gt_odom_topic, Odometry, gt_odom_callback)



    print("DR/GT marker publisher started")

    rospy.spin()
