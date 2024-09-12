#!/usr/bin/env python3

import rospy

import numpy as np

# from geometry_msgs.msg import Quaternion, TransformStamped
# from sensor_msgs.msg import NavSatFix
# from nav_msgs.msg import Odometry
# import tf
# from geodesy import utm
# import numpy as np
# import tf2_ros
# import message_filters
# import time

from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose, PoseArray

import tf
import tf2_ros
from geodesy import utm


class PubBuoyTf(object):

    def __init__(self):

        self.cnt = 0
        self.utm_frame = rospy.get_param('~utm_frame', 'utm')

        self.buoy_positions_sat = []
        self.buoy_positions_utm = []
        self.buoy_markers = None
        self.rope_inner_markers = None
        self.rope_outer_markers = None
        self.rope_lines = None  # Is this need for anything??
        self.marker_rate = 1.0
        self.marker_scale = 1.0

        # Broadcast UTM to map frame
        self.listener = tf.TransformListener()
        self.static_tf_bc = tf2_ros.StaticTransformBroadcaster()

        # 10 Measured gps point of algae farm buoys
        # Some are duplicates
        # self.lats = [58.2508586667, 58.25076345,
        #              58.2507474333, 58.2506673333,
        #              58.2507919833, 58.2509082667,
        #              58.2509016, 58.2509062833,
        #              58.25101055, 58.2510166]
        # self.longs = [11.4514784167, 11.45136245,
        #               11.4513825333, 11.4512373833,
        #               11.4508920167, 11.4510152,
        #               11.45100355,11.4509971833,
        #               11.4511224667 , 11.4511216833]

        self.lats = [58.2508586667, 58.25076345,
                     58.2506673333,
                     58.2507919833,
                     58.2509016,
                     58.25101055]

        self.longs = [11.4514784167, 11.45136245,
                      11.4512373833,
                      11.4508920167,
                      11.45100355,
                      11.4511224667]

        self.ropes = [[0, 5],
                      [1, 4],
                      [2, 3]]

        self.n_buoys_per_rope = 8

        # marker publishers
        self.marker_pub = rospy.Publisher('/sam/real/marked_positions', MarkerArray, queue_size=10)

        self.rope_marker_pub = rospy.Publisher('/sam/real/marked_rope', MarkerArray, queue_size=10)
        self.rope_outer_marker_pub = rospy.Publisher('/sam/real/rope_outer_marker', MarkerArray, queue_size=10)

        self.rope_lines_pub = rospy.Publisher('/sam/real/marked_rope_lines', MarkerArray, queue_size=10)

    def trigger_measurement_callback(self, msg):
        rospy.loginfo("Trigger measurement callback")
        self.measure_now = True

    def publisher_transform(self, navSatFix):

        num_buoys = len(self.buoy_positions_sat)
        buoy_frame = "buoy_{}_frame".format(self.cnt)
        print("buoy ", buoy_frame)
        self.cnt += 1
        self.buoy_positions_sat.append(navSatFix)

        buoy_utm = utm.fromLatLong(navSatFix.latitude, navSatFix.longitude)

        try:
            (world_trans, world_rot) = self.listener.lookupTransform(self.utm_frame,
                                                                     buoy_frame,
                                                                     rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException):
            rospy.loginfo("GPS node: broadcasting transform %s to %s" % (self.utm_frame, buoy_frame))
            transformStamped = TransformStamped()
            quat = [0, 0, 0, 1.]
            # quat = tf.transformations.quaternion_from_euler(np.pi, -np.pi/2., 0., axes='rxzy')
            transformStamped.transform.translation.x = buoy_utm.easting
            transformStamped.transform.translation.y = buoy_utm.northing
            transformStamped.transform.translation.z = 0.
            transformStamped.transform.rotation = Quaternion(*quat)
            transformStamped.header.frame_id = self.utm_frame
            transformStamped.child_frame_id = buoy_frame
            transformStamped.header.stamp = rospy.Time.now()
            self.static_tf_bc.sendTransform(transformStamped)

    def construct_buoy_markers(self):
        if len(self.buoy_positions_sat) == 0:
            return

        # ===== Buoys ====
        self.buoy_positions_utm = []  # Used mostly for debugging
        self.buoy_marker_list = []  # Used later for the forming the lines
        self.buoy_markers = MarkerArray()
        for i, buoy in enumerate(self.buoy_positions_sat):
            buoy_utm = utm.fromLatLong(buoy.latitude, buoy.longitude)
            self.buoy_positions_utm.append([buoy_utm.easting, buoy_utm.northing, 0.0])

            marker = Marker()
            marker.header.frame_id = self.utm_frame
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.lifetime = rospy.Duration(0)
            marker.pose.position.x = buoy_utm.easting
            marker.pose.position.y = buoy_utm.northing
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = self.marker_scale
            marker.scale.y = self.marker_scale
            marker.scale.z = self.marker_scale
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.buoy_marker_list.append(marker)
            self.buoy_markers.markers.append(marker)

        # ===== Ropes =====
        if self.ropes is not None and len(self.ropes) > 0:
            self.rope_inner_markers = MarkerArray()  # intermediate buoys on each rope
            self.rope_outer_markers = MarkerArray()  # end buoys of each rope

            self.rope_lines = MarkerArray()

            for rope_ind, rope in enumerate(self.ropes):
                # Create points of the line
                point_start = Point(*self.buoy_positions_utm[rope[0]])
                point_end = Point(*self.buoy_positions_utm[rope[1]])

                # Rope as a line
                # === Construct rope line msg ===
                rope_line_marker = Marker()
                rope_line_marker.header.frame_id = self.utm_frame
                rope_line_marker.type = Marker.LINE_STRIP
                rope_line_marker.action = Marker.ADD
                rope_line_marker.id = rope_ind
                rope_line_marker.scale.x = 1.0  # Line width
                rope_line_marker.color.r = 1.0  # Line color (red)
                rope_line_marker.color.a = 1.0  # Line transparency (opaque)
                rope_line_marker.points = [point_start, point_end]
                rope_line_marker.pose.orientation.w = 1

                self.rope_lines.markers.append(rope_line_marker)

                # Rope as a line of buoys
                rope_buoys = self.calculate_rope_buoys(point_start, point_end)

                self.rope_outer_markers.markers.append(self.buoy_marker_list[rope[0]])
                self.rope_outer_markers.markers.append(self.buoy_marker_list[rope[1]])

                for buoy_ind, rope_buoy in enumerate(rope_buoys):
                    marker = Marker()
                    marker.header.frame_id = self.utm_frame
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.id = 10 * rope_ind + buoy_ind
                    marker.lifetime = rospy.Duration(0)
                    marker.pose.position.x = rope_buoy.x
                    marker.pose.position.y = rope_buoy.y
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.x = 0
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                    marker.pose.orientation.w = 1
                    marker.scale.x = self.marker_scale * 0.75
                    marker.scale.y = self.marker_scale * 0.75
                    marker.scale.z = self.marker_scale * 0.75
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0

                    # # Create the line segment points
                    # line_points = [point_start, point_end]
                    #
                    # # Set the line points
                    # marker.points = line_points

                    # Append the marker to the MarkerArray
                    self.rope_inner_markers.markers.append(marker)
                    print(f'|MAP_MARKER| Rope {rope_ind} added')


    def sam_gps(self, msg):
        self.last_gps_msg = msg

    def publish_tf(self):

        # Two measurements per second
        self.rate = rospy.Rate(50)

        for i in range(len(self.lats)):
            print(f'Publishing - {i}')

            navsatfix = NavSatFix()
            navsatfix.latitude = self.lats[i]
            navsatfix.longitude = self.longs[i]

            self.publisher_transform(navsatfix)
            rospy.sleep(0.1)

    def publish_markers(self):
        r = rospy.Rate(self.marker_rate)
        while not rospy.is_shutdown():
            if self.buoy_markers is not None:
                self.marker_pub.publish(self.buoy_markers)

            if self.rope_inner_markers is not None:
                self.rope_marker_pub.publish(self.rope_inner_markers)
                self.rope_lines_pub.publish(self.rope_lines)

            if self.rope_outer_markers is not None:
                self.rope_outer_marker_pub.publish(self.rope_outer_markers)

            r.sleep()

    def calculate_rope_buoys(self, start_point, end_point):
        positions = []

        # Calculate the step size between each position
        step_size = 1.0 / (self.n_buoys_per_rope + 1)

        # Calculate the vector difference between start and end points
        delta = Point()
        delta.x = end_point.x - start_point.x
        delta.y = end_point.y - start_point.y

        # Calculate the intermediate positions
        for i in range(1, self.n_buoys_per_rope + 1):
            position = Point()

            # Calculate the position based on the step size and delta
            position.x = start_point.x + i * step_size * delta.x
            position.y = start_point.y + i * step_size * delta.y
            position.z = 0.0

            positions.append(position)

        return positions


if __name__ == "__main__":
    rospy.init_node('gps_node', anonymous=False)
    rate = rospy.Rate(1)

    check_server = PubBuoyTf()
    check_server.publish_tf()
    if None in [check_server.buoy_markers, check_server.rope_inner_markers, check_server.rope_inner_markers]:
        check_server.construct_buoy_markers()
    check_server.publish_markers()

    rospy.spin()
