#!/usr/bin/env python3

import os

import numpy as np
import open3d as o3d  # Used for processing point cloud data

import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion
from sam_slam_utils import process_pointcloud2
from tf.transformations import (quaternion_matrix, compose_matrix, euler_from_quaternion, inverse_matrix,
                                quaternion_from_euler)
from visualization_msgs.msg import Marker


def stamped_transform_to_homogeneous_matrix(transform_stamped: TransformStamped):
    # Extract translation and quaternion components from the TransformStamped message
    translation = [transform_stamped.transform.translation.x,
                   transform_stamped.transform.translation.y,
                   transform_stamped.transform.translation.z]
    quaternion = [transform_stamped.transform.rotation.x,
                  transform_stamped.transform.rotation.y,
                  transform_stamped.transform.rotation.z,
                  transform_stamped.transform.rotation.w]

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = compose_matrix(
        translate=translation,
        angles=euler_from_quaternion(quaternion)
    )

    return homogeneous_matrix


def check_homogeneous_matrix(homogeneous_matrix):
    R = homogeneous_matrix[0:3, 0:3]
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0):
        print("Homogeneous matrix not pure rotation matrix")


def stamped_transform_to_rotation_matrix(transform_stamped: TransformStamped):
    # Convert quaternion to rotation matrix using tf2
    rotation_matrix = quaternion_matrix([transform_stamped.transform.rotation.x,
                                         transform_stamped.transform.rotation.y,
                                         transform_stamped.transform.rotation.z,
                                         transform_stamped.transform.rotation.w])

    return rotation_matrix


class PointCloudSaver:
    def __init__(self, topic='/sam0/mbes/odom/bathy_points',
                 save_location='', save_timeout=10):
        # Data source setting
        self.point_cloud_topic = topic

        # Data frame setting
        self.data_frame = 'odom'  # 'map' from of given mbes data
        self.robot_frame = 'sam0'  # 'sam0_base_link' transform into robot frame

        # Data recording setting
        self.save_timeout = rospy.Duration.from_sec(save_timeout)
        self.last_data_time = None
        self.data_written = False
        self.stacked_pc_original = None
        self.stacked_pc_transformed = None

        # Data saving settings
        if os.path.isdir(save_location):
            self.save_location = save_location + '/'
        else:
            self.save_location = ''

        # Initialize the saver node and tf
        print("Initializing point cloud saver")
        rospy.init_node('point_cloud_saver', anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Update time
        self.min_update_time = 1  # rospy.get_param('min_update_time')

        # Set up subscriptions
        rospy.Subscriber(self.point_cloud_topic, pc2.PointCloud2, self.point_cloud_callback, queue_size=1)

        # Set up a timer to check for data saving periodically
        rospy.Timer(self.save_timeout, self.save_timer_callback)

        self.detection_pub = rospy.Publisher('/detection_marker', Marker, queue_size=10)

    def point_cloud_callback(self, msg):
        time_now = rospy.Time.now()

        # handle first message and regulate the rate that message are processed
        if self.last_data_time is None:
            print("Pointcloud2 data received")
            self.last_data_time = time_now
        elif (time_now - self.last_data_time).to_sec() < self.min_update_time:
            return
        else:
            # Update the last received data time
            self.last_data_time = time_now

        # Find the transform to move pointcloud from odom/map to the robots frame
        try:
            # ( to_frame, from_frame, ...
            transform = self.tf_buffer.lookup_transform(self.robot_frame,
                                                        self.data_frame,
                                                        rospy.Time(0),
                                                        rospy.Duration(1, int(0.1 * 1e9)))

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to look up transform")
            print("Failed to look up transform")
            return

        # Convert the ROS transform to a 4x4 homogeneous transform
        homogeneous_transform = stamped_transform_to_homogeneous_matrix(transform)
        check_homogeneous_matrix(homogeneous_transform)
        inverse_homogeneous_transform = inverse_matrix(homogeneous_transform)

        # Convert PointCloud2 message to NumPy array
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_array = np.array(list(pc_data))

        # Convert pointcloud to od3 format
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_array)

        # Apply transform to point cloud
        pc_o3d.transform(homogeneous_transform)

        # save both the original and transformed data
        pc_array_transformed = np.asarray(pc_o3d.points)

        detector = process_pointcloud2.process_pointcloud_data(pc_array_transformed)
        if detector.detection_coords_world.size != 3:
            print("No detection!!")

        detection_homo = np.vstack([detector.detection_coords_world.reshape(3, 1),
                                   np.array([1])])

        detection_world = np.matmul(inverse_homogeneous_transform, detection_homo)

        # self.publish_detection_marker(detector.detection_coords_world, self.robot_frame)
        self.publish_detection_marker(detection_world, self.data_frame)
        print(f"Raw: {detector.detection_coords_world}")
        print(f"Transformed: {detection_world}")

        # # Store the point cloud data, original and transformed
        # # TODO can these get out of sync? check?
        # # Original
        # if self.stacked_pc_original is None:
        #     self.stacked_pc_original = pc_array
        # else:
        #     self.stacked_pc_original = np.dstack([self.stacked_pc_original, pc_array])
        #
        # # Transformed
        # if self.stacked_pc_transformed is None:
        #     self.stacked_pc_transformed = pc_array_transformed
        # else:
        #     self.stacked_pc_transformed = np.dstack([self.stacked_pc_transformed, pc_array_transformed])

    def save_timer_callback(self, event):
        # Check if no new data has been received for the specified interval
        if self.last_data_time is None:
            return
        if rospy.Time.now() - self.last_data_time > self.save_timeout:
            # Save point cloud data to a CSV file
            # self.save_stacked_point_cloud()
            rospy.signal_shutdown("Script shutting down")

    def save_stacked_point_cloud(self):
        if self.stacked_pc_original is not None and self.stacked_pc_transformed is not None:
            original_name = f"{self.save_location}pc_original.npy"
            transformed_name = f"{self.save_location}pc_transformed.npy"

            np.save(original_name, self.stacked_pc_original)
            np.save(transformed_name, self.stacked_pc_transformed)

            self.data_written = True

            print(f"Stacked point cloud saved")
        else:
            print("No point cloud data to save.")

        if self.data_written:
            rospy.signal_shutdown("Script shutting down")

    def publish_detection_marker(self, detection, frame):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """
        heading_quat = quaternion_from_euler(0, 0, 0)
        heading_quaternion_type = Quaternion(*heading_quat)

        detection_flat = detection.reshape(-1,)

        # Estimated detection location
        marker = Marker()
        marker.header.frame_id = frame
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.lifetime = rospy.Duration(0)
        marker.pose.position.x = detection_flat[0]
        marker.pose.position.y = detection_flat[1]
        marker.pose.position.z = detection_flat[2]
        marker.pose.orientation = heading_quaternion_type
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.detection_pub.publish(marker)
        print("Published marker")


if __name__ == '__main__':
    try:
        # Change '/your/point_cloud_topic' to the actual point cloud topic you want to subscribe to
        # Change 'point_cloud_data.csv' to the desired file name
        # Change 10 to the desired save interval in seconds
        point_cloud_subscriber = PointCloudSaver(topic='/sam0/mbes/odom/bathy_points',
                                                 save_location='/home/julian/catkin_ws/src/sam_slam/processing scripts/data',
                                                 save_timeout=5)

        # Run the ROS node
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
