#!/usr/bin/env python

import os
import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import csv
import tf2_ros
from tf.transformations import quaternion_matrix, compose_matrix, euler_from_quaternion
from geometry_msgs.msg import TransformStamped
import open3d as o3d  # Used for processing point cloud data


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

        # Set up subscriptions
        rospy.Subscriber(self.point_cloud_topic, pc2.PointCloud2, self.point_cloud_callback)

        # Set up a timer to check for data saving periodically
        rospy.Timer(self.save_timeout, self.save_timer_callback)

    def point_cloud_callback(self, msg):
        if self.last_data_time is None:
            print("Pointcloud2 data received")

        # Update the last received data time
        self.last_data_time = rospy.Time.now()

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

        # Store the point cloud data, original and transformed
        # TODO can these get out of sync? check?
        # Original
        if self.stacked_pc_original is None:
            self.stacked_pc_original = pc_array
        else:
            self.stacked_pc_original = np.dstack([self.stacked_pc_original, pc_array])

        # Transformed
        if self.stacked_pc_transformed is None:
            self.stacked_pc_transformed = pc_array_transformed
        else:
            self.stacked_pc_transformed = np.dstack([self.stacked_pc_transformed, pc_array_transformed])

    def save_timer_callback(self, event):
        # Check if no new data has been received for the specified interval
        if self.last_data_time is None:
            return
        if rospy.Time.now() - self.last_data_time > self.save_timeout:
            # Save point cloud data to a CSV file
            # self.save_to_csv()
            self.save_stacked_point_cloud()

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
