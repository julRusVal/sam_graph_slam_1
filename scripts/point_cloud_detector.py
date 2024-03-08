#!/usr/bin/env python3

import os

import numpy as np
import open3d as o3d  # Used for processing point cloud data

import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion, Pose
from sam_slam_utils import process_pointcloud2
from tf.transformations import (quaternion_matrix, compose_matrix, euler_from_quaternion, inverse_matrix,
                                quaternion_from_euler)
from visualization_msgs.msg import Marker
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from sss_object_detection.consts import ObjectID

'''
Basic detector with the ability to save point clouds for offline processing and analysis
'''


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
                 save_data=False,
                 save_location='', save_timeout=10):
        self.robot_name = 'sam'

        # Data source setting
        self.point_cloud_topic = topic

        # Data frame setting
        self.data_frame = 'odom'  # 'map' from of given mbes data
        self.robot_frame = 'sam0'  # 'sam0_base_link' transform into robot frame

        # Data recording setting
        self.save_data = save_data
        self.save_timeout = rospy.Duration.from_sec(save_timeout)
        self.last_data_time = None
        self.data_written = False
        self.stacked_pc_original = None
        self.stacked_pc_transformed = None
        self.stacked_world_to_robot_transform = None
        self.stacked_robot_to_world_transform = None

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

        # Set up publishers for the detections
        # 1) marker for visualization
        # 2) message for the online sam slam listener
        self.detection_marker_pub = rospy.Publisher('/detection_marker', Marker, queue_size=10)
        self.detection_pub = rospy.Publisher(f'/{self.robot_name}/payload/sidescan/detection_hypothesis',
                                             Detection2DArray,
                                             queue_size=2)

        self.confidence_dummy = 0.5  # Confidence is just a dummy value for now

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

        # Process pointcloud data
        detector = process_pointcloud2.process_pointcloud_data(pc_array_transformed)

        if detector.detection_coords_world.size != 3:
            print("No detection!!")
        else:
            # publish a marker indicating the position of the pipeline detection, world coords
            detection_homo = np.vstack([detector.detection_coords_world.reshape(3, 1),
                                        np.array([1])])

            detection_world = np.matmul(inverse_homogeneous_transform, detection_homo)

            # self.publish_detection_marker(detector.detection_coords_world, self.robot_frame)
            self.publish_detection_marker(detection_world, self.data_frame)

            # Debugging print outs
            # print(f"Raw: {detector.detection_coords_world}")
            # print(f"Transformed: {detection_world}")

            # Publish the message for SLAM
            self.publish_mbes_pipe_detection(detector.detection_coords_world, self.confidence_dummy, msg.header.stamp)
        # # Store the point cloud data, original and transformed
        if self.save_data:
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

            # record the transform
            if self.stacked_world_to_robot_transform is None:
                self.stacked_world_to_robot_transform = homogeneous_transform
            else:
                self.stacked_world_to_robot_transform = np.dstack(
                    [self.stacked_world_to_robot_transform, homogeneous_transform])

            if self.stacked_robot_to_world_transform is None:
                self.stacked_robot_to_world_transform = inverse_homogeneous_transform
            else:
                self.stacked_robot_to_world_transform = np.dstack(
                    [self.stacked_robot_to_world_transform, inverse_homogeneous_transform])

    def save_timer_callback(self, event):
        # Check if no new data has been received for the specified interval
        if self.last_data_time is None:
            return
        if rospy.Time.now() - self.last_data_time > self.save_timeout:
            # Save point cloud data to a CSV file
            if self.save_data:
                self.save_stacked_point_cloud()
            rospy.signal_shutdown("Script shutting down")

    def save_stacked_point_cloud(self):
        if self.stacked_pc_original is None or self.stacked_pc_transformed is None:
            print("No point cloud data to save.")
        elif self.stacked_world_to_robot_transform is None or self.stacked_robot_to_world_transform is None:
            print("No transforms to save.")
        else:
            original_name = f"{self.save_location}pc_original.npy"
            transformed_name = f"{self.save_location}pc_transformed.npy"
            world_to_local_name = f"{self.save_location}world_to_local.npy"
            local_to_world_name = f"{self.save_location}local_to_world.npy"

            np.save(original_name, self.stacked_pc_original)
            np.save(transformed_name, self.stacked_pc_transformed)
            np.save(world_to_local_name, self.stacked_world_to_robot_transform)
            np.save(local_to_world_name, self.stacked_robot_to_world_transform)

            self.data_written = True

            print(f"Stacked point cloud saved")

        if self.data_written:
            rospy.signal_shutdown("Script shutting down")

    def publish_detection_marker(self, detection, frame):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """
        heading_quat = quaternion_from_euler(0, 0, 0)
        heading_quaternion_type = Quaternion(*heading_quat)

        detection_flat = detection.reshape(-1, )

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

        self.detection_marker_pub.publish(marker)
        # print("Published marker")

    def publish_mbes_pipe_detection(self, detection_coords,
                                    detection_confidence,
                                    stamp,
                                    seq_id=None):

        """
        Publishes message for mbes detection given [x, y, z] coords.

        :param detection_coords: numpy array of x, y, z coordinates
        :param detection_confidence: float confidence, in some cases this is used for data association (but not here)
        :param stamp: stamp of the message
        :param seq_id: seq_id can be used for data association
        :return:
        """

        # Convert detection coords into a Pose()
        detection_flat = detection_coords.reshape(-1, )
        detection_pose = Pose()
        detection_pose.position.x = detection_flat[0]
        detection_pose.position.y = detection_flat[1]
        detection_pose.position.z = detection_flat[2]

        # Form the message
        # Detection2DArray()
        # -> list of Detection2D()
        # ---> list of ObjectHypothesisWithPose()
        # -----> Contains: id, score, pose
        # pose is PoseWithCovariance()

        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = self.robot_frame
        detection_array_msg.header.stamp = stamp

        # Individual detection, Detection2D, currently only a single detection per call will be published
        detection_msg = Detection2D()
        detection_msg.header = detection_array_msg.header

        # Define single ObjectHypothesisWithPose
        object_hypothesis = ObjectHypothesisWithPose()
        object_hypothesis.id = ObjectID.PIPE.value

        if seq_id is not None:
            object_hypothesis.score = seq_id
        else:
            object_hypothesis.score = detection_confidence
        object_hypothesis.pose.pose = detection_pose

        # Append the to form a complete message
        detection_msg.results.append(object_hypothesis)
        detection_array_msg.detections.append(detection_msg)

        self.detection_pub.publish(detection_array_msg)
        print("Published")
        return


if __name__ == '__main__':
    try:
        point_cloud_subscriber = PointCloudSaver(topic='/sam0/mbes/odom/bathy_points',
                                                 save_data=True,
                                                 save_location='/home/julian/catkin_ws/src/sam_slam/processing scripts/data',
                                                 save_timeout=5)

        # Run the ROS node
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
