#!/usr/bin/env python3

import gtsam
import numpy as np

import rospy
import tf
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

"""
Implements dr and gt publisher - dr based on gt
gtsam objects, Pose3, are used to maintain the dr pose
"""


def ros_pose_to_gtsam_pose3_and_stamp(pose: PoseStamped):
    rot3 = gtsam.Rot3.Quaternion(w=pose.pose.orientation.w,
                                 x=pose.pose.orientation.x,
                                 y=pose.pose.orientation.y,
                                 z=pose.pose.orientation.z)

    stamp = pose.header.stamp
    return gtsam.Pose3(rot3, [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]), stamp


def gtsam_pose3_to_ros_pose3(pose: gtsam.Pose3, stamp=None, frame_id=None):
    quaternion_wxyz = pose.rotation().toQuaternion()
    translation = pose.translation()

    new_stamped_pose = PoseStamped()
    if stamp is not None:
        new_stamped_pose.header.stamp = stamp
    if frame_id is not None:
        new_stamped_pose.header.frame_id = frame_id
    new_stamped_pose.pose.position.x = translation[0]
    new_stamped_pose.pose.position.y = translation[1]
    new_stamped_pose.pose.position.z = translation[2]
    new_stamped_pose.pose.orientation.x = quaternion_wxyz.x()
    new_stamped_pose.pose.orientation.y = quaternion_wxyz.y()
    new_stamped_pose.pose.orientation.z = quaternion_wxyz.z()
    new_stamped_pose.pose.orientation.w = quaternion_wxyz.w()

    return new_stamped_pose


class pipeline_sim_dr_gt_publisher:
    def __init__(self):
        rospy.init_node('pipeline_sim_dr_gt_publisher', anonymous=True)
        print("Starting: pipeline_sim_dr_gt_publisher")

        # ===== Topics =====
        self.gt_topic = '/sam/sim/odom'
        self.imu_topic = '/sam0/core/imu'
        self.pub_dr_topic = '/dr_odom'
        self.pub_gt_topic = '/gt_odom'

        # ===== Frame and tf stuff =====
        self.map_frame = 'odom'  # 'map'
        self.robot_frame = 'sam0_base_link'
        self.dr_tf_frame = 'dr_frame'  # Currently intended for the detector saving dr poses with pointcloud data

        # Set up TF broadcaster
        # self.tf_br = tf2_ros.TransformBroadcaster()  # was having problems!!! tuple has no attribute x
        self.tf_br = tf.TransformBroadcaster()  # Feels bad to mix tf ans tf2

        # Set up TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ===== Initialize variables =====
        self.current_gt_pose = None  # PoseStamped()
        self.last_gt_pose = None  # PoseStamped()

        self.gt_between = None
        self.dr_pose3 = None
        self.dr_stamp = None
        self.dr_frame = None  # parent frame of the dr odometry

        # Subscribe to simulated IMU topic
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)

        # Publishers
        self.dr_publisher = rospy.Publisher(self.pub_dr_topic, Odometry, queue_size=10)
        self.gt_publisher = rospy.Publisher(self.pub_gt_topic, Odometry, queue_size=10)

        # DR noise parameters
        self.add_noise = True
        self.bound_depth = True
        self.bound_pitch_roll = True

        # Initialization noise
        self.init_position_sigmas = np.array([1.0, 1.0, 1.0])  # x, y, z
        self.init_rotation_sigmas = np.array([np.pi / 1e3, np.pi / 1e3, np.pi / 50])  # roll, pitch, yaw

        # Step noise
        self.delta_position_sigmas = np.array([0.001, 0.001, 0.001])  # x, y, z - per second
        self.delta_rotation_sigmas = np.array([np.pi / 1e5, np.pi / 1e5, np.pi / 1e2])  # roll, pitch, yaw - per second

        self.depth_sigma = 0.1

        # ===== logging settings =====

    def imu_callback(self, imu_msg: Imu):
        # for simulated data current pose is ground truth
        self.update_current_pose_world_frame()

        # initial
        if self.last_gt_pose is None and self.current_gt_pose is not None:
            # generate current dr
            self.dr_pose3, self.dr_stamp = ros_pose_to_gtsam_pose3_and_stamp(self.current_gt_pose)
            self.dr_frame = self.map_frame

            if self.add_noise:
                self.add_noise_to_initial_pose()

            # Publish dr and gt
            self.publish_dr_pose()
            self.publish_gt_pose()

        # Normal running condition
        elif self.last_gt_pose is not None and self.current_gt_pose is not None:
            self.update_dr_pose3()

            # Publish dr and gt
            self.publish_dr_pose()
            self.publish_gt_pose()

    def update_current_pose_world_frame(self):
        # Get the current transform from the base_link to the world frame
        try:
            # ( to_frame, from_frame, ...
            transform = self.tf_buffer.lookup_transform(self.map_frame,
                                                        self.robot_frame,
                                                        rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to look up transform.")
            return

        # Define gt pose based on the above transform
        gt_pose = PoseStamped()

        gt_pose.header.stamp = transform.header.stamp
        gt_pose.header.frame_id = self.map_frame
        gt_pose.pose.position.x = transform.transform.translation.x
        gt_pose.pose.position.y = transform.transform.translation.y
        gt_pose.pose.position.z = transform.transform.translation.z
        gt_pose.pose.orientation.x = transform.transform.rotation.x
        gt_pose.pose.orientation.y = transform.transform.rotation.y
        gt_pose.pose.orientation.z = transform.transform.rotation.z
        gt_pose.pose.orientation.w = transform.transform.rotation.w

        self.last_gt_pose = self.current_gt_pose
        self.current_gt_pose = gt_pose

    def add_noise_to_initial_pose(self):
        if self.dr_pose3 is None:
            return
        # Translational noise
        position_noise = np.random.normal(0, self.init_position_sigmas)

        # Rotational noise
        roll_noise, pitch_noise, yaw_noise = np.random.normal(0, self.init_rotation_sigmas)
        # Still confused by ypr vs rpy :)
        rotation_noise = gtsam.Rot3.Ypr(yaw_noise, pitch_noise, roll_noise)

        noise_pose3 = gtsam.Pose3(rotation_noise, position_noise)

        noisy_pose3 = self.dr_pose3.compose(noise_pose3)
        self.dr_pose3 = noisy_pose3

        return

    def update_dr_pose3(self):
        # calculate between
        init_pose3, init_time = ros_pose_to_gtsam_pose3_and_stamp(self.last_gt_pose)
        final_pose3, final_time = ros_pose_to_gtsam_pose3_and_stamp(self.current_gt_pose)

        between_pose3 = init_pose3.between(final_pose3)
        dt = (final_time - init_time).to_sec()

        new_dr_pose = self.dr_pose3.compose(between_pose3)

        if self.add_noise:
            noise_pose3 = self.return_step_noise_pose3(dt)
            new_dr_pose = new_dr_pose.compose(noise_pose3)

        if self.bound_depth:
            # Determine depth values
            depth_noise = np.random.normal(0, self.depth_sigma)
            bounded_noisey_depth = final_pose3.z() + depth_noise

            # Update the depth, z, value
            new_translation = new_dr_pose.translation()
            new_translation[2] = bounded_noisey_depth

            # Reform Pose3
            rotation = new_dr_pose.rotation()
            new_dr_pose = gtsam.Pose3(rotation, new_translation)

        if self.bound_pitch_roll:
            # determine pitch and roll values
            roll_noise, pitch_noise, _ = np.random.normal(0, self.init_rotation_sigmas)  # roll, pitch, yaw
            _, pitch_current, roll_current = final_pose3.rotation().ypr()
            bounded_pitch = pitch_current + pitch_noise
            bounded_roll = roll_current + roll_noise
            additive_yaw = new_dr_pose.rotation().yaw()

            # form the newly bounded Rot3
            bounded_rotation = gtsam.Rot3.Ypr(additive_yaw, bounded_pitch, bounded_roll)

            # Reform Pose3
            translation = new_dr_pose.translation()
            new_dr_pose = gtsam.Pose3(bounded_rotation, translation)

        self.dr_pose3 = new_dr_pose
        self.dr_stamp = final_time

    def return_step_noise_pose3(self, dt):

        roll_noise, pitch_noise, yaw_noise = np.random.normal(0, self.delta_rotation_sigmas * np.sqrt(dt))  # r, p, y
        x_noise, y_noise, z_noise = np.random.normal(0, self.delta_position_sigmas * np.sqrt(dt))

        rotation_noise = gtsam.Rot3.Ypr(yaw_noise, pitch_noise, roll_noise)  # yaw, pitch, roll
        translation_noise = gtsam.Point3(x_noise, y_noise, z_noise)
        pose_noise = gtsam.Pose3(rotation_noise, translation_noise)

        return pose_noise

    def publish_gt_pose(self):

        # Create an Odometry message for the Dead Reckoning pose
        gt_pose_msg = Odometry()
        gt_pose_msg.header = self.current_gt_pose.header
        gt_pose_msg.pose.pose = self.current_gt_pose.pose

        # Publish the Dead Reckoning pose
        self.gt_publisher.publish(gt_pose_msg)

    def publish_dr_pose(self):
        dr_pose_msg = Odometry()
        current_dr_pose = gtsam_pose3_to_ros_pose3(self.dr_pose3, self.dr_stamp, self.dr_frame)
        dr_pose_msg.header = current_dr_pose.header
        dr_pose_msg.pose.pose = current_dr_pose.pose

        # Publish the Dead Reckoning pose
        self.dr_publisher.publish(dr_pose_msg)
        self.publish_dr_transform(dr_pose_msg=dr_pose_msg)

    def publish_dr_transform(self, dr_pose_msg: Odometry):
        """
        Publish the transform of the dr pose
        :param dr_pose_msg: Odometry of the dr pose
        :return:
        """
        # Generate content for transform
        dr_timestamp = dr_pose_msg.header.stamp
        dr_frame = dr_pose_msg.header.frame_id
        dr_trans = (dr_pose_msg.pose.pose.position.x,
                    dr_pose_msg.pose.pose.position.y,
                    dr_pose_msg.pose.pose.position.z)
        dr_rot = (dr_pose_msg.pose.pose.orientation.x,
                  dr_pose_msg.pose.pose.orientation.y,
                  dr_pose_msg.pose.pose.orientation.z,
                  dr_pose_msg.pose.pose.orientation.w)

        # Form transform for the tf2_ros broadcaster
        # dr_t = TransformStamped()
        # dr_t.header.stamp = dr_timestamp
        # dr_t.header.frame_id = dr_frame
        # dr_t.child_frame_id = self.dr_tf_frame
        # dr_t.transform.translation = dr_trans
        # dr_t.transform.rotation = dr_rot

        # Attempt to broadcast
        try:
            # self.tf_br.sendTransform(transform=dr_t)  # tf2_ros broadcaster accepts a TransformStamped
            self.tf_br.sendTransform(translation=dr_trans,
                                     rotation=dr_rot,
                                     time=dr_timestamp,
                                     child=self.dr_tf_frame,
                                     parent=dr_frame)

        except rospy.ROSException as e:
            rospy.logerr('Error broadcasting tf transform: {}'.format(str(e)))


if __name__ == '__main__':
    try:
        dr_publisher_node = pipeline_sim_dr_gt_publisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
