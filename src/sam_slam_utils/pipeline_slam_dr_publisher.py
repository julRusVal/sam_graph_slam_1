import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs

import gtsam

import numpy as np


class pipeline_sim_dr_publisher:
    def __init__(self):
        rospy.init_node('pipeline_sim_dr_publisher', anonymous=True)
        print("Starting: pipeline_sim_dr_publisher")

        # ===== Topics =====
        self.gt_topic = '/sam/sim/odom'
        self.imu_topic = '/sam0/core/imu'  # /sam0/core/imu
        self.pub_dr_topic = '/dr_odom'
        self.pub_gt_topic = '/gt_odom'

        # ===== Frame and tf stuff =====
        self.map_frame = 'map'
        self.robot_frame = 'sam0_base_link'
        self.imu_frame = 'sam0_imu_link'

        # Set up TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)  # is this needed

        # Initialize variables
        self.current_pose = PoseStamped()
        self.current_twist = TwistStamped()
        self.update_current_pose_world_frame()
        self.initial_pose = self.current_pose  # used to initialize factor graph

        # Subscribe to simulated IMU topic
        self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)

        # Publishers
        self.dr_publisher = rospy.Publisher(self.pub_dr_topic, Odometry, queue_size=10)
        self.gt_publisher = rospy.Publisher(self.pub_gt_topic, Odometry, queue_size=10)

        # ===== Set up all the IMU preintegration stuff =====
        # IMU biases
        accel_bias = np.array([0.01, 0.01, 0.01])
        gyro_bias = np.array([0.01, 0.01, 0.01])
        self.imu_bias = gtsam.gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

        gravity = 0  # 9.818
        gravity_down = True
        accel_sigma = 0.1  # 1e-3  # units for stddev are σ = m/s²/√Hz
        gyro_sigma = 0.1  # 1e-3
        integration_sigma = 1e-7

        if gravity_down:
            self.pre_int_params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        else:
            self.pre_int_params = gtsam.PreintegrationParams.MakeSharedD(gravity)

        self.pre_int_params.setAccelerometerCovariance(accel_sigma ** 2 * np.eye(3))
        self.pre_int_params.setGyroscopeCovariance(gyro_sigma ** 2 * np.eye(3))
        self.pre_int_params.setIntegrationCovariance(integration_sigma ** 2 * np.eye(3))

        self.pre_int_measure = gtsam.PreintegratedImuMeasurements(self.pre_int_params, self.imu_bias)

        # Set up other variables
        self.last_imu_time = None
        self.last_factor_time = None
        self.imu_factor_update_time = 2

        # ===== IMU factor gtraph =====
        self.imu_graph = None
        self.imu_graph_initial_estimate = None
        self.imu_graph_current_estimate = None
        self.isam2_parameters = None
        self.isam2 = None
        self.x_keys = None  # keys for 3d poses
        self.x_stamps = None  # stamps for 3d poses
        self.v_keys = None  # keys for velocities
        self.b_keys = None  # keys for biases

        # TODO Need to figure out the format of these
        # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
        self.init_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]))  # [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]
        self.init_velocity_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([5, 5, 5]))

        # ===== logging settings =====
        self.verbose_gtsam = True

        self.initialize_factor_graph()

    def initialize_factor_graph(self):
        self.imu_graph = gtsam.NonlinearFactorGraph()
        self.imu_graph_initial_estimate = gtsam.Values()

        self.x_keys = {}  # keys for 3d poses
        self.x_stamps = {}  # stamps for 3d poses
        self.v_keys = {}  # keys for velocities
        self.b_keys = {}  # keys for biases

        self.isam2_parameters = gtsam.ISAM2Params()
        self.isam2 = gtsam.ISAM2(self.isam2_parameters)

        # initial values
        graph_initial_pose = self.ros_pose_to_gtsam_pose3(self.initial_pose)
        graph_initial_stamp = self.initial_pose.header.stamp
        # TODO better initial velocity estimate
        graph_initial_velocity = np.array([0.001, 0.001, 0.001])

        # generate new keys
        self.x_keys[0] = gtsam.symbol('x', 0)
        self.v_keys[0] = gtsam.symbol('v', 0)
        self.b_keys[0] = gtsam.symbol('b', 0)

        # Record stamp
        self.x_stamps[0] = graph_initial_stamp

        # Add prior factors
        self.imu_graph.add(
            gtsam.PriorFactorPose3(self.x_keys[0], graph_initial_pose, self.init_pose_noise))
        self.imu_graph.add(
            gtsam.PriorFactorVector(self.v_keys[0], graph_initial_velocity, self.init_velocity_noise))

        self.imu_graph_initial_estimate.insert(self.b_keys[0], self.imu_bias)
        self.imu_graph_initial_estimate.insert(self.x_keys[0], graph_initial_pose)
        self.imu_graph_initial_estimate.insert(self.v_keys[0], graph_initial_velocity)

        self.isam2.update(self.imu_graph, self.imu_graph_initial_estimate)
        self.imu_graph_current_estimate = self.isam2.calculateEstimate()

        self.imu_graph_initial_estimate.clear()

        self.last_imu_time = rospy.Time.now()
        self.last_factor_time = rospy.Time.now()

    def imu_callback(self, imu_msg: Imu):

        # for simulated data current pose is ground truth
        self.update_current_pose_world_frame()
        # Publish Dead Reckoning pose
        self.publish_gt_pose()

        if self.imu_graph is not None:
            dt = (rospy.Time.now() - self.last_imu_time).to_sec()
            if dt == 0.0:
                return

            # print("GTSAM stuff start")
            # TODO Look into z accel
            measured_accel = np.array([imu_msg.linear_acceleration.x,
                                       imu_msg.linear_acceleration.y,
                                       0.01])  # imu_msg.linear_acceleration.z
            measured_angular = np.array([imu_msg.angular_velocity.x,
                                         imu_msg.angular_velocity.y,
                                         imu_msg.angular_velocity.z])

            dt = (rospy.Time.now() - self.last_imu_time).to_sec()

            self.last_imu_time = rospy.Time.now()

            # TODO Need to add some noise to measurements

            # Integrate incoming imu measurements
            self.pre_int_measure.integrateMeasurement(measuredAcc=measured_accel,
                                                      measuredOmega=measured_angular,
                                                      deltaT=dt)

            # Added Imu factors at specified rate
            dt_factor = (rospy.Time.now() - self.last_factor_time).to_sec()
            if dt_factor >= self.imu_factor_update_time:
                print("GTSAM adding factor")
                current_index = len(self.x_keys)
                self.x_stamps[current_index] = imu_msg.header.stamp  # Add stamp

                # Generate keys for the new factors and values
                self.x_keys[current_index] = gtsam.symbol('x', current_index)
                self.v_keys[current_index] = gtsam.symbol('v', current_index)

                factor = gtsam.ImuFactor(self.x_keys[current_index - 1], self.v_keys[current_index - 1],
                                         self.x_keys[current_index], self.v_keys[current_index],
                                         self.b_keys[0], self.pre_int_measure)
                # Tyr to add to a seperate graph
                new_graph = gtsam.NonlinearFactorGraph()
                new_graph.add(factor)
                # add to existing factor graph
                # self.imu_graph.add(factor)

                # Update factor time
                self.last_factor_time = rospy.Time.now()

                # Find initial values for the next step
                last_pose = self.imu_graph_current_estimate.atPose3(self.x_keys[current_index - 1])
                last_velocity = self.imu_graph_current_estimate.atVector(self.v_keys[current_index - 1])
                last_nav_state = gtsam.NavState(pose=last_pose, v=last_velocity)

                # construct initial estimate of the next state
                # TODO Check if predict is the correct method
                next_nav_state = self.pre_int_measure.predict(last_nav_state, self.imu_bias)
                next_pose = next_nav_state.pose()
                next_velocity = next_nav_state.velocity()

                self.imu_graph_initial_estimate.insert(self.x_keys[current_index], next_pose)
                self.imu_graph_initial_estimate.insert(self.v_keys[current_index], next_velocity)

                # Update and optimize isma2
                try:
                    # Testing
                    self.isam2.update(new_graph, self.imu_graph_initial_estimate)
                    # Original
                    # self.isam2.update(self.imu_graph, self.imu_graph_initial_estimate)
                    self.imu_graph_current_estimate = self.isam2.calculateEstimate()
                except Exception as e:
                    print(f"Error optimizing at step {current_index}: {e}")

                # Reset initial estimate
                self.imu_graph_initial_estimate.clear()

                # reset preintegrator
                self.pre_int_measure.resetIntegration()

                if self.verbose_gtsam:
                    print(f'Adding factor - {current_index}')
                    print(f'Measured accel: {measured_accel}')
                    print(f'Measured angular: {measured_angular}')
                    # print(f'Integrated: {self.pre_int_measure.preintegrated()}')
                    print(f"last position: {last_pose.translation()}")
                    print(f"Current position: {next_pose.translation()}")

            # print("GTSAM stuff end")

    def update_current_pose_world_frame(self):
        # Get the current transform from the base_link to the world frame
        try:
            # ( to_frame, from_frame, ...
            transform = self.tf_buffer.lookup_transform(self.map_frame,
                                                        self.robot_frame,
                                                        rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Failed to look up transform.")
            return

        # Update the current pose using the transform and orientation
        self.current_pose.header.stamp = transform.header.stamp
        self.current_pose.header.frame_id = self.map_frame
        self.current_pose.pose.position.x = transform.transform.translation.x
        self.current_pose.pose.position.y = transform.transform.translation.y
        self.current_pose.pose.position.z = transform.transform.translation.z
        self.current_pose.pose.orientation.x = transform.transform.rotation.x
        self.current_pose.pose.orientation.y = transform.transform.rotation.y
        self.current_pose.pose.orientation.z = transform.transform.rotation.z
        self.current_pose.pose.orientation.w = transform.transform.rotation.w

    def publish_gt_pose(self):

        # Create an Odometry message for the Dead Reckoning pose
        gt_pose_msg = Odometry()
        gt_pose_msg.header = self.current_pose.header
        gt_pose_msg.pose.pose = self.current_pose.pose
        # dr_pose_msg.twist.twist = self.current_twist

        # Publish the Dead Reckoning pose
        self.gt_publisher.publish(gt_pose_msg)

    # ===== Transforms and poses =====
    # See sam_slam_ros_classes.py

    # ===== ROS <--> GTSAM =====
    @staticmethod
    def ros_pose_to_gtsam_pose3(pose: PoseStamped):

        rot3 = gtsam.Rot3.Quaternion(w=pose.pose.orientation.w,
                                     x=pose.pose.orientation.x,
                                     y=pose.pose.orientation.y,
                                     z=pose.pose.orientation.z)

        return gtsam.Pose3(rot3, [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])


if __name__ == '__main__':
    try:
        dr_publisher_node = pipeline_sim_dr_publisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
