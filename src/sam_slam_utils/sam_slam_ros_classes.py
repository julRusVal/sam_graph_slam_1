#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo

# cv_bridge and cv2 to convert and save images
# from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys

from sam_slam_utils.sam_slam_helper_funcs import show_simple_graph_2d
from sam_slam_utils.sam_slam_helper_funcs import write_array_to_csv


class sam_slam_listener:
    """
    This class defines the behavior of the slam_listener node
    Dead reckoning (dr) and ground truth data is saved [x, y, z, q_w, q_x, q_y, q_z] in the map frame
    Note I about the gt data, there are two potential sources of gt data
    - the topic: /sam/sim/odom
    - the frame attached to the simulation: gt/sam/base_link (currently used)
    Note II about the gt data, I have tried to transform all the poses to the map frame but even after this I need to
    invert the sign of the x-axis and corrected_heading = pi - original_heading

    Detections are saved in two lists. {There is no need for both}
    detections format: [x_map, y_map, z_map, q_w, q_x, q_y, q_z, corresponding dr id, score]
    detections_graph format: [x_map, y_map, z_map, x_rel, y_rel, z_vel, corresponding dr id]

    Online
    If an online_graph object is passed to the listener it will be updated at every detection
    - dr_callback: first update and odometry updates
    - det_callback: update
    - buoy_callback: send buoy info to online graph
    - time_check_callback: Save results when there is no longer any dr update

    """

    def __init__(self, gt_top_name, dr_top_name, det_top_name, buoy_top_name, frame_name, path_name=None,
                 online_graph=None):
        # Topic names
        self.gt_topic = gt_top_name
        self.dr_topic = dr_top_name
        self.det_topic = det_top_name
        self.buoy_topic = buoy_top_name

        # Frame names: For the most part everything is transformed to the map frame
        self.frame = frame_name
        self.gt_frame_id = 'gt/sam/base_link'

        # tf stuff
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Buoy positions
        self.buoys = None

        # Online SLAM w/ iSAM2
        self.online_graph = online_graph

        # Logging
        # Raw logging, occurs at the rate the data is received
        self.gt_poses = []
        self.dr_poses = []
        self.detections = []

        # Graph logging
        """
        dr_callback will update at a set rate will also record ground truth pose
        det_callback will update all three
        # Current format: [index w.r.t. dr_poses_graph[], x, y, z]
        """
        self.gt_poses_graph = []
        self.dr_poses_graph = []
        self.detections_graph = []

        # File paths for logging
        if isinstance(path_name, str):
            self.dr_poses_graph_file_path = path_name + '/dr_poses_graph.csv'
            self.gt_poses_graph_file_path = path_name + '/gt_poses_graph.csv'
            self.detections_graph_file_path = path_name + '/detections_graph.csv'
            self.buoys_file_path = path_name + '/buoys.csv'
        else:
            self.dr_poses_graph_file_path = 'dr_poses_graph.csv'
            self.gt_poses_graph_file_path = 'gt_poses_graph.csv'
            self.detections_graph_file_path = 'detections_graph.csv'
            self.buoys_file_path = 'buoys.csv'

        # Timing and state
        self.last_time = rospy.Time.now()
        self.dr_updated = False
        self.gt_updated = False
        self.buoy_updated = False
        self.data_written = False
        self.update_time = 0.5

        # ===== Subscribers =====
        # Ground truth
        self.gt_subscriber = rospy.Subscriber(self.gt_topic,
                                              Odometry,
                                              self.gt_callback)

        # Dead reckoning
        self.dr_subscriber = rospy.Subscriber(self.dr_topic,
                                              Odometry,
                                              self.dr_callback)

        # Detections
        self.det_subscriber = rospy.Subscriber(self.det_topic,
                                               Detection2DArray,
                                               self.det_callback)

        # Buoys
        self.buoy_subscriber = rospy.Subscriber(self.buoy_topic,
                                                MarkerArray,
                                                self.buoy_callback)

        self.time_check = rospy.Timer(rospy.Duration(2),
                                      self.time_check_callback)

    # Subscriber callbacks
    def gt_callback(self, msg):
        """
        Call back for the ground truth subscription, msg is of type nav_msgs/Odometry.
        The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.frame)

        gt_position = transformed_pose.pose.position
        gt_quaternion = transformed_pose.pose.orientation

        self.gt_poses.append([gt_position.x, gt_position.y, gt_position.z,
                              gt_quaternion.w, gt_quaternion.x, gt_quaternion.y, gt_quaternion.z])

        self.gt_updated = True

    def dr_callback(self, msg):
        """
        Call back for the dead reckoning subscription, msg is of type nav_msgs/Odometry.
        The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        # transform odom to the map frame
        transformed_dr_pose = self.transform_pose(msg.pose,
                                                  from_frame=msg.header.frame_id,
                                                  to_frame=self.frame)

        dr_position = transformed_dr_pose.pose.position
        dr_quaternion = transformed_dr_pose.pose.orientation

        # Record dr poses in format compatible with GTSAM
        self.dr_poses.append([dr_position.x, dr_position.y, dr_position.z,
                              dr_quaternion.w, dr_quaternion.x, dr_quaternion.y, dr_quaternion.z])

        # Conditions for updating dr: (1) first time or (2) stale data or (3) online graph is still uninitialized
        time_now = rospy.Time.now()
        first_time_cond = not self.dr_updated and self.gt_updated
        stale_data_cond = self.dr_updated and (time_now - self.last_time).to_sec() > self.update_time
        if self.online_graph is not None:
            online_waiting_cond = self.gt_updated and self.online_graph.initial_pose_set is False
        else:
            online_waiting_cond = False

        if first_time_cond or stale_data_cond or online_waiting_cond:
            # Add to the dr and gt lists
            self.dr_poses_graph.append(self.dr_poses[-1])
            self.gt_poses_graph.append(self.get_gt_trans_in_map())

            # Update time and state
            self.last_time = time_now
            self.dr_updated = True

            # ===== Online first update =====
            if self.online_graph is not None and self.online_graph.initial_pose_set is False:
                # TODO
                print("First update")
                self.online_graph.add_first_pose(self.dr_poses_graph[-1], self.gt_poses_graph[-1])

            elif self.online_graph is not None:
                print("Odometry update")
                self.online_graph.online_update(self.dr_poses_graph[-1], self.gt_poses_graph[-1])

    def det_callback(self, msg):
        for det_ind, detection in enumerate(msg.detections):
            for res_ind, result in enumerate(detection.results):
                # Pose in base_link
                detection_position = result.pose
                # Convert to map
                transformed_pose = self.transform_pose(detection_position,
                                                       from_frame=msg.header.frame_id,
                                                       to_frame=self.frame)
                # Extract the position
                det_position = transformed_pose.pose.position

                # ===== Log data for the graph =====
                # First update dr and gr with the most current
                self.dr_poses_graph.append(self.dr_poses[-1])
                self.gt_poses_graph.append(self.get_gt_trans_in_map())
                # (OLD) self.gt_poses_graph.append(self.gt_poses[-1])

                # detection position:
                # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of dr_pose_graph]
                index = len(self.dr_poses_graph) - 1
                self.detections_graph.append([det_position.x,
                                              det_position.y,
                                              det_position.z,
                                              detection_position.pose.position.x,
                                              detection_position.pose.position.y,
                                              detection_position.pose.position.z,
                                              index])

                # ===== Online detection update =====
                if self.online_graph is not None:
                    # TODO
                    print("Detection update")
                    self.online_graph.online_update(self.dr_poses_graph[-1], self.gt_poses_graph[-1],
                                                    np.array((detection_position.pose.position.x,
                                                              detection_position.pose.position.y), dtype=np.float64))

    def buoy_callback(self, msg):
        if not self.buoy_updated:
            self.buoys = []
            for marker in msg.markers:
                self.buoys.append([marker.pose.position.x,
                                   marker.pose.position.y,
                                   marker.pose.position.z])

            if self.online_graph is not None:
                # TODO buoy info to online graph
                print("Online: buoy update")
                self.online_graph.buoy_setup(self.buoys)

            self.buoy_updated = True

    # Timer callback
    def time_check_callback(self, event):
        if not self.dr_updated:
            return
        delta_t = rospy.Time.now() - self.last_time
        if delta_t.to_sec() >= 1 and not self.data_written:
            print('Data written')
            self.write_data()
            self.data_written = True

            if self.online_graph is not None:
                # TODO Save final results
                print("Print online graph")
                show_simple_graph_2d(graph=self.online_graph.graph,
                                     x_keys=self.online_graph.x,
                                     b_keys=self.online_graph.b,
                                     values=self.online_graph.current_estimate,
                                     label="Online Graph")

        return

    # ===== Transforms =====
    def transform_pose(self, pose, from_frame, to_frame):
        trans = self.wait_for_transform(from_frame=from_frame,
                                        to_frame=to_frame)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame,
                                                        from_frame,
                                                        rospy.Time(),
                                                        rospy.Duration(1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))

        return trans

    def get_gt_trans_in_map(self):
        """
        Finds pose of the ground truth.
        First,the transform between the map and the ground truth frame.
        Second, the transform is applied to a null_pose located at the origin.
        Modifying the orientation of this pose might be need to prevent later
        processing on the ground truth
        Returns [ x, y, z, q_w, q_x, q_y, q_z]
        """

        trans = self.wait_for_transform(from_frame=self.gt_frame_id,
                                        to_frame=self.frame)

        null_pose = PoseStamped()
        null_pose.pose.orientation.w = 1.0
        pose_in_map = tf2_geometry_msgs.do_transform_pose(null_pose, trans)

        pose_list = [pose_in_map.pose.position.x,
                     pose_in_map.pose.position.y,
                     pose_in_map.pose.position.z,
                     pose_in_map.pose.orientation.w,
                     pose_in_map.pose.orientation.x,
                     pose_in_map.pose.orientation.y,
                     pose_in_map.pose.orientation.z]

        return pose_list

    # ===== iSAM2 =====

    # ===== Random utility methods =====
    def write_data(self):
        """
        Save all the relevant data
        """
        write_array_to_csv(self.dr_poses_graph_file_path, self.dr_poses_graph)
        write_array_to_csv(self.gt_poses_graph_file_path, self.gt_poses_graph)
        write_array_to_csv(self.detections_graph_file_path, self.detections_graph)
        write_array_to_csv(self.buoys_file_path, self.buoys)

        return


class sam_image_saver:
    def __init__(self, camera_down_top_name, camera_left_top_name, camera_right_top_name, buoy_top_name,
                 file_path=None):
        # ===== Set topic names ====== file paths for output =====
        # Down
        self.cam_down_image_topic = camera_down_top_name + '/image_color'
        self.cam_down_info_topic = camera_down_top_name + '/camera_info'
        # Left
        self.cam_left_image_topic = camera_left_top_name + '/image_color'
        self.cam_left_info_topic = camera_left_top_name + '/camera_info'
        # Right
        self.cam_right_image_topic = camera_right_top_name + '/image_color'
        self.cam_right_info_topic = camera_right_top_name + '/camera_info'
        # Buoys
        self.buoy_topic = buoy_top_name

        # ===== File paths for output =====
        self.file_path = file_path
        if self.file_path is None or not isinstance(file_path, str):
            file_path_prefix = ''
        else:
            file_path_prefix = self.file_path + '/'

        # Down
        self.down_info_file_path = file_path_prefix + 'down_info.csv'
        self.down_gt_file_path = file_path_prefix + 'down_gt.csv'
        # Left
        self.left_info_file_path = file_path_prefix + 'left_info.csv'
        self.left_gt_file_path = file_path_prefix + 'left_gt.csv'
        # Right
        self.right_info_file_path = file_path_prefix + 'right_info.csv'
        self.right_gt_file_path = file_path_prefix + 'right_gt.csv'
        # Buoy
        self.buoy_info_file_path = file_path_prefix + 'buoy_info.csv'

        # ===== Frame and tf stuff =====
        self.frame = 'map'
        self.gt_frame_id = 'gt/sam/base_link'
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ==== Local data =====
        # Camera ground_truth and information
        self.down_gt = []
        self.down_info = []
        self.down_times = []
        self.left_gt = []
        self.left_info = []
        self.left_times = []
        self.right_gt = []
        self.right_info = []
        self.left_times = []
        self.buoys = []

        # ===== Image processing =====
        # self.bridge = CvBridge()  # Not had problems

        # ===== States =====
        self.last_time = rospy.Time.now()
        self.tf_ready = False
        self.buoys_received = False
        self.image_received = False
        self.data_written = False

        # ===== Subscriptions =====
        # Down camera
        self.cam_down_image_subscriber = rospy.Subscriber(self.cam_down_image_topic,
                                                          Image,
                                                          self.down_image_callback)

        self.cam_down_info_subscriber = rospy.Subscriber(self.cam_down_info_topic,
                                                         CameraInfo,
                                                         self.down_info_callback)

        # Left camera
        self.cam_left_image_subscriber = rospy.Subscriber(self.cam_left_image_topic,
                                                          Image,
                                                          self.left_image_callback)

        self.cam_left_info_subscriber = rospy.Subscriber(self.cam_left_info_topic,
                                                         CameraInfo,
                                                         self.left_info_callback)

        # Right camera
        self.cam_right_image_subscriber = rospy.Subscriber(self.cam_right_image_topic,
                                                           Image,
                                                           self.right_image_callback)

        self.cam_right_info_subscriber = rospy.Subscriber(self.cam_right_info_topic,
                                                          CameraInfo,
                                                          self.right_info_callback)

        # Buoys
        self.buoy_subscriber = rospy.Subscriber(self.buoy_topic,
                                                MarkerArray,
                                                self.buoy_callback)

        # ===== Timers =====
        self.time_check = rospy.Timer(rospy.Duration(1),
                                      self.time_check_callback)

    # ===== Callbacks =====
    # Down
    def down_image_callback(self, msg):

        # record gt
        current = self.get_gt_trans_in_map()
        current.append(msg.header.seq)
        self.down_gt.append(current)

        print(f'Down image callback: {msg.header.seq}')
        print(current)

        # TODO get cvbridge working
        # Convert to cv format
        # try:
        #     # Convert your ROS Image message to OpenCV2
        #     cv2_img = self.bridge.imgmsg_to_cv2(msg)  # "bgr8"
        # except CvBridgeError:
        #     print('CvBridge Error')
        # else:
        #     # Save your OpenCV2 image as a jpeg
        #     # time = msg.header.stamp  # cast as string to use in name
        #     if self.file_path is None or not isinstance(self.file_path, str):
        #         save_path = f'{msg.header.seq}.jpg'
        #     else:
        #         save_path = self.file_path + f'/d:{msg.header.seq}.jpg'
        #     cv2.imwrite(save_path, cv2_img)

        # Convert with home-brewed conversion
        # https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
        cv2_img = self.imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.file_path is None or not isinstance(self.file_path, str):
            save_path = f'{msg.header.seq}.jpg'
        else:
            save_path = self.file_path + f'/down/d_{msg.header.seq}.jpg'
        cv2.imwrite(save_path, cv2_img)

        # Update state and timer
        self.image_received = True
        self.last_time = rospy.Time.now()

        return

    def down_info_callback(self, msg):
        if len(self.down_info) == 0:
            self.down_info.append(msg.K)
            self.down_info.append(msg.P)
            self.down_info.append([msg.width, msg.height])

    # Left
    def left_image_callback(self, msg):
        """
        Based on down_image_callback
        """
        # record gt
        current = self.get_gt_trans_in_map()
        current.append(msg.header.seq)
        self.left_gt.append(current)
        print(current)

        # Convert to cv2 format
        cv2_img = self.imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.file_path is None or not isinstance(self.file_path, str):
            save_path = f'{msg.header.seq}.jpg'
        else:
            save_path = self.file_path + f'/left/l_{msg.header.seq}.jpg'
        cv2.imwrite(save_path, cv2_img)

        # Update state and timer
        self.image_received = True
        self.last_time = rospy.Time.now()

        return

    def left_info_callback(self, msg):
        if len(self.left_info) == 0:
            self.left_info.append(msg.K)
            self.left_info.append(msg.P)
            self.left_info.append([msg.width, msg.height])

    # Right
    def right_image_callback(self, msg):

        # record gt
        current = self.get_gt_trans_in_map()
        current.append(msg.header.seq)
        self.right_gt.append(current)
        print(current)

        # Convert to cv2 format
        cv2_img = self.imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.file_path is None or not isinstance(self.file_path, str):
            save_path = f'{msg.header.seq}.jpg'
        else:
            save_path = self.file_path + f'/right/r_{msg.header.seq}.jpg'
        cv2.imwrite(save_path, cv2_img)

        # Update state and timer
        self.image_received = True
        self.last_time = rospy.Time.now()

        return

    def right_info_callback(self, msg):
        if len(self.right_info) == 0:
            self.right_info.append(msg.K)
            self.right_info.append(msg.P)
            self.right_info.append([msg.width, msg.height])

    def buoy_callback(self, msg):
        if not self.buoys_received:
            for marker in msg.markers:
                self.buoys.append([marker.pose.position.x,
                                   marker.pose.position.y,
                                   marker.pose.position.z])

            self.buoys_received = True

    # Timer
    def time_check_callback(self, event):
        if not self.image_received:
            return
        delta_t = rospy.Time.now() - self.last_time
        if delta_t.to_sec() >= 5 and not self.data_written:
            print('Data written')
            self.write_data()
            self.data_written = True

        return

    # ===== Transforms =====
    def transform_pose(self, pose, from_frame, to_frame):
        trans = self.wait_for_transform(from_frame=from_frame,
                                        to_frame=to_frame)
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
        return pose_transformed

    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame,
                                                        from_frame,
                                                        rospy.Time(),
                                                        rospy.Duration(1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))

        return trans

    def get_gt_trans_in_map(self):
        """
        Finds pose of the ground truth.
        First,the transform between the map and the ground truth frame.
        Second, the transform is applied to a null_pose located at the origin.
        Modifying the orientation of this pose might be need to prevent later
        processing on the ground truth
        Returns [ x, y, z, q_w, q_x, q_y, q_z]
        """

        trans = self.wait_for_transform(from_frame=self.gt_frame_id,
                                        to_frame=self.frame)

        null_pose = PoseStamped()
        null_pose.pose.orientation.w = 1.0
        pose_in_map = tf2_geometry_msgs.do_transform_pose(null_pose, trans)

        pose_list = [pose_in_map.pose.position.x,
                     pose_in_map.pose.position.y,
                     pose_in_map.pose.position.z,
                     pose_in_map.pose.orientation.w,
                     pose_in_map.pose.orientation.x,
                     pose_in_map.pose.orientation.y,
                     pose_in_map.pose.orientation.z]

        return pose_list

    # ===== Utilities =====
    def write_data(self):
        """
        Save all the relevant data
        """
        # Down
        write_array_to_csv(self.down_info_file_path, self.down_info)
        write_array_to_csv(self.down_gt_file_path, self.down_gt)
        # Left
        write_array_to_csv(self.left_info_file_path, self.left_info)
        write_array_to_csv(self.left_gt_file_path, self.left_gt)
        # Right
        write_array_to_csv(self.right_info_file_path, self.right_info)
        write_array_to_csv(self.right_gt_file_path, self.right_gt)
        # Buoy
        write_array_to_csv(self.buoy_info_file_path, self.buoys)

        return

    @staticmethod
    def imgmsg_to_cv2(img_msg):
        """
        Its assumed that the input image is rgb, opencv expects bgr
        """
        dtype = np.dtype("uint8")  # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                                  dtype=dtype, buffer=img_msg.data)
        # flip converts rgb to bgr
        image_opencv = np.flip(image_opencv, axis=2)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv
