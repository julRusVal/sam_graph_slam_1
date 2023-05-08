#!/usr/bin/env python3
import os.path
import sys
from functools import partial

import rospy
from std_msgs.msg import Time
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from smarc_msgs.msg import Sidescan

import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf2_geometry_msgs

# cv_bridge and cv2 to convert and save images
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

from sam_slam_utils.sam_slam_helpers import show_simple_graph_2d
from sam_slam_utils.sam_slam_helpers import write_array_to_csv, overwrite_directory
from sam_slam_utils.sam_slam_proc_classes import analyze_slam


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

    def __init__(self, robot_name, frame_name='sam', simulated_data=False,
                 path_name=None,
                 online_graph=None):

        # ===== Real or simulated =====
        self.simulated_data = simulated_data
        self.simulated_detections = rospy.get_param("simulated_detections", True)
        self.correct_dr = True

        # ===== Topic names =====
        self.robot_name = robot_name

        # === Dead reckoning and ground truth ===
        self.gt_topic = f'/{self.robot_name}/sim/odom'
        self.dr_topic = f'/{self.robot_name}/dr/odom'
        self.det_topic = f'/{self.robot_name}/payload/sidescan/detection_hypothesis'
        self.buoy_topic = f'/{self.robot_name}/sim/marked_positions'

        self.roll_topic = f'/{self.robot_name}/dr/roll'
        self.pitch_topic = f'/{self.robot_name}/dr/pitch'
        self.depth_topic = f'/{self.robot_name}/dr/depth'

        # === Sonar ===
        self.sss_topic = f'/{self.robot_name}/payload/sidescan'

        # === Cameras ===
        if self.simulated_data:
            # Camera: down
            self.cam_down_image_topic = f'/{self.robot_name}/perception/csi_cam_0/camera/image_color'
            self.cam_down_info_topic = f'/{self.robot_name}/perception/csi_cam_0/camera/camera_info'
            # Camera: left
            self.cam_left_image_topic = f'/{self.robot_name}/perception/csi_cam_1/camera/image_color'
            self.cam_left_info_topic = f'/{self.robot_name}/perception/csi_cam_1/camera/camera_info'
            # Camera: right
            self.cam_right_image_topic = f'/{self.robot_name}/perception/csi_cam_2/camera/image_color'
            self.cam_right_info_topic = f'/{self.robot_name}/perception/csi_cam_2/camera/camera_info'

        else:
            # Camera: down
            self.cam_down_image_topic = f'/{self.robot_name}/payload/cam_down/image_raw'
            self.cam_down_info_topic = f'/{self.robot_name}/payload/cam_down/camera_info'
            # Camera: left
            self.cam_left_image_topic = f'/{self.robot_name}/payload/cam_port/image_raw'
            self.cam_left_info_topic = f'/{self.robot_name}/payload/cam_port/camera_info'
            # Camera: right
            self.cam_right_image_topic = f'/{self.robot_name}/payload/cam_starboard/image_raw'
            self.cam_right_info_topic = f'/{self.robot_name}/payload/cam_starboard/camera_info'

        # Frame names: For the most part everything is transformed to the map frame
        self.frame = frame_name
        self.gt_frame_id = 'gt/' + self.robot_name + '/base_link'

        # ===== File paths for logging =====
        self.file_path = path_name
        if self.file_path is None or not os.path.isdir(self.file_path):
            print("Invalid file path provided")

        if self.file_path[-1] != '/':
            self.file_path = path_name + '/'

        # Create folders for sensor data
        data_folders = ['left', 'right', 'down', 'sss']
        for data_folder in data_folders:
            overwrite_directory(self.file_path + data_folder)

        self.gt_poses_graph_file_path = self.file_path + 'gt_poses_graph.csv'
        self.dr_poses_graph_file_path = self.file_path + 'dr_poses_graph.csv'
        self.buoys_file_path = self.file_path + 'buoys.csv'

        # === Sonar ===
        # Currently detections are provided by the published buoy location
        self.detections_graph_file_path = self.file_path + 'detections_graph.csv'
        self.sss_graph_file_path = self.file_path + 'detections_graph.csv'

        # === Camera ===
        # Down
        self.down_info_file_path = self.file_path + 'down_info.csv'
        self.down_gt_file_path = self.file_path + 'down_gt.csv'
        # Left
        self.left_info_file_path = self.file_path + 'left_info.csv'
        self.left_times_file_path = self.file_path + 'left_times.csv'
        self.left_gt_file_path = self.file_path + 'left_gt.csv'
        # Right
        self.right_info_file_path = self.file_path + 'right_info.csv'
        self.right_gt_file_path = self.file_path + 'right_gt.csv'

        # tf stuff
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Buoy positions
        self.buoys = []

        # Online SLAM w/ iSAM2
        self.online_graph = online_graph

        # ===== Logging =====
        # Raw logging, occurs at the rate the data is received
        self.gt_poses = []
        self.dr_poses = []
        self.detections = []

        self.rolls = []
        self.pitches = []
        self.depths = []

        # === Sonar logging ===
        self.sss_buffer_len = 10
        self.sss_data_len = 1000  # This is determined by the message
        self.sss_buffer = np.zeros((self.sss_buffer_len, 2 * self.sss_data_len), dtype=np.ubyte)

        # === Camera stuff ===
        # down
        self.down_gt = []
        self.down_info = []
        self.down_times = []
        # left
        self.left_gt = []
        self.left_info = []
        self.left_times = []
        # right
        self.right_gt = []
        self.right_info = []
        self.right_times = []

        # Graph logging
        # TODO check comment for accuracy
        """
        dr_callback will update at a set rate will also record ground truth pose
        det_callback will update all three
        # Current format: [index w.r.t. dr_poses_graph[], x, y, z]
        """
        self.gt_poses_graph = []
        self.dr_poses_graph = []
        self.detections_graph = []

        # States and timings
        if simulated_data:
            self.gt_updated = False  # for simulated data updates are skipped until gt is received
        else:
            self.gt_updated = True

        self.dr_updated = False
        self.roll_updated = False
        self.pitch_updated = False
        self.depth_updated = False
        self.buoy_updated = False
        self.data_written = False
        self.image_received = False

        # self.gt_last_time = rospy.Time.now()
        # self.gt_timeout = 10.0  # Time out used to save data at end of simulation

        self.dr_last_time = rospy.Time.now()
        # Time for limiting the rate that odometry factors are added to graph
        self.dr_update_time = rospy.get_param("dr_update_time", 2.0)
        self.dr_timeout = 10.0  # Time out used to save data at end of simulation

        self.detect_last_time = rospy.Time.now()
        # Time for limiting the rate that detection factors are added to graph
        self.detect_update_time = rospy.get_param("detect_update_time", 0.5)

        self.camera_last_time = rospy.Time.now()
        # Time for limiting the rate that pose with camera data are added to graph
        self.camera_update_time = rospy.get_param("camera_update_time", 0.5)
        self.camera_last_seq = -1

        # Time for limiting the rate that pose with camera data are added to graph
        self.sss_update_time = rospy.get_param("sss_update_time", 1.0)
        self.sss_last_time = rospy.Time.now() - rospy.Time.from_sec(self.sss_update_time)

        # ===== Subscribers =====
        # Ground truth
        self.gt_subscriber = rospy.Subscriber(self.gt_topic,
                                              Odometry,
                                              self.gt_callback)

        # Dead reckoning
        self.dr_subscriber = rospy.Subscriber(self.dr_topic,
                                              Odometry,
                                              self.dr_callback,
                                              self.correct_dr)  # bool to set if the dr is corrected

        # Additional odometry topics: roll, pitch, depth
        self.roll_subscriber = rospy.Subscriber(self.roll_topic,
                                                Float64,
                                                self.roll_callback)

        self.pitch_subscriber = rospy.Subscriber(self.pitch_topic,
                                                 Float64,
                                                 self.pitch_callback)

        self.depth_subscriber = rospy.Subscriber(self.depth_topic,
                                                 Float64,
                                                 self.depth_callback)

        # Buoys
        self.buoy_subscriber = rospy.Subscriber(self.buoy_topic,
                                                MarkerArray,
                                                self.buoy_callback)

        # Detections
        self.det_subscriber = rospy.Subscriber(self.det_topic,
                                               Detection2DArray,
                                               self.det_callback)

        # Sonar
        self.sss_subscriber = rospy.Subscriber(self.sss_topic,
                                               Sidescan,
                                               self.sss_callback)

        # Cameras
        # Down camera
        self.cam_down_image_subscriber = rospy.Subscriber(self.cam_down_image_topic,
                                                          Image,
                                                          self.image_callback,
                                                          'down')

        self.cam_down_info_subscriber = rospy.Subscriber(self.cam_down_info_topic,
                                                         CameraInfo,
                                                         self.info_callback,
                                                         'down')

        # Left camera
        self.cam_left_image_subscriber = rospy.Subscriber(self.cam_left_image_topic,
                                                          Image,
                                                          self.image_callback,
                                                          'left')

        self.cam_left_info_subscriber = rospy.Subscriber(self.cam_left_info_topic,
                                                         CameraInfo,
                                                         self.info_callback,
                                                         'left')

        # Right camera
        self.cam_right_image_subscriber = rospy.Subscriber(self.cam_right_image_topic,
                                                           Image,
                                                           self.image_callback,
                                                           'right')

        self.cam_right_info_subscriber = rospy.Subscriber(self.cam_right_info_topic,
                                                          CameraInfo,
                                                          self.info_callback,
                                                          'right')

        self.time_check = rospy.Timer(rospy.Duration(2),
                                      self.time_check_callback)

    # Subscriber callbacks
    def gt_callback(self, msg):
        """
        Call back for the ground truth subscription, msg is of type nav_msgs/Odometry.
        The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z, time].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.frame)

        gt_position = transformed_pose.pose.position
        gt_quaternion = transformed_pose.pose.orientation
        gt_time = transformed_pose.header.stamp.to_sec()

        self.gt_poses.append([gt_position.x, gt_position.y, gt_position.z,
                              gt_quaternion.w, gt_quaternion.x, gt_quaternion.y, gt_quaternion.z,
                              gt_time])

        self.gt_updated = True
        self.gt_last_time = rospy.Time.now()

    def dr_callback(self, msg, correct_dr):
        """
        Call back for the dead reckoning subscription, msg is of type nav_msgs/Odometry.
        WAS: The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
        NOW: The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z, roll, pitch, depth].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        # wait for gt (if this real)
        if False in [self.gt_updated, self.roll_updated, self.pitch_updated, self.depth_updated]:
            return

        # transform odom to the map frame
        transformed_dr_pose = self.transform_pose(msg.pose,
                                                  from_frame=msg.header.frame_id,
                                                  to_frame=self.frame)

        dr_position = transformed_dr_pose.pose.position
        dr_quaternion = transformed_dr_pose.pose.orientation

        curr_roll = self.rolls[-1]
        curr_pitch = self.pitches[-1]
        curr_depth = self.depths[-1]

        # Record dr poses in format compatible with GTSAM
        if correct_dr:
            # Correction of position
            corrected_x = -dr_position.x
            # Correction of orientation
            # Incorrect method
            # uncorrected_q = Quaternion(dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w)
            # uncorrected_rpy = euler_from_quaternion(dr_quaternion)
            # uncorrected_rpy = euler_from_quaternion([dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w])
            # corrected_y = np.pi - uncorrected_rpy[2]
            # corrected_q = quaternion_from_euler(uncorrected_rpy[0], uncorrected_rpy[1], corrected_y)
            #
            # Correct? method
            r_q = [0, -1, 0, 0]  # The correct orientation correction factor

            dr_q = [dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w]
            corrected_q = tf.transformations.quaternion_multiply(r_q, dr_q)

            self.dr_poses.append([corrected_x, dr_position.y, dr_position.z,
                                  corrected_q[3], corrected_q[0], corrected_q[1], corrected_q[2],
                                  curr_roll, curr_pitch, curr_depth])

        else:
            self.dr_poses.append([dr_position.x, dr_position.y, dr_position.z,
                                  dr_quaternion.w, dr_quaternion.x, dr_quaternion.y, dr_quaternion.z,
                                  curr_roll, curr_pitch, curr_depth])

        # Conditions for updating dr: (1) first time or (2) stale data or (3) online graph is still uninitialized
        time_now = rospy.Time.now()
        first_time_cond = not self.dr_updated and self.gt_updated
        stale_data_cond = self.dr_updated and (time_now - self.dr_last_time).to_sec() > self.dr_update_time
        if self.online_graph is not None:
            online_waiting_cond = self.gt_updated and self.online_graph.initial_pose_set is False
        else:
            online_waiting_cond = False

        if first_time_cond or stale_data_cond or online_waiting_cond:
            # Add to the dr and gt lists
            dr_pose = self.dr_poses[-1]
            self.dr_poses_graph.append(dr_pose)
            if self.simulated_data:
                gt_pose = self.gt_poses[-1]
                self.gt_poses_graph.append(gt_pose)
            else:
                gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

            # Update time and state
            self.dr_last_time = time_now
            self.dr_updated = True

            # ===== Online first update =====
            if self.online_graph is None:
                return
            if self.online_graph.initial_pose_set is False and not self.online_graph.busy:
                print("DR - First update - x0")
                self.online_graph.add_first_pose(dr_pose, gt_pose)

            elif not self.online_graph.busy:
                print(f"DR - Odometry update - x{self.online_graph.current_x_ind + 1}")
                self.online_graph.online_update(dr_pose, gt_pose)

            else:
                print('Busy condition found')

    def roll_callback(self, msg):
        """
        """
        self.rolls.append(msg.data)
        self.roll_updated = True

    def pitch_callback(self, msg):
        """
        """
        self.pitches.append(msg.data)
        self.pitch_updated = True

    def depth_callback(self, msg):
        """
        """
        self.depths.append(msg.data)
        self.depth_updated = True

    def det_callback(self, msg):
        # Check that topics have received messages
        if False in [self.gt_updated, self.roll_updated, self.pitch_updated, self.depth_updated]:
            return

        # check elapsed time
        detect_time_now = rospy.Time.now()
        if (detect_time_now - self.detect_last_time).to_sec() < self.detect_update_time:
            return

        # Reset timer
        self.detect_last_time = rospy.Time.now()

        # Process detection
        for det_ind, detection in enumerate(msg.detections):
            for res_ind, result in enumerate(detection.results):
                # Pose in base_link
                det_pose_base = result.pose

                # Convert to map
                det_pos_map = self.transform_pose(det_pose_base,
                                                  from_frame=msg.header.frame_id,
                                                  to_frame=self.frame)

                # ===== Log data for the graph =====
                # First update dr and gr with the most current
                dr_pose = self.dr_poses[-1]
                self.dr_poses_graph.append(dr_pose)
                if self.simulated_data:
                    gt_pose = self.gt_poses[-1]
                    self.gt_poses_graph.append(gt_pose)
                else:
                    gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

                # detection position:
                # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of dr_pose_graph]
                index = len(self.dr_poses_graph) - 1
                self.detections_graph.append([det_pos_map.pose.position.x,
                                              det_pos_map.pose.position.y,
                                              det_pos_map.pose.position.z,
                                              det_pose_base.pose.position.x,
                                              det_pose_base.pose.position.y,
                                              det_pose_base.pose.position.z,
                                              index])

                # ===== Output =====
                print(f'Detection callback: {index}')

                # ===== Online detection update =====
                if self.online_graph is None:
                    return
                if not self.online_graph.busy:
                    print(f"Detection update - x{self.online_graph.current_x_ind + 1}")
                    self.online_graph.online_update(dr_pose, gt_pose,
                                                    np.array((det_pose_base.pose.position.x,
                                                              det_pose_base.pose.position.y),
                                                             dtype=np.float64))
                else:
                    print('Busy condition found')

    def sss_callback(self, msg):
        """
        The sss callback is responsible for filling the sss_buffer.
        """
        # Record start time
        sss_time_now = rospy.Time.now()

        # Fill buffer regardless of other conditions
        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        self.sss_buffer[1:, :] = self.sss_buffer[:-1, :]
        self.sss_buffer[0, :] = meas

        # Copy buffer
        sss_current = np.copy(self.sss_buffer)

        # Check that topics have received messages
        if False in [self.gt_updated, self.roll_updated, self.pitch_updated, self.depth_updated]:
            return

        # check elapsed time
        if (sss_time_now - self.sss_last_time).to_sec() < self.detect_update_time:
            return

        sss_id = msg.header.seq
        print(f"sss frame: {sss_id}")

        dr_pose = self.dr_poses[-1]
        self.dr_poses_graph.append(dr_pose)
        if self.simulated_data:
            gt_pose = self.gt_poses[-1]
            self.gt_poses_graph.append(gt_pose)
        else:
            gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        # TODO refactor as online update will check for initial pose set
        if self.online_graph is not None \
                and not self.online_graph.busy:

            if self.online_graph.initial_pose_set is False:
                print("SSS - First update w/ sss data")
            else:
                print(f"SSS - Odometry and sss update - {self.online_graph.current_x_ind + 1}")

            # self.online_graph.add_first_pose(dr_pose, gt_pose,
            #                                  initial_estimate=None,
            #                                  id_string=f'sss_{sss_id}')

            self.online_graph.online_update(dr_pose, gt_pose,
                                            relative_detection=None,
                                            id_string=f'sss_{sss_id}')

            # Reset timer after successful completion
            self.sss_last_time = rospy.Time.now()

            # Write to 'disk'
            if self.file_path is None or not os.path.isdir(self.file_path):
                print("Provide valid file path sss output")
            else:
                save_path = self.file_path + f'/sss/{sss_id}.jpg'
                cv2.imwrite(save_path, sss_current)

    def info_callback(self, msg, camera_id):
        if camera_id == 'down':
            if len(self.down_info) == 0:
                self.down_info.append(msg.K)
                self.down_info.append(msg.P)
                self.down_info.append([msg.width, msg.height])
        elif camera_id == 'left':
            if len(self.left_info) == 0:
                self.left_info.append(msg.K)
                self.left_info.append(msg.P)
                self.left_info.append([msg.width, msg.height])
        elif camera_id == 'right':
            if len(self.right_info) == 0:
                self.right_info.append(msg.K)
                self.right_info.append(msg.P)
                self.right_info.append([msg.width, msg.height])
        else:
            print('Unknown camera_id passed to info callback')

    def image_callback(self, msg, camera_id):
        """
        Based
        """
        # Check that topics have received messages
        if False in [self.gt_updated, self.roll_updated, self.pitch_updated, self.depth_updated]:
            return

        # Identifies frames
        # We want to save down, left , and right images of the same frame
        current_id = msg.header.seq

        # check elapsed time
        camera_time_now = rospy.Time.now()
        camera_stale = (camera_time_now - self.camera_last_time).to_sec() > self.camera_update_time

        if camera_stale or not self.image_received:
            self.camera_last_seq = current_id
            self.camera_last_time = rospy.Time.now()
            new_frame = True  # Used to only add one node to graph for each camera frame: down, left, and right
            self.image_received = True
            print(f'New camera frame: {camera_id} - {current_id}')
        elif current_id != self.camera_last_seq:
            return
        else:
            new_frame = False  # Do not add a node to the graph
            print(f'Current camera frame: {camera_id} - {current_id}')

        now_stamp = rospy.Time.now()
        msg_stamp = msg.header.stamp

        dr_pose = self.dr_poses[-1]
        self.dr_poses_graph.append(dr_pose)
        if self.simulated_data:
            gt_pose = self.gt_poses[-1]
            self.gt_poses_graph.append(gt_pose)
        else:
            gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        if new_frame:
            print(f"Adding img-{current_id} to graph")
            if self.online_graph is not None and not self.online_graph.busy:
                if self.online_graph.initial_pose_set:
                    print(f"CAM - Odometry and camera update - {self.online_graph.current_x_ind + 1}")
                else:
                    print("CAM - First update w/ camera data")
                self.online_graph.online_update(dr_pose, gt_pose,
                                                relative_detection=None,
                                                id_string=f'cam_{current_id}')

        # === Debugging ===
        """
        This is only need to verify that the data is being recorded properly
        gt_pose is of the format [x, y, z, q_w, q_x, q_y, q_z, time]
        """
        pose_current = gt_pose[0:-1]
        pose_time = gt_pose[-1]
        pose_current.append(current_id)

        # Record debugging data
        if camera_id == 'down':
            # Record the ground truth and times
            self.down_gt.append(pose_current)
            self.down_times.append([now_stamp.to_sec(),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'left':
            # Record the ground truth and times
            self.left_gt.append(pose_current)
            self.left_times.append([now_stamp.to_sec(),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'right':
            # Record the ground truth and times
            self.right_gt.append(pose_current)
            self.right_times.append([now_stamp.to_sec(),
                                     msg_stamp.to_sec(),
                                     pose_time])
        else:
            print('Unknown camera_id passed to image callback')
            return

        # Display call back info
        print(f'image callback - {camera_id}: {current_id}')
        # print(pose_current)  # debugging

        # Convert to cv2 format
        cv2_img = imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.file_path is None or not os.path.isdir(self.file_path):
            print("Provide valid file path image output")
        else:
            save_path = self.file_path + f'/{camera_id}/{current_id}.jpg'
            cv2.imwrite(save_path, cv2_img)

        return

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
        delta_t = rospy.Time.now() - self.dr_last_time  # was self.gt_last_time
        if delta_t.to_sec() >= self.dr_timeout and not self.data_written:  # was self.gt_timeout
            print('Data written')
            self.write_data()
            self.data_written = True

            if self.online_graph is not None:
                # TODO Save final results
                print("Print online graph")

                analysis = analyze_slam(self.online_graph)
                analysis.save_for_sensor_processing(self.file_path)
                analysis.save_2d_poses(self.file_path)
                analysis.show_graph_2d(label="Online Graph",
                                       show_final=True)

                # show_simple_graph_2d(graph=self.online_graph.graph,
                #                      x_keys=self.online_graph.x,
                #                      b_keys=self.online_graph.b,
                #                      values=self.online_graph.current_estimate,
                #                      label="Online Graph")

        return

    # ===== Transforms and poses =====
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

        # === Camera ===
        # Down
        write_array_to_csv(self.down_info_file_path, self.down_info)
        write_array_to_csv(self.down_gt_file_path, self.down_gt)
        # Left
        write_array_to_csv(self.left_info_file_path, self.left_info)
        write_array_to_csv(self.left_times_file_path, self.left_times)
        write_array_to_csv(self.left_gt_file_path, self.left_gt)
        # Right
        write_array_to_csv(self.right_info_file_path, self.right_info)
        write_array_to_csv(self.right_gt_file_path, self.right_gt)

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
        self.left_times_file_path = file_path_prefix + 'left_times.csv'
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
        self.gt_topic = '/sam/sim/odom'

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
        self.gt_poses_from_topic = []
        self.buoys = []

        # ===== Image processing =====
        # self.bridge = CvBridge()  # Not had problems

        # ===== States =====
        self.last_time = rospy.Time.now()
        self.tf_ready = False
        self.gt_updated = False
        self.buoys_received = False
        self.image_received = False
        self.data_written = False

        # ===== Subscriptions =====
        # Down camera
        self.cam_down_image_subscriber = rospy.Subscriber(self.cam_down_image_topic,
                                                          Image,
                                                          self.image_callback,
                                                          'down')

        self.cam_down_info_subscriber = rospy.Subscriber(self.cam_down_info_topic,
                                                         CameraInfo,
                                                         self.info_callback,
                                                         'down')

        # Left camera
        self.cam_left_image_subscriber = rospy.Subscriber(self.cam_left_image_topic,
                                                          Image,
                                                          self.image_callback,
                                                          'left')

        self.cam_left_info_subscriber = rospy.Subscriber(self.cam_left_info_topic,
                                                         CameraInfo,
                                                         self.info_callback,
                                                         'left')

        # Right camera
        self.cam_right_image_subscriber = rospy.Subscriber(self.cam_right_image_topic,
                                                           Image,
                                                           self.image_callback,
                                                           'right')

        self.cam_right_info_subscriber = rospy.Subscriber(self.cam_right_info_topic,
                                                          CameraInfo,
                                                          self.info_callback,
                                                          'right')

        # Ground truth
        self.gt_subscriber = rospy.Subscriber(self.gt_topic,
                                              Odometry,
                                              self.gt_callback)

        # Buoys
        self.buoy_subscriber = rospy.Subscriber(self.buoy_topic,
                                                MarkerArray,
                                                self.buoy_callback)

        # ===== Timers =====
        self.time_check = rospy.Timer(rospy.Duration(1),
                                      self.time_check_callback)

    # ===== Callbacks =====
    def old_down_image_callback(self, msg, camera_id):

        print(camera_id)
        # Record gt
        current, _ = self.get_gt_trans_in_map()
        current.append(msg.header.seq)
        self.down_gt.append(current)

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
            save_path = f'd_{msg.header.seq}.jpg'
        else:
            save_path = self.file_path + f'/down/d_{msg.header.seq}.jpg'
        cv2.imwrite(save_path, cv2_img)

        # Update state and timer
        self.image_received = True
        self.last_time = rospy.Time.now()

        return

    def info_callback(self, msg, camera_id):
        if camera_id == 'down':
            if len(self.down_info) == 0:
                self.down_info.append(msg.K)
                self.down_info.append(msg.P)
                self.down_info.append([msg.width, msg.height])
        elif camera_id == 'left':
            if len(self.left_info) == 0:
                self.left_info.append(msg.K)
                self.left_info.append(msg.P)
                self.left_info.append([msg.width, msg.height])
        elif camera_id == 'right':
            if len(self.right_info) == 0:
                self.right_info.append(msg.K)
                self.right_info.append(msg.P)
                self.right_info.append([msg.width, msg.height])
        else:
            print('Unknown camera_id passed to info callback')

    def image_callback(self, msg, camera_id):
        """
        Based on down_image_callback
        """

        now_stamp = rospy.Time.now()
        msg_stamp = msg.header.stamp

        # record gt
        current_id = msg.header.seq

        # ===== Pick one method =====
        # Method 1 - frame transform
        # current_pose, pose_stamp = self.get_gt_trans_in_map(gt_time=msg.header.stamp)
        # pose_time = pose_stamp.to_sec()
        # current.append(current_id)

        # Method 2 - subscription method of gt
        current_pose_and_time = self.gt_poses_from_topic[-1]
        current = current_pose_and_time[0:-1]
        pose_time = current_pose_and_time[-1]
        current.append(current_id)

        # Record data
        if camera_id == 'down':
            # Record the ground truth and times
            self.left_gt.append(current)
            self.left_times.append([now_stamp.to_sec(),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'left':
            # Record the ground truth and times
            self.left_gt.append(current)
            self.left_times.append([now_stamp.to_sec(),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'right':
            # Record the ground truth and times
            self.left_gt.append(current)
            self.left_times.append([now_stamp.to_sec(),
                                    msg_stamp.to_sec(),
                                    pose_time])
        else:
            print('Unknown camera_id passed to image callback')
            return

        # Display call back info
        print(f'image callback - {camera_id}: {current_id}')
        print(current)

        # Convert to cv2 format
        cv2_img = imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.file_path is None or not isinstance(self.file_path, str):
            save_path = f'{camera_id}_{current_id}.jpg'
        else:
            save_path = self.file_path + f'/{camera_id}/{current_id}.jpg'
        cv2.imwrite(save_path, cv2_img)

        # Update state and timer
        self.image_received = True
        self.last_time = rospy.Time.now()

        return

    def gt_callback(self, msg):
        """
        Call back for the ground truth subscription, msg is of type nav_msgs/Odometry.
        The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.frame,
                                               req_transform_time=None)

        gt_position = transformed_pose.pose.position
        gt_quaternion = transformed_pose.pose.orientation
        gt_time = transformed_pose.header.stamp.to_sec()

        self.gt_poses_from_topic.append([gt_position.x, gt_position.y, gt_position.z,
                                         gt_quaternion.w, gt_quaternion.x, gt_quaternion.y, gt_quaternion.z,
                                         gt_time])

        self.gt_updated = True

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
    def transform_pose(self, pose, from_frame, to_frame, req_transform_time=None):
        trans = self.wait_for_transform(from_frame=from_frame,
                                        to_frame=to_frame,
                                        req_transform_time=req_transform_time)

        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)

        return pose_transformed

    def wait_for_transform(self, from_frame, to_frame, req_transform_time=None):
        """Wait for transform from from_frame to to_frame"""
        trans = None

        if isinstance(req_transform_time, Time):
            transform_time = req_transform_time
        else:
            transform_time = rospy.Time()

        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame,
                                                        from_frame,
                                                        transform_time,
                                                        rospy.Duration(1))

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))

        return trans

    def get_gt_trans_in_map(self, gt_time=None):
        """
        Finds pose of the ground truth.
        First,the transform between the map and the ground truth frame.
        Second, the transform is applied to a null_pose located at the origin.
        Modifying the orientation of this pose might be need to prevent later
        processing on the ground truth
        Returns [ x, y, z, q_w, q_x, q_y, q_z]
        """

        trans = self.wait_for_transform(from_frame=self.gt_frame_id,
                                        to_frame=self.frame,
                                        req_transform_time=gt_time)

        null_pose = PoseStamped()
        null_pose.pose.orientation.w = 1.0
        pose_in_map = tf2_geometry_msgs.do_transform_pose(null_pose, trans)

        pose_stamp = pose_in_map.header.stamp

        pose_list = [pose_in_map.pose.position.x,
                     pose_in_map.pose.position.y,
                     pose_in_map.pose.position.z,
                     pose_in_map.pose.orientation.w,
                     pose_in_map.pose.orientation.x,
                     pose_in_map.pose.orientation.y,
                     pose_in_map.pose.orientation.z]

        return pose_list, pose_stamp

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
        write_array_to_csv(self.left_times_file_path, self.left_times)
        write_array_to_csv(self.left_gt_file_path, self.left_gt)
        # Right
        write_array_to_csv(self.right_info_file_path, self.right_info)
        write_array_to_csv(self.right_gt_file_path, self.right_gt)
        # Buoy
        write_array_to_csv(self.buoy_info_file_path, self.buoys)

        return
