#!/usr/bin/env python3

# ROS imports
import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
# from tf.transformations import euler_from_quaternion

# General imports
import csv

from sam_slam_utils.sam_slam_classes import sam_slam_listener


"""
import gtsam
"""


# Functions
def write_data_set(file_path, data_array):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data_array:
            writer.writerow(row)


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

    """
    def __init__(self, gt_top_name, dr_top_name, det_top_name, buoy_top_name, frame_name):
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

        # Current data
        self.gt_pose = None
        self.dr_pose = None
        self.det_info = None
        self.buoys = None

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
        self.dr_poses_file_path = 'dr_poses.csv'
        self.gt_poses_file_path = 'gt_poses.csv'
        self.detections_file_path = 'detections.csv'

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

        # Subscribers
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
        self.gt_pose = transformed_pose

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
        self.dr_pose = transformed_dr_pose

        dr_position = transformed_dr_pose.pose.position
        dr_quaternion = transformed_dr_pose.pose.orientation
        self.dr_poses.append([dr_position.x, dr_position.y, dr_position.z,
                              dr_quaternion.w, dr_quaternion.x, dr_quaternion.y, dr_quaternion.z])

        # Conditions for updating dr: (1) first time or (2) stale data
        time_now = rospy.Time.now()
        first_time_cond = not self.dr_updated and self.gt_updated
        stale_data_cond = self.dr_updated and (time_now - self.last_time).to_sec() > self.update_time

        if first_time_cond or stale_data_cond:
            # Add to the dr and gt lists
            self.dr_poses_graph.append(self.dr_poses[-1])
            self.gt_poses_graph.append(self.get_gt_trans_in_map())
            # (OLD) self.gt_poses_graph.append(self.gt_poses[-1])

            # Update time and state
            self.last_time = time_now
            self.dr_updated = True

    def det_callback(self, msg):
        self.det_info = msg
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
                det_quaternion = transformed_pose.pose.orientation
                # Perform raw logging of detections
                # Append [x,y,z,id,score]
                self.detections.append([det_position.x, det_position.y, det_position.z,
                                        det_quaternion.w, det_quaternion.x, det_quaternion.y, det_quaternion.z,
                                        result.id,
                                        result.score])

                # ===== Log data for the graph =====
                # First update dr and gr with the most current
                self.dr_poses_graph.append(self.dr_poses[-1])
                self.gt_poses_graph.append(self.get_gt_trans_in_map())
                # (OLD) self.gt_poses_graph.append(self.gt_poses[-1])

                # detection position:
                # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of ]
                index = len(self.dr_poses_graph) - 1
                self.detections_graph.append([det_position.x,
                                              det_position.y,
                                              det_position.z,
                                              detection_position.pose.position.x,
                                              detection_position.pose.position.y,
                                              detection_position.pose.position.z,
                                              index])

    def buoy_callback(self, msg):
        if not self.buoy_updated:
            self.buoys = []
            for marker in msg.markers:
                self.buoys.append([marker.pose.position.x,
                                   marker.pose.position.y,
                                   marker.pose.position.z])
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
        return

    # Transforms
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
                                                        rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))
        return trans

    def get_gt_trans_in_map(self):
        """
        Returns [ x, y, z, q_w, q_x, q_y, q_z]
        """
        # TODO only need to return one of these

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

    # Random utility methods
    def write_data(self):
        # Save dead reckoning
        write_data_set(self.dr_poses_file_path, self.dr_poses)
        write_data_set(self.dr_poses_graph_file_path, self.dr_poses_graph)

        # Save ground truth
        write_data_set(self.gt_poses_file_path, self.gt_poses)
        write_data_set(self.gt_poses_graph_file_path, self.gt_poses_graph)
        # (OLD) write_data_set(self.gt_poses_graph_file_path, self.gt_poses_graph)

        # Save detections
        write_data_set(self.detections_file_path, self.detections)
        write_data_set(self.detections_graph_file_path, self.detections_graph)

        write_data_set(self.buoys_file_path, self.buoys)
        return


def main():
    rospy.init_node('slam_listener', anonymous=True)
    rospy.Rate(5)

    ground_truth_topic = '/sam/sim/odom'
    dead_reckon_topic = '/sam/dr/odom'
    detection_topic = '/sam/payload/sidescan/detection_hypothesis'
    buoy_topic = '/sam/sim/marked_positions'
    frame = 'map'

    print('initializing listener')
    listener = sam_slam_listener(ground_truth_topic, dead_reckon_topic, detection_topic, buoy_topic, frame)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    main()
