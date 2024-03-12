#!/usr/bin/env python3

import rospy
import tf
import ast
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

def change_marker_depth(marker: Marker, depth: float):
    """
    Changes the marker z value to the specified depth
    :param marker: Marker from visualization_msgs.msg
    :param depth: new z value, float
    :return:
    """
    modified_marker = marker
    modified_marker.pose.position.z = depth
    return modified_marker

def change_marker_color(marker: Marker, color: [float, float, float]):
    """
    Changes the marker color to the specified value
    :param marker: Marker from visualization_msgs.msg
    :param color: [r, g, b] given in float values
    :return:
    """

    modified_marker = marker
    modified_marker.color.r = color[0]
    modified_marker.color.g = color[1]
    modified_marker.color.b = color[2]
    return modified_marker

class publish_pipeline_markers(object):

    def __init__(self,
                 robot_name,  # Robot name
                 map_frame,  # default frame, used to
                 simulated_data,  # adjust if the data is real or simulated
                 pipeline_end_coords,
                 pipeline_depth,
                 pipeline_lines,
                 pipeline_colors):

        self.robot_name = robot_name
        self.map_frame = map_frame
        if simulated_data:
            self.data_type = 'sim'
        else:
            self.data_type = 'real'

        self.cnt = 0
        self.buoy_positions_utm = []
        self.buoy_markers = None
        self.rope_inner_markers = None
        self.rope_outer_markers = None
        self.rope_lines = None  # Is this needed for anything??
        self.marker_rate = 1.0
        self.outer_marker_scale = 5
        self.inner_marker_scale = 2

        self.buoy_marker_list = None

        self.end_coords = pipeline_end_coords
        # self.end_coords = [[-260, -829],
        #                    [-263, -930],
        #                    [-402, -1081],
        #                    [-403, -1178]]

        self.depth = pipeline_depth
        # self.depth = -85

        self.ropes = pipeline_lines
        # self.ropes = [[0, 1],  # was called self.ropes
        #               [1, 2],
        #               [2, 3]]

        if len(pipeline_lines) == len(pipeline_colors):
            self.pipeline_colors = pipeline_colors
        else:
            self.pipeline_colors = [[1.0, 1.0, 0.0] for i in range(len(pipeline_lines))]

        # Visualization options
        self.n_buoys_per_rope = 8
        self.rope_markers_on_z_plane = True  # Modifies rope markers to z=0, useful for 2d estimation

        # marker publishers
        # used to define buoys
        # These buoys are used by the graph to initialize the buoy and rope priors
        # Important to be accurate
        self.marker_pub = rospy.Publisher(f'/{self.robot_name}/{self.data_type}/marked_positions',
                                          MarkerArray, queue_size=10)

        # Used to define the extents of ropes
        self.rope_outer_marker_pub = rospy.Publisher(f'/{self.robot_name}/{self.data_type}/rope_outer_marker',
                                                     MarkerArray, queue_size=10)
        # used to visualize the ropes
        self.rope_marker_pub = rospy.Publisher(f'/{self.robot_name}/{self.data_type}/marked_rope',
                                               MarkerArray, queue_size=10)

        # Visualizes the rope as a line -> I think it might be kind of ugly
        self.rope_lines_pub = rospy.Publisher(f'/{self.robot_name}/{self.data_type}/marked_rope_lines',
                                              MarkerArray, queue_size=10)

    def construct_buoy_markers(self):
        if len(self.end_coords) == 0:
            return

        # ===== Buoys ====

        self.buoy_marker_list = []  # Used later for the forming the lines
        self.buoy_markers = MarkerArray()
        for i, end_coord in enumerate(self.end_coords):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.lifetime = rospy.Duration(0)
            marker.pose.position.x = end_coord[0]
            marker.pose.position.y = end_coord[1]
            marker.pose.position.z = self.depth
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = self.outer_marker_scale
            marker.scale.y = self.outer_marker_scale
            marker.scale.z = self.outer_marker_scale
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.buoy_markers.markers.append(marker)

            self.buoy_marker_list.append(marker)  # Record the markers for later use as the rope end

        # ===== Ropes =====
        if self.ropes is not None and len(self.ropes) > 0:
            self.rope_inner_markers = MarkerArray()  # intermediate buoys on each rope
            self.rope_outer_markers = MarkerArray()  # end buoys of each rope

            self.rope_lines = MarkerArray()

            for rope_ind, rope in enumerate(self.ropes):
                # Create points of the line
                point_start = Point(self.end_coords[rope[0]][0], self.end_coords[rope[0]][1], self.depth)
                point_end = Point(self.end_coords[rope[1]][0], self.end_coords[rope[1]][1], self.depth)

                # Rope as a line
                # === Construct rope line msg ===
                rope_line_marker = Marker()
                rope_line_marker.header.frame_id = self.map_frame
                rope_line_marker.type = Marker.LINE_STRIP
                rope_line_marker.action = Marker.ADD
                rope_line_marker.id = int(self.n_buoys_per_rope + 1) * rope_ind
                rope_line_marker.scale.x = 1.0  # Line width
                rope_line_marker.color.r = 1.0  # Line color (red)
                rope_line_marker.color.a = 1.0  # Line transparency (opaque)
                rope_line_marker.points = [point_start, point_end]
                rope_line_marker.pose.orientation.w = 1

                self.rope_lines.markers.append(rope_line_marker)

                # Rope as a line of buoys

                # markers representing the ends of the ropes
                start_marker = change_marker_color(self.buoy_marker_list[rope[0]], self.pipeline_colors[rope_ind])
                end_marker = change_marker_color(self.buoy_marker_list[rope[1]], self.pipeline_colors[rope_ind])

                if self.rope_markers_on_z_plane:
                    start_marker = change_marker_depth(start_marker, 0.0)
                    end_marker = change_marker_depth(end_marker, 0.0)

                self.rope_outer_markers.markers.append(start_marker)
                self.rope_outer_markers.markers.append(end_marker)

                # Intermediate rope markers
                rope_buoys = self.calculate_rope_buoys(point_start, point_end)

                for buoy_ind, rope_buoy in enumerate(rope_buoys):
                    marker_id = int(self.n_buoys_per_rope + 1) * rope_ind + buoy_ind + 1
                    marker = Marker()
                    marker.header.frame_id = self.map_frame
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker.lifetime = rospy.Duration(0)

                    marker.pose.position.x = rope_buoy.x
                    marker.pose.position.y = rope_buoy.y
                    if self.rope_markers_on_z_plane:
                        marker.pose.position.z = 0.0
                    else:
                        marker.pose.position.z = self.depth

                    marker.pose.orientation.x = 0
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                    marker.pose.orientation.w = 1

                    marker.scale.x = self.inner_marker_scale
                    marker.scale.y = self.inner_marker_scale
                    marker.scale.z = self.inner_marker_scale

                    marker.color.r = self.pipeline_colors[rope_ind][0]
                    marker.color.g = self.pipeline_colors[rope_ind][1]
                    marker.color.b = self.pipeline_colors[rope_ind][2]
                    marker.color.a = 1.0

                    # # Create the line segment points
                    # line_points = [point_start, point_end]
                    #
                    # # Set the line points
                    # marker.points = line_points

                    # Append the marker to the MarkerArray
                    self.rope_inner_markers.markers.append(marker)
                    print(f'|MAP_MARKER| Rope {rope_ind} - {buoy_ind} added')

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
    rospy.init_node('pipeline_markers_nodes', anonymous=False)
    rate = rospy.Rate(1)

    robot_name = rospy.get_param('robot_name', 'sam')
    frame_name = rospy.get_param('frame', 'map')
    simulated_data = rospy.get_param('simulated_data', False)

    # Map Defaults
    default_end_coords = [[-260, -829],
                          [-263, -930],
                          [-402, -1081],
                          [-403, -1178]]

    default_depth = -85

    default_lines = [[0, 1],  # was called self.ropes
                     [1, 2],
                     [2, 3]]

    # Load Map from params if possible
    pipeline_end_coords = ast.literal_eval(rospy.get_param("pipeline_end_coords", "[]"))
    if len(pipeline_end_coords) == 0:
        print("Pipeline marker node: Using default coordinates")
        pipeline_end_coords = default_end_coords

    pipeline_depth = rospy.get_param("pipeline_depth", default=1)
    if pipeline_depth > 0:
        print("Pipeline marker node: Using default depth")
        pipeline_depth = default_depth

    pipeline_lines = ast.literal_eval(rospy.get_param("pipeline_lines", "[]"))
    if len(pipeline_lines) == 0:
        print("Pipeline marker node: Using default lines")
        pipeline_lines = default_lines

    pipeline_colors = ast.literal_eval(rospy.get_param("pipeline_colors", "[]"))

    pipeline_marker_server = publish_pipeline_markers(robot_name=robot_name, map_frame=frame_name,
                                                      simulated_data=simulated_data,
                                                      pipeline_end_coords=pipeline_end_coords,
                                                      pipeline_depth=pipeline_depth,
                                                      pipeline_lines=pipeline_lines,
                                                      pipeline_colors=pipeline_colors)

    if None in [pipeline_marker_server.buoy_markers, pipeline_marker_server.rope_inner_markers]:
        pipeline_marker_server.construct_buoy_markers()

    pipeline_marker_server.publish_markers()

    rospy.spin()
