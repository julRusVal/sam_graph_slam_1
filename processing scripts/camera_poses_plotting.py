#!/usr/bin/env python3
"""
The Goal of this script is to load saved gt poses (base_link) and convert them to the camera poses in the map frame.
This is part of the work towards projecting images onto a planes to make algae farm maps.
"""
import gtsam
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from sam_slam_utils.sam_slam_helper_funcs import read_csv_to_array, read_csv_to_list
import gtsam.utils.plot as gtsam_plot


# %% Functions
def create_Pose3(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    # rot3_xyz = rot3.xyz()
    return gtsam.Pose3(rot3, input_pose[0:3])


def convert_poses_to_Pose3(poses):
    """
    Poses is is of the form: [[x,y,z,q_w,q_x,q_,y,q_z]]
    """
    pose3s = []
    for pose in poses:
        pose3s.append(create_Pose3(pose))

    return pose3s


def apply_transformPoseFrom(pose3s, transform):
    """
    pose3s: [gtsam.Pose3]
    transform: gtsam.Pose3

    Apply the transform given in local coordinates, result is expressed in the world coords
    """
    transformed_pose3s = []
    for pose3 in pose3s:
        transformed_pose3 = pose3.transformPoseFrom(transform)
        transformed_pose3s.append(transformed_pose3)

    return transformed_pose3s


def projectPixelTo3dRay(u, v, cx, cy, fx, fy):
    """
    From ROS-perception
    https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/image_geometry/cameramodels.py
    Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
    using the camera :math:`P` matrix.
    This is the inverse of :math:`project3dToPixel`.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    norm = math.sqrt(x * x + y * y + 1)
    x /= norm
    y /= norm
    z = 1.0 / norm
    return x, y, z


class rope_section:
    def __init__(self, start_buoy, end_buoy, start_coord, end_coord, bottom_coord):
        self.start_buoy = start_buoy
        self.end_buoy = end_buoy

        self.start_coord = start_coord
        self.end_coord = end_coord
        self.bottom_coord = bottom_coord

        # ===== Define plane =====
        # Basis vectors
        self.v_x = end_coord - start_coord
        self.mag_x = np.sqrt(np.sum(np.multiply(self.v_x, self.v_x)))
        self.v_x = self.v_x / self.mag_x

        self.v_y = bottom_coord - start_coord
        self.mag_y = np.sqrt(np.sum(np.multiply(self.v_y, self.v_y)))
        self.v_y = self.v_y / self.mag_y

        # normal vector
        # Note:due to downward orientation of y the norm points into the plane
        self.normal = np.cross(self.v_x, self.v_y)

        # Point on plane
        self.Q = self.start_coord

        # ===== Storage for images and masks
        self.images = {}
        self.masks = {}

    def find_intersection(self, point, direction):
        """
        Given the plane does the line described by the point and direction intersect.
        The
        """

        # Check for intersection
        # Note the direction is also checked because we only want to consider intersections from the 'font'
        # of the plane
        if np.dot(self.normal, direction) <= 0:
            return False, False, False, False
        else:
            # Calculate the intersection point
            t = -np.dot(self.normal, (point - self.Q)) / np.dot(self.normal, direction)
            intersection_point = point + t * direction

            # Express the intersection point as a linear combination of v1 and v2 using the pseudoinverse
            A = np.vstack([self.v_x, self.v_y]).T
            b = intersection_point - self.Q
            x = np.linalg.pinv(A).dot(b)
            s, t = x
            # intersection_point_linear_combination = f"{s}*v1 + {t}*v2 + {Q}"

            if 0 <= s <= self.mag_x and 0 <= t <= self.mag_y:
                within_bounds = True
            else:
                within_bounds = False

            return True, intersection_point, (s, t), within_bounds


class image_mapping:
    def __init__(self, base_link_pose, camera_info, relative_camera_pose, buoy_info, ropes):
        # ===== Base pose =====
        self.base_pose = base_link_pose
        self.base_pose3s = convert_poses_to_Pose3(self.base_pose)
        self.pose_ids = self.base_pose[:, -1]

        # ===== Camera =====
        self.camera_info = camera_info
        self.relative_camera_pose = relative_camera_pose
        # TODO Currently hard coded to use left
        self.camera_pose3s = apply_transformPoseFrom(self.base_pose3s, self._return_left_relative_pose())

        # form k and P  matrices, plumb bob distortion model
        # K: 3x3
        self.K = np.array(self.camera_info[0], dtype=np.float64)
        self.K = np.reshape(self.K, (3, 3))
        # P: 3x4
        self.P = np.array(self.camera_info[1], dtype=np.float64)
        self.P = np.reshape(self.P, (3, 4))

        # Convert focal length to meters
        # self.P[0, 0] = self.P[0, 0] / 1000
        # self.P[1, 1] = self.P[1, 1] / 1000

        # Extract key parameters
        self.width = int(self.camera_info[2][0])
        self.height = int(self.camera_info[2][1])
        self.fx = self.P[0, 0]
        self.fy = self.P[1, 1]
        self.cx = self.P[0, 2]
        self.cy = self.P[1, 2]
        self.orig_img_corners = np.array([[0, 0],
                                          [self.width, 0],
                                          [0, self.height],
                                          [self.width, self.height]], dtype=np.int32)

        self.fov_rays = None  # rays are defined w.r.t. the camera frame
        self.find_fov_rays()

        # ===== Map info =====
        self.buoys = buoy_info
        self.ropes = ropes
        self.depth = 15  # Used to define the vertical extent of the planes

        self.planes = []
        self.build_planes_from_buoys_ropes()

        self.spatial_2_pixel = 100  # 1 meter = 100 pixels

    @staticmethod
    def _return_left_relative_pose():
        l_t_x = 1.313
        l_t_y = 0.048
        l_t_z = -0.007
        l_r_x = -0.733244
        l_r_y = 0.310005
        l_r_z = -0.235671
        l_r_w = 0.557413

        return create_Pose3([l_t_x, l_t_y, l_t_z, l_r_w, l_r_x, l_r_y, l_r_z])

    def find_fov_rays(self):
        """
        Calculates and sets FOV points
        """
        # Corners
        # (0,0,1), (width,0,1), (0, height, 1), (width, height, 1)
        p_0 = projectPixelTo3dRay(0, 0, self.cx, self.cy, self.fx, self.fy)
        p_1 = projectPixelTo3dRay(self.width, 0, self.cx, self.cy, self.fx, self.fy)
        p_2 = projectPixelTo3dRay(0, self.height, self.cx, self.cy, self.fx, self.fy)
        p_3 = projectPixelTo3dRay(self.width, self.height, self.cx, self.cy, self.fx, self.fy)
        # Center
        p_c = projectPixelTo3dRay(self.cx, self.cy, self.cx, self.cy, self.fx, self.fy)

        self.fov_rays = np.array([p_0, p_1, p_2, p_3, p_c], dtype=np.float64)

    def build_planes_from_buoys_ropes(self):

        for rope in self.ropes:
            rope_for = [rope[0], rope[1]]
            rope_rev = [rope[1], rope[0]]
            directions = [rope_for, rope_rev]
            for direction in directions:
                start_buoy = int(direction[0])
                end_buoy = int(direction[1])

                start_coord = self.buoys[start_buoy, :]
                end_coord = self.buoys[end_buoy, :]
                bottom_coord = self.buoys[start_buoy, :] - np.array([0, 0, self.depth], dtype=np.float64)

                self.planes.append(rope_section(start_buoy=start_buoy,
                                                end_buoy=end_buoy,
                                                start_coord=start_coord,
                                                end_coord=end_coord,
                                                bottom_coord=bottom_coord))

    def plot_fancy(self, other_pose3s=None):
        # Parameters
        fig_num = 0
        base_scale = .1
        other_scale = 1

        fig = plt.figure(fig_num)
        axes = fig.add_subplot(projection='3d')

        axes.set_xlabel("X axis")
        axes.set_ylabel("Y axis")
        axes.set_zlabel("Z axis")

        # Plot buoys
        # for buoy in buoy_info:
        axes.scatter(buoy_info[:, 0], buoy_info[:, 1], buoy_info[:, 2], c='b', linewidths=5)

        # plot base_link gt pose3s
        for i_base, pose3 in enumerate(self.base_pose3s):
            # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)
            if i_base == 25:
                gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)
        # plot camera gt pose3s
        if other_pose3s is not None:
            for i_other, pose3 in enumerate(other_pose3s):
                # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                if i_other == 25:
                    gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                    # plot fov
                    for i_ray, ray in enumerate(self.fov_rays):
                        point_end = pose3.transformFrom(10 * ray)
                        x_comp = [pose3.x(), point_end[0]]
                        y_comp = [pose3.y(), point_end[1]]
                        z_comp = [pose3.z(), point_end[2]]
                        # Plot (0,0) as magenta
                        if i_ray == 0:
                            axes.plot(x_comp, y_comp, z_comp, c='m')
                        # Plot center line as yellow
                        elif i_ray == 4:
                            axes.plot(x_comp, y_comp, z_comp, c='y')
                        # Other fov lines plotted as black
                        else:
                            axes.plot(x_comp, y_comp, z_comp, c='k')

                    # plot intersection
                    for plane in self.planes:
                        # Find the center ray in the camera frame and then find the world coord given pose
                        center_ray = self.fov_rays[-1]
                        center_end_point_pose3 = pose3.transformFrom(center_ray)

                        end = np.array([center_end_point_pose3[0],
                                        center_end_point_pose3[1],
                                        center_end_point_pose3[2]], dtype=np.float64)

                        start = np.array([pose3.x(),
                                          pose3.y(),
                                          pose3.z()])

                        direction = end - start

                        intrcpt_status, intrcpt_w_coords, intrcpt_p_coords, in_bounds = plane.find_intersection(start,
                                                                                                                direction)

                        if in_bounds:
                            axes.scatter(intrcpt_w_coords[0], intrcpt_w_coords[1], intrcpt_w_coords[2], c='r')

        plt.axis('equal')
        plt.title("Testing the transform")
        plt.show()

    def find_corner_coords(self, plane_id, pose_id):
        """

        """
        # Camera pose
        pose3 = self.camera_pose3s[pose_id]

        # intersection points of camera fov with plane
        # given in coords of the plane
        corner_cords = np.zeros((4, 2), dtype=np.float64)

        # Compute an intersection for each ray
        for i_ray in range(4):
            ray = self.fov_rays[i_ray]
            ray_end_point_point3 = pose3.transformFrom(ray)

            end = np.array([ray_end_point_point3[0],
                            ray_end_point_point3[1],
                            ray_end_point_point3[2]], dtype=np.float64)

            start = np.array([pose3.x(),
                              pose3.y(),
                              pose3.z()], dtype=np.float64)

            direction = end - start

            corner_status, corner_w_coords, corner_p_coords, _ = self.planes[plane_id].find_intersection(start,
                                                                                                         direction)

            corner_cords[i_ray, :] = corner_p_coords

        return corner_cords

    def convert_spatial_corners_to_pixel(self, plane_id, corner_coords):
        corner_coords_pixels = corner_coords * self.spatial_2_pixel
        corner_coords_pixels = np.floor_divide(corner_coords_pixels, 1).astype(np.int32)

        # Find min and max and apply offset
        min_x = min(min(corner_coords_pixels[:, 0]), 0)
        min_y = min(min(corner_coords_pixels[:, 1]), 0)

        offset = np.array([min_x, min_y], dtype=np.int32)

        corner_coords_pixels = corner_coords_pixels - offset

        max_x = max(max(corner_coords_pixels[:, 0]), (self.planes[plane_id].mag_x * self.spatial_2_pixel) // 1 - 1)
        max_y = max(max(corner_coords_pixels[:, 1]), (self.planes[plane_id].mag_y * self.spatial_2_pixel) // 1 - 1)

        max_inds = np.array([max_x, max_y], dtype=np.int32)

        return corner_coords_pixels, offset, max_inds

    def find_pixels_of_3d_point(self, pose_id, map_point):
        """

        :param pose_id:
        :param map_point: a point2, np.array (3x1) that hold the world coords of a point
        :return:
        """
        camera_pose3 = self.camera_pose3s[pose_id]

        # express the provided point (given in map coords) in camera coords.
        camera_point3 = camera_pose3.transformTo(map_point)
        # convert to homogeneous coords
        # Convert m -> mm
        camera_point_hmg = np.ones((4,))
        camera_point_hmg[:3] = camera_point3[:] * 1000

        # Apply projection matrix
        uvw_array = np.matmul(self.P, camera_point_hmg)

        u = uvw_array[0]
        v = uvw_array[1]
        w = uvw_array[2]

        if w != 0.0:
            return (u / w), (v / w)
        else:
            return float('nan'), float('nan')


# %% Load and process data
# This is the gt of the base_link indexed for the left images
base_gt = read_csv_to_array('data/left_gt.csv')
left_info = read_csv_to_list('data/left_info.csv')
buoy_info = read_csv_to_array('data/buoy_info.csv')

# Define the connections between buoys
# list of start and stop indices of buoys
# TODO remove hard coded rope structure
ropes = [[0, 4], [4, 2],
         [1, 5], [5, 3]]

img_map = image_mapping(base_link_pose=base_gt,
                        camera_info=left_info,
                        relative_camera_pose=None,
                        buoy_info=buoy_info,
                        ropes=ropes)

# %% Plot
img_map.plot_fancy(img_map.camera_pose3s)
# plot_fancy(base_gt_pose3s, left_gt_pose3s, buoy_info, points)

# %% Testing parameters
do_testing_1 = False
do_testing_2 = True
# %% Testing 1
if do_testing_1:
    print("Testing 1")

    p_org = np.array([-5.0, 4.0, -0])
    p_x = np.array([-5.0, 9.0, -0])
    p_y = np.array([-5.0, 4.0, -15])

    point = np.array([-4.0, 3.0, -6])
    direction = np.array([-1.0, 0, 0])

    a, b, c, d = img_map.planes[0].find_intersection(point, direction)

    corner_coords = img_map.find_corner_coords(0, 25)

    corner_pixels, offset, max_inds = img_map.convert_spatial_corners_to_pixel(0, corner_coords)

    M = cv2.getPerspectiveTransform(img_map.orig_img_corners.astype(np.float32), corner_pixels.astype(np.float32))

    img_id = int(img_map.base_pose[25, -1])

    img = cv2.imread(f"/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/left/l_{img_id}.jpg")

    new_img = cv2.warpPerspective(img, M, (max_inds[0] + 1, max_inds[1] + 1))

    # cv2.imshow('thing', new_img)

    # cv2.imwrite('new_img.jpg', new_img)

    # truncate warped image
    x_original = corner_pixels[0] + offset

    x_start = abs(offset[0]) + 76
    x_size = int((img_map.planes[0].mag_x * img_map.spatial_2_pixel) // 1)
    x_end = x_start + x_size

    y_start = abs(offset[1])
    y_size = int((img_map.planes[0].mag_y * img_map.spatial_2_pixel) // 1)
    y_end = y_start + y_size

    new_img = new_img[y_start:y_end, x_start:x_end]

    cv2.imwrite('new_img.jpg', new_img)

# %% Testing 2
if do_testing_2:
    print("Testing 2")
    pose_id = 25
    proper_img_id = int(img_map.base_pose[pose_id, -1])
    test_next_n = 3

    test_point_b0 = np.array([-5.0, 4.0, 0])
    test_point_b1 = np.array([-5.0, 9.0, 0])
    test_point_c = np.array([-5.0, 6.592, -0.706])
    x_pix_0, y_pix_0 = img_map.find_pixels_of_3d_point(pose_id, test_point_b0)
    x_pix_1, y_pix_1 = img_map.find_pixels_of_3d_point(pose_id, test_point_b1)
    x_pix_c, y_pix_c = img_map.find_pixels_of_3d_point(pose_id, test_point_c)

    # Mark images
    for i in range(int(test_next_n + 1)):
        img_id = proper_img_id + i
        img = cv2.imread(f"/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/left/l_{img_id}.jpg")
        img_marked = cv2.circle(img, (int(x_pix_0//1), int(y_pix_0//1)), 5, (0, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_1 // 1), int(y_pix_1 // 1)), 5, (0, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_c // 1), int(y_pix_c // 1)), 5, (0, 255, 255), -1)

        img_marked[:, int(img_map.cx), :] = (0, 255, 255)
        img_marked[int(img_map.cy), :, :] = (0, 255, 255)

        #
        cv2.imshow(f"Marked Image: {img_id}", img_marked)

    # Waits for a keystroke
    cv2.waitKey(0)

    # Destroys all the windows created
    cv2.destroyAllWindows()
