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
    def __init__(self, start_buoy, end_buoy, start_coord, end_coord, depth):
        self.start_buoy = start_buoy
        self.end_buoy = end_buoy

        self.start_coord = start_coord
        self.end_coord = end_coord

        self.start_bottom_coord = start_coord - np.array([0, 0, depth], dtype=np.float64)
        self.end_bottom_coord = end_coord - np.array([0, 0, depth], dtype=np.float64)

        # ===== Define plane =====
        # Basis vectors
        # x is the horizontal component and y is the vertical component
        self.v_x = end_coord - start_coord
        self.mag_x = np.sqrt(np.sum(np.multiply(self.v_x, self.v_x)))
        self.v_x = self.v_x / self.mag_x

        self.v_y = self.start_bottom_coord - self.start_coord
        self.mag_y = np.sqrt(np.sum(np.multiply(self.v_y, self.v_y)))
        self.v_y = self.v_y / self.mag_y

        # normal vector
        # Note:due to downward orientation of y the norm points into the plane
        self.normal = np.cross(self.v_x, self.v_y)

        # Point on plane
        self.Q = self.start_coord

        # ===== Storage for images and masks
        self.images = []
        self.masks = []

    def find_intersection(self, point, direction):
        """
        Given the plane does the line described by the point and direction intersect.
        :param point: np.array([x, y, z])
        :param direction: np.array([delta_x, delta_y, delta_z])
        :return: status: uses the dot product of the plane norm and camera z-axis to exclude applying images to 'back'
        :return: w_coords:
        :return: p_coords:
        :return: in_bounds: true is central line of camera intersects with the bounds of the plane
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
    def __init__(self, gt_base_link_poses, base_link_poses, camera_info, relative_camera_pose, buoy_info, ropes):
        # ===== true base pose =====
        self.gt_base_pose = gt_base_link_poses
        self.gt_base_pose3s = convert_poses_to_Pose3(self.gt_base_pose)

        # ===== Base pose =====
        self.base_pose = base_link_poses
        self.base_pose3s = convert_poses_to_Pose3(self.base_pose)

        self.pose_ids = self.base_pose[:, -1]

        # ===== Camera =====
        self.camera_info = camera_info
        # TODO Currently hard coded to use left
        self.relative_camera_pose = self.return_left_relative_pose(use_rpy=False)

        self.gt_camera_pose3s = apply_transformPoseFrom(self.gt_base_pose3s, self.relative_camera_pose)
        self.camera_pose3s = apply_transformPoseFrom(self.base_pose3s, self.relative_camera_pose)

        # form k and P  matrices, plumb bob distortion model
        # K: 3x3
        self.K = np.array(self.camera_info[0], dtype=np.float64)
        self.K = np.reshape(self.K, (3, 3))
        # P: 3x4
        self.P = np.array(self.camera_info[1], dtype=np.float64)
        self.P = np.reshape(self.P, (3, 4))

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
        self.depth = 7.5  # Used to define the vertical extent of the planes

        self.planes = []
        self.build_planes_from_buoys_ropes()

        self.spatial_2_pixel = 100  # 1 meter = 100 pixels

    @staticmethod
    def return_left_relative_pose(use_rpy=False):
        # TODO return relative camera pose to 'correct' value of 1.313
        l_t_x = 1.313
        l_t_y = 0.048
        l_t_z = -0.007
        # l_t_x = 0.6
        # l_t_y = 0.0
        # l_t_z = 0.0

        # ===== Quaternion values from ROS tf messages =====
        l_r_x = -0.733244
        l_r_y = 0.310005
        l_r_z = -0.235671
        l_r_w = 0.557413

        # ===== RPY values from stonefish =====
        # these values have been converted to quaternions
        # [-0.7332437391795082, 0.31000489232064976, -0.23567133276329172, 0.5574133193644455]
        l_r_x_2 = -0.7332437391795082
        l_r_y_2 = 0.31000489232064976
        l_r_z_2 = -0.23567133276329172
        l_r_w_2 = 0.5574133193644455

        if use_rpy:
            return create_Pose3([l_t_x, l_t_y, l_t_z, l_r_w_2, l_r_x_2, l_r_y_2, l_r_z_2])
        else:
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

                self.planes.append(rope_section(start_buoy=start_buoy,
                                                end_buoy=end_buoy,
                                                start_coord=start_coord,
                                                end_coord=end_coord,
                                                depth=self.depth))

    def plot_fancy(self, other_pose3s=None):
        # Parameters
        fig_num = 0
        base_scale = .5
        other_scale = 1
        plot_base = [10, 11, 12, 13, 14]
        plot_other = [10, 11, 12, 13, 14]

        fig = plt.figure(fig_num)
        axes = fig.add_subplot(projection='3d')

        axes.set_xlabel("X axis")
        axes.set_ylabel("Y axis")
        axes.set_zlabel("Z axis")

        # Plot buoys and vertical 'ropes'
        for buoy in self.buoys:
            axes.scatter(buoy[0], buoy[1], buoy[2], c='b', linewidths=5)
            axes.plot([buoy[0], buoy[0]],
                      [buoy[1], buoy[1]],
                      [buoy[2], buoy[2] - 10], c='g')

        # plot base_link gt pose3s
        for i_base, pose3 in enumerate(self.base_pose3s):
            # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)
            if i_base in plot_base or len(plot_base) == 0:
                gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)

        # plot camera gt pose3s
        if other_pose3s is not None:
            for i_other, pose3 in enumerate(other_pose3s):
                # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                if i_other in plot_other or len(plot_other) == 0:
                    gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                    # plot fov
                    for i_ray, ray in enumerate(self.fov_rays):
                        point_end = pose3.transformFrom(5 * ray)
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
                        start, direction = self.camera_center_point_direction(pose3)

                        intrcpt_status, intrcpt_w_coords, _, in_bounds = plane.find_intersection(start,
                                                                                                 direction)

                        if in_bounds:
                            axes.scatter(intrcpt_w_coords[0], intrcpt_w_coords[1], intrcpt_w_coords[2], c='r')

        # plt.axis('equal')
        plt.title("Testing the transform")
        plt.show()

    def find_fov_corner_coords(self, plane_id, pose_id):
        """
        Find the intersections of the fov with the define plane
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

    def find_plane_corner_pixels(self, plane_id, pose_id):
        """
        Each plane is defined by a start buoy and an end buoy, a lower buoy is defined as being below the start buoy.
        The fourth corner is placed below the end buoy. This function will return the pixel positions of these corners.
        """

        corner_pixels = np.zeros((4, 2), dtype=np.float64)

        # the world coordinates of the plane corners
        w_corner_coords = np.zeros((4, 3), dtype=np.float64)
        w_corner_coords[0, :] = self.planes[plane_id].start_coord
        w_corner_coords[1, :] = self.planes[plane_id].end_coord
        w_corner_coords[2, :] = self.planes[plane_id].start_bottom_coord
        w_corner_coords[3, :] = self.planes[plane_id].end_bottom_coord

        # Compute an intersection for each ray
        for i_corner in range(4):
            x_pixel, y_pixel = self.find_pixels_of_3d_point(pose_id, w_corner_coords[i_corner, :])

            corner_pixels[i_corner, :] = x_pixel, y_pixel

        return corner_pixels

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

    def camera_center_point_direction(self, pose):
        """
        returns the center point and direction of central ray of camera given pose
        can accept pose_id or id
        """
        if isinstance(pose, gtsam.Pose3):
            camera_pose3 = pose
        elif 0 <= int(pose) < len(self.camera_pose3s):
            camera_pose3 = self.camera_pose3s[pose]
        else:
            print("Malformed request")

        center_ray = self.fov_rays[-1]
        center_end_point_pose3 = camera_pose3.transformFrom(center_ray)

        end = np.array([center_end_point_pose3[0],
                        center_end_point_pose3[1],
                        center_end_point_pose3[2]], dtype=np.float64)

        start = np.array([camera_pose3.x(),
                          camera_pose3.y(),
                          camera_pose3.z()])

        direction = end - start

        return start, direction

    def process_images(self, path_name, image_path, verbose=False):

        for current_pose_id in range(len(self.camera_pose3s)):
            current_img_id = int(self.base_pose[current_pose_id][-1])
            camera_pose3 = self.camera_pose3s[current_pose_id]

            for plane_id, plane in enumerate(self.planes):
                # Find which if plane to apply the image to
                c_center, c_direction = self.camera_center_point_direction(camera_pose3)
                status, w_coords, p_coords, in_bounds = plane.find_intersection(c_center, c_direction)

                # properly oriented plane that is centrally located w.r.t. camera frame
                if status and in_bounds:
                    corners = self.find_plane_corner_pixels(plane_id=plane_id, pose_id=current_pose_id)

                    # TODO remove hardcoded left camera
                    # TODO figure why this increment is need, should not be
                    # mod_id = int(img_id + 1)
                    mod_id = int(current_img_id + 1)
                    img = cv2.imread(image_path + f"{mod_id}.jpg")

                    if not isinstance(img, np.ndarray):
                        continue

                    if verbose:
                        img_verbose = img.copy()

                        # Draw corners
                        for corner in corners:
                            if math.isnan(corner[0]) or math.isnan(corner[1]):
                                continue
                            corner_x = int(corner[0] // 1)
                            corner_y = int(corner[1] // 1)
                            img_verbose = cv2.circle(img_verbose, (corner_x, corner_y), 5, (0, 0, 255), -1)

                        # Draw center
                        center = self.find_pixels_of_3d_point(pose_id=current_pose_id, map_point=w_coords)

                        if not math.isnan(center[0]) and not math.isnan(center[1]):
                            center_x = int(center[0] // 1)
                            center_y = int(center[1] // 1)
                            img_verbose = cv2.circle(img_verbose, (center_x, center_y), 5, (255, 0, 255), -1)

                        cv2.imwrite(path_name + f"images_registered/Processing_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                    img_verbose)

                    # perform extraction
                    plane_spatial_width = plane.mag_x  # meters
                    plane_spatial_height = plane.mag_y  # meters

                    plane_pixel_width = int(plane_spatial_width * self.spatial_2_pixel//1)
                    plane_pixel_height = int((plane_spatial_height * self.spatial_2_pixel)//1)

                    destination_corners = np.array([[0, 0],
                                                    [plane_pixel_width - 1, 0],
                                                    [0, plane_pixel_height - 1],
                                                    [plane_pixel_width - 1, plane_pixel_height - 1]], dtype=np.float64)

                    homography = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                                             destination_corners.astype(np.float32))
                    # Apply homography to image
                    img_warped = cv2.warpPerspective(img, homography,
                                                     (plane_pixel_width, plane_pixel_height))

                    # apply homography to form a mask
                    mask_warped = cv2.warpPerspective(np.ones_like(img) * 255, homography,
                                                      (plane_pixel_width, plane_pixel_height))

                    self.planes[plane_id].images.append(img_warped)
                    self.planes[plane_id].masks.append(mask_warped)

                    cv2.imwrite(path_name + f"images_warped/Warping_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                img_warped)

                    cv2.imwrite(path_name + f"images_masked/Warping_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                mask_warped)


# %% Load and process data
# This is the gt of the base_link indexed for the left images
# base_gt = read_csv_to_array('data/left_gt.csv')
# left_info = read_csv_to_list('data/left_info.csv')
# buoy_info = read_csv_to_array('data/buoy_info.csv')

# linux
path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/online_testing/'
img_path_name = path_name + "left/"
gt_base = read_csv_to_array(path_name + 'camera_gt.csv')
base = read_csv_to_array(path_name + 'camera_est.csv')
left_info = read_csv_to_list(path_name + 'left_info.csv')
buoy_info = read_csv_to_array(path_name + 'buoys.csv')

# Define the connections between buoys
# list of start and stop indices of buoys
# TODO remove hard coded rope structure
ropes = [[0, 4], [4, 2],
         [1, 5], [5, 3]]

img_map = image_mapping(gt_base_link_poses=gt_base,
                        base_link_poses=base,
                        camera_info=left_info,
                        relative_camera_pose=None,
                        buoy_info=buoy_info,
                        ropes=ropes)

# %% Plot
img_map.plot_fancy(img_map.camera_pose3s)
# img_map.plot_fancy(img_map.gt_camera_pose3s)  # plot the ground ruth as other
# plot_fancy(base_gt_pose3s, left_gt_pose3s, buoy_info, points)
img_map.process_images(path_name, img_path_name, True)

# %% Testing parameters
do_testing_1 = False
do_testing_2 = False
do_testing_3 = False
do_testing_4 = False
# %% Testing 1
if do_testing_1:
    print("Testing 1")

    p_org = np.array([-5.0, 4.0, -0])
    p_x = np.array([-5.0, 9.0, -0])
    p_y = np.array([-5.0, 4.0, -15])

    point = np.array([-4.0, 3.0, -6])
    direction = np.array([-1.0, 0, 0])

    a, b, c, d = img_map.planes[0].find_intersection(point, direction)

    corner_coords = img_map.find_fov_corner_coords(0, 25)

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

    # buoys
    test_point_b0 = np.array([-5.0, 4.0, 0])
    test_point_b1 = np.array([-5.0, 9.0, 0])

    x_pix_0, y_pix_0 = img_map.find_pixels_of_3d_point(pose_id, test_point_b0)
    x_pix_1, y_pix_1 = img_map.find_pixels_of_3d_point(pose_id, test_point_b1)

    # Lower points
    depth = 7.5
    test_point_b0_deep = test_point_b0
    test_point_b0_deep[2] = -depth

    test_point_b1_deep = test_point_b1
    test_point_b1_deep[2] = -depth

    x_pix_0_d, y_pix_0_d = img_map.find_pixels_of_3d_point(pose_id, test_point_b0_deep)
    x_pix_1_d, y_pix_1_d = img_map.find_pixels_of_3d_point(pose_id, test_point_b1_deep)

    # Center
    # test_point_c = np.array([-5.0, 6.592, -0.706])
    # x_pix_c, y_pix_c = img_map.find_pixels_of_3d_point(pose_id, test_point_c)

    # Mark images
    for i in range(int(test_next_n + 1)):
        img_id = proper_img_id + i
        img = cv2.imread(f"/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/left/l_{img_id}.jpg")

        # Buoys
        img_marked = cv2.circle(img, (int(x_pix_0 // 1), int(y_pix_0 // 1)), 5, (0, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_1 // 1), int(y_pix_1 // 1)), 5, (0, 0, 255), -1)

        # Lower points
        img_marked = cv2.circle(img, (int(x_pix_0_d // 1), int(y_pix_0_d // 1)), 5, (255, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_1_d // 1), int(y_pix_1_d // 1)), 5, (255, 0, 255), -1)

        # Center
        # img_marked = cv2.circle(img_marked, (int(x_pix_c // 1), int(y_pix_c // 1)), 5, (0, 255, 255), -1)

        img_marked[:, int(img_map.cx), :] = (0, 255, 255)
        img_marked[int(img_map.cy), :, :] = (0, 255, 255)

        #
        cv2.imshow(f"Marked Image: {img_id}", img_marked)
        cv2.imwrite(f"Test2:{proper_img_id}_{img_id}.jpg", img_marked)

    # Waits for a keystroke
    cv2.waitKey(0)

    # Destroys all the windows created
    cv2.destroyAllWindows()

# %% Testing 3 - checking rotation transforms and equivalences
if do_testing_3:
    """
    Testing for the transform between base_link and left camera
    """
    pose_id = 25
    base_pose3 = img_map.base_pose3s[pose_id]
    # Ros transform using quaternions reported by ros
    # stnfish using rpy in stonefish config -> quaternion w/ rpy_2_quat.py
    ros_b_2_c = img_map.return_left_relative_pose()
    stnfsh_b_2_c = img_map.return_left_relative_pose(use_rpy=True)

    ros_left_pose3 = base_pose3.transformPoseFrom(ros_b_2_c)
    stnfsh_left_pose3 = base_pose3.transformPoseFrom(stnfsh_b_2_c)

    # Convert matrix to rpy w/ gtsam
    ros_check = ros_b_2_c.rotation().rpy()
    stnfsh_check = stnfsh_b_2_c.rotation().rpy()

# %% Testing 4 - mark images with centers
if do_testing_4:
    for pose in img_map.base_pose:
        img_id = int(pose[-1])

        img = cv2.imread(img_path_name + f"{img_id}.jpg")

        center_x = img_map.cx
        center_y = img_map.cy

        img[:, int(img_map.cx), :] = (0, 255, 255)
        img[int(img_map.cy), :, :] = (0, 255, 255)

        cv2.imwrite(path_name + f"centers/centers_{img_id}.jpg", img)
