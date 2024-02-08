#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:59:40 2023

@author: julian
"""

import sys
import pygame
import math
import numpy as np
from numpy.random import default_rng
import csv
from sam_slam_utils.sam_slam_helpers import angle_between_rads


# %% Function Definitions

def norm_ang(angle):
    return math.remainder(angle, 2 * np.pi)


def save_array_as_csv(array, file_path):
    """
    Saves a 2D array as a CSV file.

    Parameters:
        array (list of lists): The 2D array to be saved.
        file_path (str): The path to the file where the array will be saved.

    Returns:
        None
    """
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in array:
            writer.writerow(row)


def read_csv_to_array(file_path):
    """
    Reads a CSV file and returns the contents as a 2D Numpy array.

    Parameters:
        file_path (str): The path to the CSV file to be read.

    Returns:
        numpy.ndarray: The contents of the CSV file as a 2D Numpy array.
    """
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        return np.array(data, dtype=np.float64)
    except:
        return -1


# %% Class Definitions

class Simulation:
    def __init__(self, screen_length, screen_size, buoys=1, walls=1, farm_theta=0, target_fps=30):
        # =============================================================================
        # screen_length: The 'physical' size fo the area being shown in the simulation
        # screen_size: The number of pixels of the screen
        # =============================================================================

        self.screen_x_dim = screen_length
        self.screen_y_dim = screen_length
        self.screen_w = screen_size
        self.screen_h = screen_size
        self.scale = screen_size / screen_length
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.target_fps = target_fps

        self.buoys = buoys
        self.walls = walls
        self.farm_theta = farm_theta
        self.farm_walls, self.farm_buoys = self.generate_algae_farm()

        self.gt_col = (0, 204, 0)
        self.path_col = (204, 0, 0)  # red
        self.back_col = (0, 102, 204)  # nice dark blue
        self.buoy_col = (255, 255, 255)
        self.rope_col = (255, 255, 0)
        self.char_col = (255, 255, 51)
        self.dtct_col = (204, 0, 204)

        self.buoy_rad = 5

        self.running = True

    def generate_algae_farm(self):
        # =============================================================================
        # Source: https://github.com/cisprague/smarc_missions/blob/master/smarc_bt/notebooks/algae_buoys.ipynb
        # =============================================================================
        n_walls = int(self.walls)
        n_buoys = int(self.buoys)
        theta = self.farm_theta

        # Generate algae walls
        walls = list()
        for x in np.linspace(-10, 10, num=n_walls):
            wall = list()
            for y in np.linspace(-10, 10, num=n_buoys):
                wall.append([x, y, 0.0])
            walls.append(np.array(wall))
        walls = np.array(walls)

        # rotate walls to ensure that stuff below generally works
        # NOTE: this code is inefficient because this is just for generating an example
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        newwalls = list()
        for wall in walls:
            newwall = list()
            for buoy in wall:
                newbuoy = np.hstack((
                    np.dot(rot, buoy[:2]),
                    buoy[-1]
                ))
                newwall.append(newbuoy)
            newwalls.append(newwall)
        walls = np.array(newwalls)

        # Center The farm
        x_offset = self.screen_x_dim / 2
        y_offset = self.screen_y_dim / 2

        # Apply offsets
        walls[:, :, 0] = walls[:, :, 0] + x_offset
        walls[:, :, 1] = walls[:, :, 1] + y_offset

        buoys = np.reshape(walls[:, :, :2], (-1, 2))

        return walls, buoys

    def map_2_game(self, point):
        x, y = point
        flipped_y = self.screen_h - y * self.scale
        return (x * self.scale, flipped_y)

    # Plotting methonds
    def plot_static_elements(self):
        # Clear screen
        self.screen.fill(self.back_col)

        # Plot buoys
        if self.buoys is not None:
            for i, position in enumerate(self.farm_buoys):
                pygame.draw.circle(self.screen,
                                   self.buoy_col,
                                   self.map_2_game(position),
                                   self.buoy_rad)

        # Plot ropes
        # if self.ropes is not None:
        #     for i, line_inds in enumerate(self.ropes):
        #         pygame.draw.line(simulation.screen, 
        #                          simulation.rope_col, 
        #                          simulation.map_2_game(landmarks[line_inds[0]]), 
        #                          simulation.map_2_game(landmarks[line_inds[1]]),  
        #                          2)
        # Plot ropes (new)
        for rope in self.farm_walls:
            if len(rope) > 1:
                for i in range(1, len(rope)):
                    pygame.draw.line(simulation.screen,
                                     simulation.rope_col,
                                     simulation.map_2_game(rope[i - 1, :2]),
                                     simulation.map_2_game(rope[i, :2]),
                                     3)

    def plot_path(self, path, color=None):
        # Default color is the path
        if color is None:
            color = self.path_col

        for i, p in enumerate(path):
            if i > 0:
                pygame.draw.line(self.screen,
                                 color,
                                 self.map_2_game(path[i - 1][0:2]),
                                 self.map_2_game(p[0:2]),
                                 1)

    def plot_char(self, position, char_rad):
        pygame.draw.circle(simulation.screen,
                           self.char_col,
                           simulation.map_2_game(position),
                           char_rad)

    def plot_detection_zone(self, pose, detect_range, detect_half_angle, color):
        position = np.array([pose[0], pose[1]])
        heading = pose[2]
        # TODO This looks a wee bit lazy and ugly!!

        theta_plus = math.remainder(heading + np.pi / 2 + detect_half_angle,
                                    2 * np.pi)

        theta_minus = math.remainder(heading + np.pi / 2 - detect_half_angle,
                                     2 * np.pi)

        rot_plus = np.array([[np.cos(theta_plus), -np.sin(theta_plus)],
                             [np.sin(theta_plus), np.cos(theta_plus)]])

        rot_minus = np.array([[np.cos(theta_minus), -np.sin(theta_minus)],
                              [np.sin(theta_minus), np.cos(theta_minus)]])

        range_array = np.array([detect_range, 0])

        # These are the relative positions of the vertices that designate detection zone
        P_plus = np.dot(rot_plus, range_array)
        P_minus = np.dot(rot_minus, range_array)

        # List to contain the vertices of the detection zone, 
        # one is always the center of the agen.
        detection_zone_left = [self.map_2_game(position)]
        detection_zone_right = [self.map_2_game(position)]

        # Use the relative positions to find the absolute positions of the detection zone
        # due to symetry the other detection zone can be found from the relative positions above
        for i, array in enumerate([P_plus, P_minus]):
            # left detection zone
            lp_x = position[0] + array[0]
            lp_y = position[1] + array[1]

            # right detection zone
            rp_x = position[0] - array[0]
            rp_y = position[1] - array[1]

            detection_zone_left.append(self.map_2_game((lp_x, lp_y)))
            detection_zone_right.append(self.map_2_game((rp_x, rp_y)))

        # OLD
        # pygame.draw.polygon(self.screen, (0,255,0), detection_zone_left, width = 3)
        # pygame.draw.polygon(self.screen, (0,255,0), detection_zone_right, width = 3)

        self.draw_polygon(detection_zone_left, color, 255 / 2)
        self.draw_polygon(detection_zone_right, color, 255 / 2)

    def plot_detections(self, detections):
        thickness = 5
        for i, detection in enumerate(detections):
            buoy_id = detection[3]
            buoy_pos = self.farm_buoys[buoy_id]
            pygame.draw.circle(simulation.screen,
                               self.dtct_col,
                               simulation.map_2_game(buoy_pos),
                               self.buoy_rad + thickness,
                               thickness)

    def update_sim_status(self, keys):
        # Quit on q key press
        if keys[pygame.K_q]:
            self.running = False

    @staticmethod
    def write_text(text, font_size, position, screen, center=False):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        if center:
            text_rect.centerx = position[0]
            text_rect.centery = position[1]
        else:
            text_rect.x = position[0]
            text_rect.y = position[1]
        screen.blit(text_surface, text_rect)

    def draw_polygon(self, vertices, color, alpha):

        # Create a new surface with per-pixel alpha
        surface = pygame.Surface((self.screen.get_width(),
                                  self.screen.get_height()))
        surface.set_alpha(alpha)
        surface.fill((0, 0, 0, 0))
        surface.convert_alpha()

        # Draw the polygon on the new surface
        polygon_color = color
        pygame.draw.polygon(surface, polygon_color, vertices)

        # Blit the surface onto the screen
        self.screen.blit(surface, (0, 0))


class SAM_sim:
    def __init__(self, x, y, angle, velocity, max_velocity, rev_per_sec, character_radius, simulation,
                 detection_range, detect_half_angle):
        # Simulation
        self.sim = simulation

        # random number generator
        self.rng = default_rng()

        # Initial
        self.x = x
        self.y = y
        self.angle = angle
        self.vel = velocity

        # Motion limits
        self.max_vel = max_velocity
        self.vel_inc = self.max_vel / 20
        self.max_rps = rev_per_sec
        self.rad = character_radius
        self.turn_rate = 2 * np.pi * self.max_rps / self.sim.target_fps

        # Initial position noise
        self.dist_sigma_initial = 0.01
        self.ang_sigma_initial = np.pi / 50

        # Motion noise
        self.dist_sigma = 0.01
        self.ang_sigma = np.pi / 200

        # Sensor settings
        self.detect_range = detection_range
        self.detect_half_angle = detect_half_angle

        #
        # self.path is the measured trajectory
        # self.path_noisy is with noise applied (the ground truth)
        self.path = [[self.x, self.y, self.angle]]
        self.path_noisy = [self.Generate_noisy_initial_pose()]

        self.odometry = []
        self.odometry_noisy = []

        # Detections
        self.detections = []

    def update_sam_keyboard(self, keys):
        # =============================================================================
        # Take in keyboard input and modify same state accordingly     
        # =============================================================================
        # Increase velocity on up arrow key press
        if keys[pygame.K_UP]:
            self.vel = min(self.vel + self.vel_inc, self.max_vel)

        # Decrease velocity on down arrow key press
        if keys[pygame.K_DOWN]:
            self.vel = max(self.vel - self.vel_inc, -self.max_vel)

        # Turn counterclockwise on left arrow key press
        if keys[pygame.K_LEFT]:
            self.angle += self.turn_rate
            self.angle = math.remainder(self.angle, 2 * np.pi)

        # Turn clockwise on right arrow key press
        if keys[pygame.K_RIGHT]:
            self.angle -= self.turn_rate
            self.angle = math.remainder(self.angle, 2 * np.pi)

        # Calculate new position based on velocity and angle
        self.x += self.vel / self.sim.target_fps * math.cos(self.angle)
        self.y += self.vel / self.sim.target_fps * math.sin(self.angle)

        # Restrict character to screen
        self.x = min(max(self.x, 0 + self.rad / self.sim.scale),
                     simulation.screen_x_dim - self.rad / self.sim.scale)

        self.y = min(max(self.y, 0 + self.rad / self.sim.scale),
                     simulation.screen_y_dim - self.rad / self.sim.scale)

        # log pose
        self.path.append([self.x, self.y, self.angle])

    def Generate_noisy_initial_pose(self):
        noisy_initial = [self.path[0][0] + self.rng.normal(0.0, self.dist_sigma_initial),
                         self.path[0][1] + self.rng.normal(0.0, self.dist_sigma_initial),
                         self.path[0][2] + self.rng.normal(0.0, self.ang_sigma_initial)]
        return noisy_initial

    def generate_noisy_path(self):
        # =============================================================================
        # This will calculate odometry from the measurements -> corrupt with noise
        # -> add to the last ground truth
        # Keywords: motion model, ground truth
        # =============================================================================
        current_pose = self.path[-1]
        previous_pose = self.path[-2]

        delta_x = current_pose[0] - previous_pose[0]
        delta_y = current_pose[1] - previous_pose[1]
        delta_dist = (delta_x ** 2 + delta_y ** 2) ** (1 / 2)
        delta_theta = norm_ang(current_pose[2] - previous_pose[2])

        # Build up the odometry lists
        self.odometry.append([delta_dist, 0, delta_theta])

        delta_dist_noisy = delta_dist + self.rng.normal(0.0, self.dist_sigma)
        delta_theta_noisy = norm_ang(delta_theta + self.rng.normal(0.0, self.ang_sigma))

        self.odometry_noisy.append([delta_dist_noisy,
                                    0,
                                    delta_theta_noisy])

        # Use the new noisy odometry (with noise) to compute the ground truth
        last_pose_noisy = self.path_noisy[-1]

        next_heading = norm_ang(last_pose_noisy[2] + delta_theta_noisy)
        next_x = last_pose_noisy[0] + delta_dist_noisy * math.cos(next_heading)
        next_y = last_pose_noisy[1] + delta_dist_noisy * math.sin(next_heading)

        self.path_noisy.append([next_x, next_y, next_heading])

    def generate_simulated_detections(self):
        # Detections are stored in a list o lists: 
        # [array index at which detection occurred, distance, relative bearing, landmark id]
        # The landmark shouldn't be used except for checking!!!
        # Detections are computed with ground truth information!!!

        current_pose = self.path_noisy[-1]
        current_ind = len(self.path_noisy) - 1

        for ind_lm, lm_state in enumerate(self.sim.farm_buoys):
            # Position rel to ground truth
            rel_x = lm_state[0] - current_pose[0]
            rel_y = lm_state[1] - current_pose[1]
            # bearing of lm w.r.t. the agent
            bearing = math.atan2(rel_y, rel_x)
            # relative bearing is the smallest angle between the bearing of the landmark and the heading of the agent
            rel_bearing = angle_between_rads(bearing, current_pose[2])
            # Check if the target is within the detection window on either side of the agent
            # currently this does not check for min/max range or anything else
            detect_dist = (rel_x ** 2 + rel_y ** 2) ** (1 / 2)
            if detect_dist > self.detect_range:
                continue

            if abs(rel_bearing) <= (np.pi / 2 + self.detect_half_angle) and abs(rel_bearing) >= (
                    np.pi / 2 - self.detect_half_angle):
                self.detections.append([current_ind, detect_dist, rel_bearing, ind_lm])


# %% Define Simulation
simulation = Simulation(75,
                        600,
                        4,
                        2,
                        np.pi / 5)

sam = SAM_sim(x=20.0,
              y=20.0,
              angle=0.0,
              velocity=0.0,
              max_velocity=5,
              rev_per_sec=0.25,
              character_radius=5,
              simulation=simulation,
              detection_range=15,
              detect_half_angle=np.pi / 32)

# %% Begin Simulation
# rng = default_rng()
# Initialize Pygame
pygame.init()

# Create game clock
clock = pygame.time.Clock()

# Start game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # Get keys pressed
    keys = pygame.key.get_pressed()
    simulation.update_sim_status(keys)
    sam.update_sam_keyboard(keys)

    sam.generate_noisy_path()

    sam.generate_simulated_detections()

    # =============================================================================
    # Start plotting 
    # =============================================================================
    # Clear screen and plot the static elements: buoys and ropes
    simulation.plot_static_elements()

    # simulation.write_text('test', int(12), (10,10), simulation.screen)
    simulation.write_text('SAMnautica', int(36), (simulation.screen_w / 2, 10), simulation.screen, center=True)

    # Draw path: measured and ground truth
    simulation.plot_path(sam.path)
    simulation.plot_path(sam.path_noisy, simulation.gt_col)

    # Show the detection zones: measured and ground truth
    simulation.plot_detection_zone(sam.path[-1],
                                   sam.detect_range,
                                   sam.detect_half_angle,
                                   simulation.path_col)

    simulation.plot_detection_zone(sam.path_noisy[-1],
                                   sam.detect_range,
                                   sam.detect_half_angle,
                                   simulation.gt_col)

    # Show detections
    simulation.plot_detections(sam.detections)

    # Draw character
    simulation.plot_char((sam.x, sam.y), sam.rad)

    # Update screen
    pygame.display.update()

    # Limit game loop to target FPS
    clock.tick(simulation.target_fps)

# Save 
save_array_as_csv(sam.path, '../processing scripts/trajectory.csv')
save_array_as_csv(sam.path_noisy, '../processing scripts/trajectory_gt.csv')
save_array_as_csv(simulation.farm_buoys, '../processing scripts/landmarks.csv')

# Quit Pygame
pygame.display.quit()
pygame.quit()

# Exit the application
sys.exit()
