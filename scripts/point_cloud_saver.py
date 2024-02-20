#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import csv


class PointCloudSaver:
    def __init__(self, topic='/sam0/mbes/odom/bathy_points', save_file='point_cloud_data', save_interval=10):
        self.point_cloud_topic = topic
        self.save_file = save_file
        self.save_interval = rospy.Duration.from_sec(save_interval)
        self.last_data_time = None
        self.data_written = False

        self.stacked_point_cloud_data = None

        rospy.init_node('point_cloud_saver', anonymous=True)
        print("Initializing point cloud saver")

        # Subscribe to the point cloud topic
        rospy.Subscriber(self.point_cloud_topic, pc2.PointCloud2, self.point_cloud_callback)

        # Set up a timer to check for data saving periodically
        rospy.Timer(self.save_interval, self.save_timer_callback)

    def point_cloud_callback(self, msg):
        if self.last_data_time is None:
            print("Pointcloud2 data received")

        # Update the last received data time
        self.last_data_time = rospy.Time.now()

        # Convert PointCloud2 message to NumPy array
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_array = np.array(list(pc_data))

        # Store the point cloud data
        if self.stacked_point_cloud_data is None:
            self.stacked_point_cloud_data = pc_array
        else:
            self.stacked_point_cloud_data = np.dstack([self.stacked_point_cloud_data, pc_array])

    def save_timer_callback(self, event):
        # Check if no new data has been received for the specified interval
        if self.last_data_time is None:
            return
        if rospy.Time.now() - self.last_data_time > self.save_interval:
            # Save point cloud data to a CSV file
            # self.save_to_csv()
            self.save_stacked_point_cloud()

    def save_to_csv(self):
        if self.point_cloud_data is not None:
            with open(self.save_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["x", "y", "z"])  # CSV header

                for point in self.point_cloud_data:
                    csv_writer.writerow(point)

            self.data_written = True

            print(f"Point cloud data saved to {self.save_file}")
        else:
            print("No point cloud data to save.")

        if self.data_written:
            rospy.signal_shutdown("Script shutting down")

    def save_stacked_point_cloud(self):
        if self.stacked_point_cloud_data is not None:
            np.save(f'/home/julian/catkin_ws/src/sam_slam/processing scripts/data/{self.save_file}.npy', self.stacked_point_cloud_data)
            self.data_written = True
            print(f"Stacked point cloud saved to {self.save_file}")
        else:
            print("No point cloud data to save.")

        if self.data_written:
            rospy.signal_shutdown("Script shutting down")

if __name__ == '__main__':
    try:
        # Change '/your/point_cloud_topic' to the actual point cloud topic you want to subscribe to
        # Change 'point_cloud_data.csv' to the desired file name
        # Change 10 to the desired save interval in seconds
        point_cloud_subscriber = PointCloudSaver(topic='/sam0/mbes/odom/bathy_points',
                                                      save_file='point_cloud_data',
                                                      save_interval=5)

        # Run the ROS node
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
