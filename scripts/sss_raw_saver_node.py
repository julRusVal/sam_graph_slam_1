#!/usr/bin/env python3
import rospy
import cv2
from smarc_msgs.msg import Sidescan
import numpy as np
from PIL import Image
from sam_slam_utils.sam_slam_helpers import write_array_to_csv

"""
Utility for viewing and saving the raw sidescan imagery produced by the sss_detector.
This is based on the view_sidescan.py found in smarc_utils/sss_viewer

Will save the all raw measurements and continue to show a buffers worth at all times. 
"""


class sss_recorder:
    def __init__(self, time_out, buffer_len):
        # Sonar parameters
        self.sss_topic = "/sam/payload/sidescan"
        sss_data_len = 1000  # This is determined by the message

        # Saved data
        self.data = []
        self.seq_ids = []

        # Output
        self.buffer_len = buffer_len
        self.output = np.zeros((buffer_len, 2 * sss_data_len), dtype=np.ubyte)

        self.time_out = time_out
        self.data_written = False

        # Initialize node
        rospy.init_node('sidescan_viewer', anonymous=True)
        self.sss_last_time = rospy.Time.now()

        cv2.namedWindow('Sidescan image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sidescan image', 2 * 256, self.buffer_len)

        # SSS data subscription
        self.sss_count = 0
        self.sss_subscription = rospy.Subscriber("/sam/payload/sidescan", Sidescan, self.sss_callback)

        self.time_out_timer = rospy.Timer(rospy.Duration(2),
                                          self.time_out_callback)

        # spin() simply keeps python from exiting until this node is stopped
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            # Display sss data
            resized = cv2.resize(self.output, (2 * 256, self.buffer_len), interpolation=cv2.INTER_AREA)
            cv2.imshow("Sidescan image", resized)
            cv2.waitKey(1)
            r.sleep()

    def sss_callback(self, msg):
        """
        Read new sss data and add to existing imagery
        """
        self.sss_count += 1

        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])

        # update the data
        self.data.append(meas)
        self.seq_ids.append(msg.header.seq)

        # Update the output
        self.output[1:, :] = self.output[:-1, :]  # shift data down
        self.output[0, :] = meas

        # update time_out timer
        self.sss_last_time = rospy.Time.now()

    def time_out_callback(self, event):
        if len(self.data) > 0 and not self.data_written:
            elapsed_time = (rospy.Time.now() - self.sss_last_time).to_sec()
            if elapsed_time > self.time_out:
                print('Saving data!')
                data_array = np.flipud(np.asarray(self.data))
                seqs_array = np.flipud(np.asarray(self.seq_ids).reshape((-1, 1)))

                data_len = data_array.shape[0]
                seqs_len = seqs_array.shape[0]
                # Save sonar as csv
                write_array_to_csv(f'sss_data_{data_len}.csv', data_array)

                # Save sonar as jpg
                data_image = Image.fromarray(data_array)
                data_image.save(f'sss_data_{data_len}.jpg')

                # Save seq ids to help with related the image back to the sensor readings
                write_array_to_csv(f'sss_seqs_{seqs_len}.csv', seqs_array)

                self.data_written = True

                # Output
                print('SSS recording complete!')
                print(f'Callback count : {self.sss_count} - Output length: {data_len} ')



if __name__ == '__main__':
    sss_recorder = sss_recorder(10, 500)