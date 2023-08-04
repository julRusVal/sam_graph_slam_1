#!/usr/bin/env python

"""
This is a little script to edit some old rosbag so that the detection IDs indicate that a buoy was detected.
 It could also be used as a template on how to work with and edit rosbags by topic or type
"""
import rosbag
from sss_object_detection.consts import ObjectID
from vision_msgs.msg import Detection2DArray

# Function to modify the desired field in the message
def modify_message(msg, new_value):
    # Modify the field of a vision_msg/Detection2DArray message you want to change
    for detection in msg.detections:
        for result in detection.results:
            result.id = new_value

    return msg

# Input and output bag file paths
input_bag_file = "/home/julian/sam_sim_1.bag"
output_bag_file = "/home/julian/sam_sim_1_edited.bag"

# Field value to change
topic_to_edit = "/sam/payload/sidescan/detection_hypothesis"
new_value = ObjectID.BUOY.value  # 2

# Open the input and output bag files
with rosbag.Bag(output_bag_file, 'w') as output_bag:
    for topic, msg, timestamp in rosbag.Bag(input_bag_file).read_messages():
        # Check if the message is of interest, by type or topic
        # Type test -> msg._type == 'vision_msgs/Detection2DArray'
        if topic == topic_to_edit:
            # Modify the field in the message
            msg = modify_message(msg, new_value)

        # Write the modified or unmodified message to the output bag
        output_bag.write(topic, msg, timestamp)
print("Bag file modification complete.")