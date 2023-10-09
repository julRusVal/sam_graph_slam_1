import rospy
import cv2
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image

# Set the ROS bag file path and output video file path
bag_file = 'your_input.bag'
output_video_file = 'output.mp4'

# Initialize the ROS node
rospy.init_node('rosbag_to_video')

# Create a VideoWriter object to save the frames as an MP4 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
video_writer = cv2.VideoWriter(output_video_file, fourcc, 30, (640, 480))  # Adjust frame rate and resolution as needed

# Create a CvBridge instance to convert ROS Image messages to OpenCV images
bridge = CvBridge()

# Open the ROS bag file
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == '/camera/image_raw':  # Replace with the actual image topic in your bag file
            try:
                # Convert the ROS Image message to an OpenCV image
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                # Write the frame to the video
                video_writer.write(frame)
            except Exception as e:
                print(f"Error converting frame: {e}")

# Release the video writer and close the video file
video_writer.release()

print("Video conversion complete.")
