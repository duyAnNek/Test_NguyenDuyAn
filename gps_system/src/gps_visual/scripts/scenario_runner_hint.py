"""
Hints for bag-driven scenarios (S1–S5). Does not embed acceptance KPI thresholds.

Examples (after sourcing ROS 2 Humble):
  ros2 bag play your_bag --topics /camera/image_raw /gps/fix /imu/data
  ros2 topic pub /vehicle/wheel_twist geometry_msgs/msg/TwistStamped ...
  ros2 topic pub /gps/metrics std_msgs/msg/Float32MultiArray "data: [2.0, 8.0, 35.0]"

S2: watch /uturn_detected (std_msgs/Bool) while turning ~180 deg within 2 s.
S3: log /lane_position (UInt8 0/1/2) vs human labels for offline scoring.
S5: compare /odometry to reference when /gps/integrity_state indicates LOST.
"""

if __name__ == "__main__":
    print(__doc__)
