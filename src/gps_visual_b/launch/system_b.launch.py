"""Launch Part B stack: landmark DB, VO+U-turn, lane, GPS monitor, fusion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("gps_visual_b")
    params_file = os.path.join(share, "config", "params.yaml")
    return LaunchDescription(
        [
            Node(
                package="gps_visual_b",
                executable="landmark_db_node",
                name="landmark_db_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="gps_visual_b",
                executable="visual_odometry_node",
                name="visual_odometry_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="gps_visual_b",
                executable="lane_detection_node",
                name="lane_detection_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="gps_visual_b",
                executable="gps_monitor_node",
                name="gps_monitor_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="gps_visual_b",
                executable="sensor_fusion_node",
                name="sensor_fusion_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
