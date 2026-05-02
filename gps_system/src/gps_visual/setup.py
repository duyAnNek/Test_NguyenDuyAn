import os
from glob import glob

from setuptools import find_packages, setup

package_name = "gps_visual"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@example.com",
    description="GPS + visual fallback ROS 2 nodes (CPU ONNX).",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            f"landmark_db_node = {package_name}.nodes.landmark_db_node:main",
            f"visual_odometry_node = {package_name}.nodes.visual_odometry_node:main",
            f"lane_detection_node = {package_name}.nodes.lane_detection_node:main",
            f"gps_monitor_node = {package_name}.nodes.gps_monitor_node:main",
            f"sensor_fusion_node = {package_name}.nodes.sensor_fusion_node:main",
        ],
    },
)
