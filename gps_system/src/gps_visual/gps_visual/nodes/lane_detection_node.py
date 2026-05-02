"""BEV lane heuristic + optional CLAHE (low light); publishes /lane_position UInt8."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

from gps_visual.lane_bev import LaneBEVDetector, LanePosition


class LaneDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_detection_node")
        self._bridge = CvBridge()
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.declare_parameter("low_light_enhance", True)
        self.declare_parameter("ipm_homography_npz", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self._enhance = bool(self.get_parameter("low_light_enhance").value)
        H_path = str(self.get_parameter("ipm_homography_npz").value)
        H = None
        if H_path and pathlib.Path(str(H_path)).is_file():
            H = np.load(str(H_path))["H"]
        self._det = LaneBEVDetector(H_ipm=H)
        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self._on_img, 10)
        self._pub = self.create_publisher(UInt8, "/lane_position", 10)

    def _on_img(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self._enhance:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        lp = self._det.infer_lane_position(frame)
        out = UInt8()
        out.data = int(lp)
        self._pub.publish(out)


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
