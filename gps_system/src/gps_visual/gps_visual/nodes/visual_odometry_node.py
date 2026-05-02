"""Monocular VO + optional wheel scale; publishes /vo/pose."""

from __future__ import annotations

import time

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool, Float32

import numpy as np

from gps_visual.uturn_detector import UTurnDetector
from gps_visual.visual_odometry import MonocularWheelScaledVO


class VisualOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("visual_odometry_node")
        self._bridge = CvBridge()
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("fy", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("wheel_topic", "/vehicle/wheel_twist")
        self.declare_parameter("imu_topic", "/imu/data")
        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._vo = MonocularWheelScaledVO(K)
        self._uturn = UTurnDetector(angle_deg=150.0, window_sec=2.0)
        self._last_t = time.monotonic()
        self._wheel_m_s = 0.0
        self._gyro_z = 0.0
        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self._on_img, 10)
        self.create_subscription(TwistStamped, str(self.get_parameter("wheel_topic").value), self._on_wheel, 10)
        self.create_subscription(Imu, str(self.get_parameter("imu_topic").value), self._on_imu, 10)
        self._pub = self.create_publisher(PoseStamped, "/vo/pose", 10)
        self._pub_scale = self.create_publisher(Float32, "/vo/scale_hint", 10)
        self._pub_uturn = self.create_publisher(Bool, "/uturn_detected", 10)

    def _on_wheel(self, msg: TwistStamped) -> None:
        self._wheel_m_s = float(msg.twist.linear.x)

    def _on_imu(self, msg: Imu) -> None:
        self._gyro_z = float(msg.angular_velocity.z)

    def _on_img(self, msg: Image) -> None:
        now = time.monotonic()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._vo.set_wheel_speed(self._wheel_m_s)
        pose = self._vo.step(frame, dt)
        # optional yaw rate fusion hint (not deeply integrated — scale focus)
        _ = self._gyro_z
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = pose.x
        ps.pose.position.y = pose.y
        ps.pose.position.z = 0.0
        cy = np.cos(pose.theta * 0.5)
        sy = np.sin(pose.theta * 0.5)
        ps.pose.orientation.x = 0.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = float(sy)
        ps.pose.orientation.w = float(cy)
        self._pub.publish(ps)
        self._pub_scale.publish(Float32(data=float(self._wheel_m_s)))
        if self._uturn.update(pose.theta):
            u = Bool()
            u.data = True
            self._pub_uturn.publish(u)


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
