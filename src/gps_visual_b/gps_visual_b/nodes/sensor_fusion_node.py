"""EKF fusion: GPS + VO + lane; publishes /odometry and /pose_stamped."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Optional, Tuple

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix, NavSatStatus
from std_msgs.msg import String, UInt8

from gps_visual_b.ekf_fusion import PoseEKF
from gps_visual_b.geo_utils import enu_from_latlon
from gps_visual_b.ghost_matching import bbox_iou, project_xy_to_uv
from gps_visual_b.gps_integrity import GpsIntegrityState
from gps_visual_b.landmark_database import LandmarkDatabase
from gps_visual_b.yolo_onnx import YoloOnnxDetector


def _quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(math.sin(yaw * 0.5))
    q.w = float(math.cos(yaw * 0.5))
    return q


class SensorFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("sensor_fusion_node")
        self._ekf = PoseEKF()
        self._origin: Optional[Tuple[float, float]] = None
        self._integrity = GpsIntegrityState.GPS_GOOD
        self._prev_integrity: Optional[GpsIntegrityState] = None
        self._prev_vo: Optional[Tuple[float, float, float]] = None
        self._lane = 0
        self._last_gps_xy: Optional[Tuple[float, float]] = None
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("gps_fix_topic", "/gps/fix")
        self.declare_parameter("integrity_topic", "/gps/integrity_state")
        self.declare_parameter("vo_topic", "/vo/pose")
        self.declare_parameter("lane_topic", "/lane_position")
        self.declare_parameter("landmark_obs_topic", "/landmark/observations")
        self.declare_parameter("enable_ghost_matching", False)
        self.declare_parameter("ghost_image_topic", "/camera/image_raw")
        self.declare_parameter("landmark_sqlite", "/tmp/gps_visual_b_landmarks.sqlite")
        self.declare_parameter("yolo_onnx_path", "")
        self.declare_parameter("camera_fx", 900.0)
        self.declare_parameter("camera_fy", 900.0)
        self.declare_parameter("camera_cx", 640.0)
        self.declare_parameter("camera_cy", 360.0)
        self._rate = float(self.get_parameter("publish_rate_hz").value)
        self._bridge = CvBridge()
        self._ghost_on = bool(self.get_parameter("enable_ghost_matching").value)
        self._landmarks = LandmarkDatabase(sqlite_path=str(self.get_parameter("landmark_sqlite").value))
        self._yolo: Optional[YoloOnnxDetector] = None
        yolo_path = str(self.get_parameter("yolo_onnx_path").value).strip()
        if self._ghost_on and yolo_path and pathlib.Path(yolo_path).is_file():
            try:
                self._yolo = YoloOnnxDetector(yolo_path)
            except Exception as e:
                self.get_logger().warn("ghost YOLO load failed: %s" % e)
        if self._ghost_on and self._yolo is not None:
            self.create_subscription(
                Image, str(self.get_parameter("ghost_image_topic").value), self._on_image_ghost, 2
            )

        self.create_subscription(NavSatFix, str(self.get_parameter("gps_fix_topic").value), self._on_gps, 10)
        self.create_subscription(UInt8, str(self.get_parameter("integrity_topic").value), self._on_integ, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("vo_topic").value), self._on_vo, 10)
        self.create_subscription(UInt8, str(self.get_parameter("lane_topic").value), self._on_lane, 10)
        self.create_subscription(String, str(self.get_parameter("landmark_obs_topic").value), self._on_landmark_obs, 10)
        self._pub_odom = self.create_publisher(Odometry, "/odometry", 10)
        self._pub_pose = self.create_publisher(PoseStamped, "/pose_stamped", 10)
        self._last_img_header = None
        self._fx = float(self.get_parameter("camera_fx").value)
        self._fy = float(self.get_parameter("camera_fy").value)
        self._cx = float(self.get_parameter("camera_cx").value)
        self._cy = float(self.get_parameter("camera_cy").value)
        self.create_timer(1.0 / max(1.0, self._rate), self._tick)

    def _on_lane(self, msg: UInt8) -> None:
        self._lane = int(msg.data)

    def _on_image_ghost(self, msg: Image) -> None:
        if self._yolo is None:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        h, w = frame.shape[:2]
        dets = self._yolo.infer(frame)
        pose_xytheta = (self._ekf.s.x, self._ekf.s.y, self._ekf.s.theta)
        best_iou = 0.0
        best_lm = None
        for lm in self._landmarks.all_landmarks():
            u, v = project_xy_to_uv(
                float(lm.p3d[0]), float(lm.p3d[1]), pose_xytheta, self._fx, self._fy, self._cx, self._cy
            )
            if not (0 <= u < w and 0 <= v < h):
                continue
            half = 40
            gx = (int(u - half), int(v - half), int(u + half), int(v + half))
            for d in dets:
                iou = bbox_iou(gx, d.xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_lm = lm
        if best_lm is not None and best_iou >= 0.12:
            self._ekf.update_landmark_xy(float(best_lm.p3d[0]), float(best_lm.p3d[1]))

    def _on_landmark_obs(self, msg: String) -> None:
        """JSON: {\"x\":..,\"y\":..} map-frame landmark anchor from upstream ghost matcher."""
        try:
            o = json.loads(msg.data)
            lx = float(o.get("x", 0.0))
            ly = float(o.get("y", 0.0))
            if float(o.get("weight", 1.0)) > 0:
                self._ekf.update_landmark_xy(lx, ly)
        except Exception:
            return

    def _on_integ(self, msg: UInt8) -> None:
        new_s = GpsIntegrityState(int(msg.data))
        if self._prev_integrity == GpsIntegrityState.GPS_LOST and new_s == GpsIntegrityState.GPS_GOOD:
            if self._last_gps_xy is not None:
                gx, gy = self._last_gps_xy
                self._ekf.update_gps(gx, gy, None)
                self._prev_vo = None
        self._prev_integrity = new_s
        self._integrity = new_s

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return
        if self._origin is None:
            self._origin = (float(msg.latitude), float(msg.longitude))
        assert self._origin is not None
        ex, ny = enu_from_latlon(self._origin[0], self._origin[1], float(msg.latitude), float(msg.longitude))
        self._last_gps_xy = (ex, ny)
        if self._integrity != GpsIntegrityState.GPS_GOOD:
            return
        self._ekf.update_gps(ex, ny, None)

    def _yaw_from_pose(self, ps: PoseStamped) -> float:
        q = ps.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    def _on_vo(self, msg: PoseStamped) -> None:
        self._last_img_header = msg.header
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        th = self._yaw_from_pose(msg)
        if self._prev_vo is None:
            self._prev_vo = (x, y, th)
            return
        px, py, pth = self._prev_vo
        dx = x - px
        dy = y - py
        dth = math.atan2(math.sin(th - pth), math.cos(th - pth))
        self._ekf.predict_map_delta(dx, dy, dth)
        self._prev_vo = (x, y, th)
        self._lane_fuse()

    def _lane_fuse(self) -> None:
        if self._lane != 0:
            self._ekf.soft_lane_update(self._lane, strength=0.06)

    def _tick(self) -> None:
        s = self._ekf.s
        pose = PoseStamped()
        if self._last_img_header is not None:
            pose.header = self._last_img_header
        else:
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
        pose.pose.position.x = s.x
        pose.pose.position.y = s.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = _quat_from_yaw(s.theta)
        self._pub_pose.publish(pose)
        o = Odometry()
        o.header = pose.header
        o.pose.pose = pose.pose
        o.twist.twist = Twist()
        self._pub_odom.publish(o)


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
