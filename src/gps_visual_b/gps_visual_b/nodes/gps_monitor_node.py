"""GPS integrity FSM + latched last-good NavSatFix."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float32MultiArray, UInt8

from gps_visual_b.gps_integrity import GPSIntegrityMonitor, GpsIntegrityState, GpsQualitySample


class GpsMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("gps_monitor_node")
        self._mon = GPSIntegrityMonitor()
        self._hdop = 1.0
        self._sats = 12
        self._snr = 40.0
        self._metrics_timeout_sec = 2.0
        self._last_metrics_t = -1e30
        self.declare_parameter("fix_topic", "/gps/fix")
        self.declare_parameter("metrics_topic", "/gps/metrics")
        self.create_subscription(NavSatFix, str(self.get_parameter("fix_topic").value), self._on_fix, 10)
        self.create_subscription(Float32MultiArray, str(self.get_parameter("metrics_topic").value), self._on_metrics, 10)
        self._pub_state = self.create_publisher(UInt8, "/gps/integrity_state", 10)
        self._pub_latched = self.create_publisher(NavSatFix, "/gps/last_good_fix", 10)

    def _on_metrics(self, msg: Float32MultiArray) -> None:
        d = msg.data
        if len(d) >= 1:
            self._hdop = float(max(0.5, d[0]))
        if len(d) >= 2:
            self._sats = int(max(0, d[1]))
        if len(d) >= 3:
            self._snr = float(d[2])
        self._last_metrics_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_fix(self, msg: NavSatFix) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_metrics_t > self._metrics_timeout_sec:
            # Heuristic from covariance if no /gps/metrics
            c = msg.position_covariance[0]
            if c > 0 and c < 1e6:
                self._hdop = float(math.sqrt(max(c, 1.0)))
            else:
                self._hdop = 2.0 if msg.status.status >= NavSatStatus.STATUS_FIX else 99.0
            self._sats = 8 if msg.status.status >= NavSatStatus.STATUS_FIX else 0
        fix_ok = msg.status.status >= NavSatStatus.STATUS_FIX
        q = GpsQualitySample(
            hdop=self._hdop,
            n_satellites=self._sats,
            mean_snr_dbhz=self._snr,
            fix_ok=fix_ok,
            stamp_sec=now,
        )
        st = self._mon.update(q, msg.latitude, msg.longitude, msg.altitude)
        smsg = UInt8()
        smsg.data = int(st)
        self._pub_state.publish(smsg)
        if self._mon.latched.valid:
            lf = NavSatFix()
            lf.header = msg.header
            lf.latitude = self._mon.latched.lat
            lf.longitude = self._mon.latched.lon
            lf.altitude = self._mon.latched.alt
            lf.status.status = NavSatStatus.STATUS_FIX
            lf.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            self._pub_latched.publish(lf)


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = GpsMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
