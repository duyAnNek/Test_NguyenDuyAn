"""LandmarkDatabase over ROS 2 — JSON on std_msgs/String (no custom .srv / MSVC build)."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from gps_visual_b.landmark_database import LandmarkDatabase


class LandmarkDbNode(Node):
    """
    Topics (defaults, override via parameters):
      ~/landmark/add       (String JSON) -> ~/landmark/add_result
      ~/landmark/query     (String JSON) -> ~/landmark/query_result

    Add JSON: {"class_name":"pothole","x":0,"y":0,"z":0,"descriptor":[float,...]}
    Add result: {"success":true,"landmark_id":"..."} or {"success":false,"error":"..."}

    Query JSON: {"descriptor":[...],"top_k":5,"radius_m":50.0,"query_x":0.0,"query_y":0.0}
    Query result: {"landmark_ids":[],"class_names":[],"scores":[],"x":[],"y":[],"z":[]}
    """

    def __init__(self) -> None:
        super().__init__("landmark_db_node")
        self.declare_parameter("sqlite_path", "/tmp/gps_visual_b_landmarks.sqlite")
        self.declare_parameter("topic_add", "/landmark/add")
        self.declare_parameter("topic_add_result", "/landmark/add_result")
        self.declare_parameter("topic_query", "/landmark/query")
        self.declare_parameter("topic_query_result", "/landmark/query_result")
        p = pathlib.Path(str(self.get_parameter("sqlite_path").value))
        p.parent.mkdir(parents=True, exist_ok=True)
        self._db = LandmarkDatabase(sqlite_path=str(p))
        t_add = str(self.get_parameter("topic_add").value)
        t_ar = str(self.get_parameter("topic_add_result").value)
        t_q = str(self.get_parameter("topic_query").value)
        t_qr = str(self.get_parameter("topic_query_result").value)
        self._pub_add = self.create_publisher(String, t_ar, 10)
        self._pub_q = self.create_publisher(String, t_qr, 10)
        self.create_subscription(String, t_add, self._on_add, 10)
        self.create_subscription(String, t_q, self._on_query, 10)
        self.get_logger().info("landmark_db_node SQLite=%s topics add=%s query=%s" % (p, t_add, t_q))

    def _on_add(self, msg: String) -> None:
        out = String()
        try:
            o: Dict[str, Any] = json.loads(msg.data)
            desc = np.asarray(o.get("descriptor", []), dtype=np.float32)
            lid = self._db.add_landmark(
                str(o["class_name"]),
                (float(o["x"]), float(o["y"]), float(o["z"])),
                desc,
                merge_if_close=bool(o.get("merge_if_close", True)),
            )
            out.data = json.dumps({"success": True, "landmark_id": lid})
        except Exception as e:
            out.data = json.dumps({"success": False, "error": str(e)})
        self._pub_add.publish(out)

    def _on_query(self, msg: String) -> None:
        out = String()
        try:
            o = json.loads(msg.data)
            desc = np.asarray(o.get("descriptor", []), dtype=np.float32)
            hits = self._db.query_landmark(
                desc,
                top_k=max(1, int(o.get("top_k", 5))),
                radius_m=float(o.get("radius_m", 50.0)),
                query_xy=(float(o.get("query_x", 0.0)), float(o.get("query_y", 0.0))),
            )
            payload = {
                "landmark_ids": [h[0].id for h in hits],
                "class_names": [h[0].class_name for h in hits],
                "scores": [float(h[1]) for h in hits],
                "x": [float(h[0].p3d[0]) for h in hits],
                "y": [float(h[0].p3d[1]) for h in hits],
                "z": [float(h[0].p3d[2]) for h in hits],
            }
            out.data = json.dumps(payload)
        except Exception as e:
            out.data = json.dumps({"error": str(e)})
        self._pub_q.publish(out)


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = LandmarkDbNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
