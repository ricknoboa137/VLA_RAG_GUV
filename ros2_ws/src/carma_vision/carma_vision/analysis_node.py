"""Runs carma scene analyzers on the live camera and publishes the results.

Which analyzers run is decided by an analyzer config file
(``configs/analyzers/*.yaml``), built through ``carma.config.build_analyzers``,
so adding your own model never touches this node.

Results go to ``/carma/analysis`` as JSON strings and, when ``mqtt_host`` is
set, to ``<mqtt_prefix>/analysis/<analyzer>`` on the broker.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from carma.config import build_analyzers, load_analyzers
from carma.types import CarmaError
from carma.types.core import Observation, PhenologyStage, Pose
from carma_vision.mqtt_link import MqttLink


class AnalysisNode(Node):
    """Analyses the newest left-camera frame at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("carma_analysis")
        self.declare_parameter("analyzers_config", "/opt/carma/configs/analyzers/field.yaml")
        self.declare_parameter("image_topic", "/carma/stereo/left/image_raw")
        self.declare_parameter("odom_topic", "/carma/odom")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("stage", "VEGETATIVE")
        self.declare_parameter("mqtt_host", "")
        self.declare_parameter("mqtt_port", 1883)
        self.declare_parameter("mqtt_prefix", "carma")

        param = self.get_parameter
        config_path = Path(param("analyzers_config").value)
        self._analyzers = build_analyzers(load_analyzers(config_path))
        self._stage = PhenologyStage[param("stage").value]
        self._prefix = param("mqtt_prefix").value
        self.get_logger().info(
            f"analyzers from {config_path}: {[a.name for a in self._analyzers]}"
        )

        host = param("mqtt_host").value
        self._mqtt = MqttLink(host, param("mqtt_port").value, "carma-analysis") if host else None

        self._bridge = CvBridge()
        self._latest: Image | None = None
        self._pose = Pose(0.0, 0.0, 0.0)
        self._t0: float | None = None

        self.create_subscription(
            Image, param("image_topic").value, self._on_image, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, param("odom_topic").value, self._on_odom, 10)
        self._pub = self.create_publisher(String, "/carma/analysis", 10)
        self.create_timer(1.0 / param("rate_hz").value, self._tick)

    def _on_image(self, msg: Image) -> None:
        self._latest = msg

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = Pose(p.x, p.y, yaw)

    def _tick(self) -> None:
        frame, self._latest = self._latest, None
        if frame is None:
            return
        stamp = frame.header.stamp.sec + frame.header.stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = stamp
        obs = Observation(
            rgb=self._bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8"),
            depth=None,
            pose=self._pose,
            t=stamp - self._t0,
            instruction="",
            stage=self._stage,
        )
        for analyzer in self._analyzers:
            try:
                result = analyzer.analyze(obs)
            except CarmaError as exc:
                self.get_logger().warning(f"{analyzer.name} failed: {exc}")
                continue
            payload = json.dumps({"stamp": stamp, **asdict(result)}, sort_keys=True, default=str)
            self._pub.publish(String(data=payload))
            if self._mqtt is not None:
                self._mqtt.publish(f"{self._prefix}/analysis/{result.analyzer}", payload, qos=1)

    def destroy_node(self) -> None:
        if self._mqtt is not None:
            self._mqtt.close()
        super().destroy_node()


def main() -> None:
    """Entry point for ``ros2 run carma_vision analysis_node``."""
    rclpy.init()
    node = AnalysisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
