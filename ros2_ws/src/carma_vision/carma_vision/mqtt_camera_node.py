"""Turns JPEG frames from MQTT into ROS stereo image topics.

A test and bridging source: it lets a camera on another machine (a webcam on a
Windows PC via ``scripts/webcam_mqtt.py``, or an existing MQTT camera project)
drive the full pipeline as if it were the robot's stereo camera.

Reads ``<p>/source/frame`` (JPEG) and the retained ``<p>/source/meta``
(``{"layout": "sbs_lr" | "mono"}``). A side-by-side frame is split into left
and right; a mono frame is published to both eyes when ``mono_as_stereo`` is
true, so the stereo stream path can be exercised with one camera. Both images
carry the same timestamp, so downstream synchronisation is exact.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from carma_vision.mqtt_link import MqttLink


class MqttCameraNode(Node):
    """Republishes MQTT camera frames as ``sensor_msgs/Image``."""

    def __init__(self) -> None:
        super().__init__("carma_mqtt_camera")
        self.declare_parameter("mqtt_host", "broker")
        self.declare_parameter("mqtt_port", 1883)
        self.declare_parameter("mqtt_prefix", "carma")
        self.declare_parameter("left_topic", "/carma/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/carma/stereo/right/image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("mono_as_stereo", True)

        param = self.get_parameter
        prefix = param("mqtt_prefix").value
        self._frame_id = param("frame_id").value
        self._mono_as_stereo = bool(param("mono_as_stereo").value)
        self._layout = "mono"
        self._frames = 0

        self._bridge = CvBridge()
        self._left = self.create_publisher(Image, param("left_topic").value, qos_profile_sensor_data)
        self._right = self.create_publisher(
            Image, param("right_topic").value, qos_profile_sensor_data
        )

        self._mqtt = MqttLink(param("mqtt_host").value, param("mqtt_port").value, "carma-mqtt-cam")
        self._mqtt.subscribe(f"{prefix}/source/meta", self._on_meta, qos=1)
        self._mqtt.subscribe(f"{prefix}/source/frame", self._on_frame)
        self.create_timer(5.0, self._report)

    def _on_meta(self, _topic: str, payload: bytes) -> None:
        try:
            self._layout = json.loads(payload).get("layout", "mono")
        except (ValueError, AttributeError):
            self.get_logger().warning(f"ignoring malformed source meta: {payload[:80]!r}")

    def _on_frame(self, _topic: str, payload: bytes) -> None:
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warning("ignoring frame that is not a decodable JPEG")
            return
        if self._layout == "sbs_lr":
            half = img.shape[1] // 2
            left, right = img[:, :half], img[:, half : 2 * half]
        else:
            left, right = img, (img if self._mono_as_stereo else None)

        stamp = self.get_clock().now().to_msg()
        for pub, eye in ((self._left, left), (self._right, right)):
            if eye is None:
                continue
            msg = self._bridge.cv2_to_imgmsg(np.ascontiguousarray(eye), encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            pub.publish(msg)
        self._frames += 1

    def _report(self) -> None:
        state = "connected" if self._mqtt.connected else "waiting for broker"
        self.get_logger().info(f"{state}; {self._frames} frames ({self._layout}) in last 5 s")
        self._frames = 0

    def destroy_node(self) -> None:
        self._mqtt.close()
        super().destroy_node()


def main() -> None:
    """Entry point for ``ros2 run carma_vision mqtt_camera_node``."""
    rclpy.init()
    node = MqttCameraNode()
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
