"""Streams the stereo camera over MQTT as side-by-side JPEG frames.

Good for monitoring from a laptop, a dashboard or a phone. For viewing on a VR
headset with head motion, see ``docs/streaming.md``: MQTT over TCP adds latency
and has no rate adaptation, and WebRTC is the planned transport for that path.

Topics (``<p>`` is ``mqtt_prefix``):
    ``<p>/camera/frame``  JPEG bytes, QoS 0, newest frame wins
    ``<p>/camera/meta``   retained JSON: layout, width, height, rate
"""

from __future__ import annotations

import json

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from carma_vision.mqtt_link import MqttLink
from carma_vision.stereo import encode_jpeg, fit_width, side_by_side


class StreamNode(Node):
    """Publishes the newest (synchronised) frame at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("carma_stream")
        self.declare_parameter("left_topic", "/carma/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/carma/stereo/right/image_raw")
        self.declare_parameter("stereo", True)
        self.declare_parameter("sync_slop_s", 0.02)
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("jpeg_quality", 70)
        self.declare_parameter("max_width", 2560)
        self.declare_parameter("mqtt_host", "broker")
        self.declare_parameter("mqtt_port", 1883)
        self.declare_parameter("mqtt_prefix", "carma")

        param = self.get_parameter
        self._stereo = bool(param("stereo").value)
        self._quality = int(param("jpeg_quality").value)
        self._max_width = int(param("max_width").value)
        self._rate_hz = float(param("rate_hz").value)
        prefix = param("mqtt_prefix").value
        self._frame_topic = f"{prefix}/camera/frame"
        self._meta_topic = f"{prefix}/camera/meta"
        self._mqtt = MqttLink(param("mqtt_host").value, param("mqtt_port").value, "carma-stream")

        self._bridge = CvBridge()
        self._pair: tuple[Image, Image | None] | None = None
        self._meta_sent: tuple[int, int] | None = None

        if self._stereo:
            left = message_filters.Subscriber(
                self, Image, param("left_topic").value, qos_profile=qos_profile_sensor_data
            )
            right = message_filters.Subscriber(
                self, Image, param("right_topic").value, qos_profile=qos_profile_sensor_data
            )
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [left, right], queue_size=5, slop=float(param("sync_slop_s").value)
            )
            self._sync.registerCallback(self._on_pair)
        else:
            self.create_subscription(
                Image, param("left_topic").value, self._on_mono, qos_profile_sensor_data
            )
        self.create_timer(1.0 / self._rate_hz, self._tick)

    def _on_pair(self, left: Image, right: Image) -> None:
        self._pair = (left, right)

    def _on_mono(self, left: Image) -> None:
        self._pair = (left, None)

    def _tick(self) -> None:
        pair, self._pair = self._pair, None
        if pair is None or not self._mqtt.connected:
            return
        left = self._bridge.imgmsg_to_cv2(pair[0], desired_encoding="bgr8")
        if pair[1] is not None:
            frame = side_by_side(left, self._bridge.imgmsg_to_cv2(pair[1], desired_encoding="bgr8"))
        else:
            frame = left
        frame = fit_width(frame, self._max_width)

        size = (frame.shape[1], frame.shape[0])
        if size != self._meta_sent:
            meta = {
                "layout": "sbs_lr" if pair[1] is not None else "mono",
                "encoding": "jpeg",
                "width": size[0],
                "height": size[1],
                "rate_hz": self._rate_hz,
            }
            self._mqtt.publish(self._meta_topic, json.dumps(meta), qos=1, retain=True)
            self._meta_sent = size
        self._mqtt.publish(self._frame_topic, encode_jpeg(frame, self._quality), qos=0)

    def destroy_node(self) -> None:
        self._mqtt.close()
        super().destroy_node()


def main() -> None:
    """Entry point for ``ros2 run carma_vision stream_node``."""
    rclpy.init()
    node = StreamNode()
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
