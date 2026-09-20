#!/usr/bin/env python3
"""Republish the simulator's ``rgb8`` stereo images as ``bgr8``.

Why this node exists. The bridge contract
(``ros2_ws/src/carma_bridge/README.md``) says the stereo topics carry ``bgr8``,
and AGENTS.md section 7 fixes BGR as the project's image convention, because
that is what OpenCV expects and what the real camera driver publishes. Gazebo
cannot produce it: ``<format>B8G8R8</format>`` is accepted by SDFormat but the
ogre2 render path rejects it at run time with ``Unsupported pixel format [8]``,
so the camera silently falls back to RGB. ``ros_gz_bridge`` passes the pixel
format straight through, which would put ``rgb8`` on a topic the contract
declares as ``bgr8`` — a channel swap that looks like nothing at all until a
colour-sensitive analyzer quietly reports the wrong thing.

So the simulator publishes on ``.../image_rgb`` and this node does the swap
onto the contract's ``.../image_raw``. It exists only in simulation; on the
robot the driver publishes ``bgr8`` directly and nothing here runs.

The swap is a reversed view of the channel axis, one copy per frame. At
640x480x3 and 15 Hz that is about 14 MB/s per camera, which is well inside the
budget the DDS transport profile in ``docker/fastdds_shm.xml`` is tuned for.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Kept identical to the incoming queue depth used by ros_gz_bridge, so that a
# slow consumer drops frames in one place rather than two.
QUEUE_DEPTH = 10


class BgrRelay(Node):
    """Swaps the channel order of every image on one topic onto another."""

    def __init__(self) -> None:
        super().__init__("carma_bgr_relay")
        self.declare_parameter(
            "pairs",
            [
                "/carma/stereo/left/image_rgb:/carma/stereo/left/image_raw",
                "/carma/stereo/right/image_rgb:/carma/stereo/right/image_raw",
            ],
        )
        # Not self._publishers: rclpy.Node uses that name for its own list of
        # publishers, and shadowing it breaks create_publisher.
        self._outputs: dict[str, object] = {}
        for pair in self.get_parameter("pairs").get_parameter_value().string_array_value:
            source, _, target = pair.partition(":")
            if not source or not target:
                raise ValueError(f"pairs entry {pair!r} is not '<source>:<target>'")
            publisher = self.create_publisher(Image, target, QUEUE_DEPTH)
            self._outputs[source] = publisher
            self.create_subscription(
                Image,
                source,
                lambda msg, pub=publisher: self._relay(msg, pub),
                QUEUE_DEPTH,
            )
            self.get_logger().info(f"relaying {source} (rgb8) -> {target} (bgr8)")

    def _relay(self, msg: Image, publisher) -> None:  # type: ignore[no-untyped-def]
        if msg.encoding == "bgr8":
            # Already correct: pass it through rather than swapping it wrong.
            publisher.publish(msg)
            return
        if msg.encoding != "rgb8":
            # Loud, once per frame is noisy, so throttle: an unexpected encoding
            # here means the world file changed and this node is now lying.
            self.get_logger().error(
                f"expected rgb8 or bgr8, got {msg.encoding!r}; dropping frame",
                throttle_duration_sec=5.0,
            )
            return
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        msg.data = np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()
        msg.encoding = "bgr8"
        publisher.publish(msg)


def main() -> None:
    """Spin the relay until shutdown."""
    rclpy.init()
    node = BgrRelay()
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
