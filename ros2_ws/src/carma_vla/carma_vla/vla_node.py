"""Live VLA pipeline: both stereo cameras and the operator's voice, into the policy server.

Subscribes
    ``/carma/stereo/{left,right}/image_raw``       world camera (platform view)
    ``/carma/arm/stereo/{left,right}/image_raw``   camera on the SO-101 arm
    ``/carma/operator/utterance``                  speech results, JSON (carma_voice)
    ``/carma/odom``                                pose, when something publishes it

Publishes
    ``/carma/vla/action``   JSON per answered request: action, uncertainty, which
                            views were sent, what answered, measured latency
    ``/carma/cmd_vel``      only with ``drive:=true``

At ``rate_hz`` it takes the newest stereo pair from each camera, the instruction
in force (see :mod:`carma_vla.pipeline` for how speech becomes one), and posts
one request to the policy server (``scripts/policy_server.py``) over the act
contract in :mod:`carma.backbone.wire`. One request is in flight at a time; a
tick that finds one still running is skipped rather than queued, so a slow
model lowers the rate instead of building a backlog of stale frames.

This is a live demonstration path, not an experimental condition: no arbiter
prices its actions, and nothing it does reaches the decision log. Driving is
off by default. With ``drive:=true``, a spoken stop, a policy ``stop``, or a
policy server that does not answer all command zero velocity: a broken link is
treated as "stop", never as "carry on with the last action".
"""

from __future__ import annotations

import json
import math
import threading
import time

import message_filters
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from carma.backbone.remote import HttpTransport
from carma.backbone.wire import ActRequest, ActResponse
from carma.types import BackboneError
from carma_vla.pipeline import (
    InstructionState,
    StereoPair,
    action_record,
    build_request,
    parse_utterance_message,
)

CAMERAS = ("world", "arm")


class VlaNode(Node):
    """Assembles observations and asks the policy server for actions."""

    def __init__(self) -> None:
        super().__init__("carma_vla")
        self.declare_parameter("policy_url", "http://host.docker.internal:8000/act")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("timeout_s", 5.0)
        self.declare_parameter("primary", "arm")
        self.declare_parameter("cameras", ["world", "arm"])
        self.declare_parameter("world_left_topic", "/carma/stereo/left/image_raw")
        self.declare_parameter("world_right_topic", "/carma/stereo/right/image_raw")
        self.declare_parameter("arm_left_topic", "/carma/arm/stereo/left/image_raw")
        self.declare_parameter("arm_right_topic", "/carma/arm/stereo/right/image_raw")
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("max_frame_age_s", 1.0)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("utterance_topic", "/carma/operator/utterance")
        self.declare_parameter("odom_topic", "/carma/odom")
        self.declare_parameter("instruction", "")
        self.declare_parameter("stage", "VEGETATIVE")
        self.declare_parameter("seed", 0)
        self.declare_parameter("samples", 4)
        self.declare_parameter("action_topic", "/carma/vla/action")
        self.declare_parameter("drive", False)
        self.declare_parameter("cmd_vel_topic", "/carma/cmd_vel")

        p = self.get_parameter
        self._primary = str(p("primary").value)
        self._cameras = [c for c in p("cameras").value if c in CAMERAS]
        if self._primary not in self._cameras:
            msg = f"primary camera {self._primary!r} is not among cameras {self._cameras}"
            raise ValueError(msg)
        self._timeout_s = float(p("timeout_s").value)
        self._max_age_s = float(p("max_frame_age_s").value)
        self._width = int(p("image_width").value)
        self._stage = str(p("stage").value)
        self._seed = int(p("seed").value)
        self._samples = int(p("samples").value)
        self._drive = bool(p("drive").value)
        self._url = str(p("policy_url").value)
        self._transport = HttpTransport(self._url)

        self._state = InstructionState(instruction=str(p("instruction").value).strip())
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[StereoPair, float]] = {}
        self._pose = (0.0, 0.0, 0.0)
        self._start = time.monotonic()
        self._step = 0
        self._busy = False
        self._answered = 0
        self._skipped = ""
        self._last_note = ""

        slop = float(p("sync_slop_s").value)
        self._syncs = []
        for cam in self._cameras:
            left = message_filters.Subscriber(
                self, Image, p(f"{cam}_left_topic").value, qos_profile=qos_profile_sensor_data
            )
            right = message_filters.Subscriber(
                self, Image, p(f"{cam}_right_topic").value, qos_profile=qos_profile_sensor_data
            )
            sync = message_filters.ApproximateTimeSynchronizer([left, right], 5, slop)
            sync.registerCallback(lambda lm, rm, cam=cam: self._on_pair(cam, lm, rm))
            self._syncs.append(sync)

        self.create_subscription(String, p("utterance_topic").value, self._on_utterance, 10)
        self.create_subscription(Odometry, p("odom_topic").value, self._on_odom, 10)
        self._actions = self.create_publisher(String, p("action_topic").value, 10)
        self._cmd = self.create_publisher(Twist, p("cmd_vel_topic").value, 10) if self._drive else None

        self.create_timer(1.0 / float(p("rate_hz").value), self._tick)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"cameras {self._cameras}, primary {self._primary}, policy {self._url}, "
            f"drive {'ON' if self._drive else 'off'}; "
            f"instruction {self._state.instruction or '(none yet - say one)'}"
        )

    # --- inputs -------------------------------------------------------------
    def _on_pair(self, cam: str, left: Image, right: Image) -> None:
        pair = (
            self._bridge.imgmsg_to_cv2(left, desired_encoding="bgr8"),
            self._bridge.imgmsg_to_cv2(right, desired_encoding="bgr8"),
        )
        with self._lock:
            self._frames[cam] = (pair, time.monotonic())

    def _on_utterance(self, msg: String) -> None:
        utterance = parse_utterance_message(msg.data)
        if utterance is None:
            self.get_logger().warning(f"ignoring utterance that is not JSON: {msg.data[:80]!r}")
            return
        with self._lock:
            note = self._state.apply(utterance)
            paused = self._state.paused
        if note:
            self.get_logger().info(f"voice: {note}")
            self._last_note = note
            if paused:
                self._halt()

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    # --- the loop -----------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._busy:
                self._skipped = "previous request still running"
                return
            if not self._state.ready:
                self._skipped = "paused" if self._state.paused else "no instruction yet"
                return
            fresh = {
                cam: pair for cam, (pair, seen) in self._frames.items() if now - seen <= self._max_age_s
            }
            if self._primary not in fresh:
                self._skipped = f"no fresh frame from the {self._primary} camera"
                return
            instruction = self._state.instruction
            self._busy = True
            step = self._step
            self._step += 1
        request = build_request(
            fresh,
            self._primary,
            instruction,
            pose=self._pose,
            t=round(now - self._start, 3),
            stage=self._stage,
            seed=self._seed + step,
            samples=self._samples,
            width=self._width,
        )
        threading.Thread(target=self._ask, args=(request,), daemon=True).start()

    def _ask(self, request: ActRequest) -> None:
        started = time.monotonic()
        try:
            reply = self._transport.post(request.to_json(), self._timeout_s)
            response = ActResponse.from_json(reply)
        except (BackboneError, ValueError) as exc:
            self.get_logger().warning(f"policy did not answer: {exc}", throttle_duration_sec=5.0)
            self._halt()
            with self._lock:
                self._busy = False
            return
        record = action_record(response, request, self._primary, time.monotonic() - started)
        self._actions.publish(String(data=json.dumps(record, sort_keys=True)))
        with self._lock:
            paused = self._state.paused
            self._answered += 1
            self._busy = False
        if self._cmd is not None:
            if paused or response.stop:
                self._halt()
            else:
                twist = Twist()
                twist.linear.x = response.linear
                twist.angular.z = response.angular
                self._cmd.publish(twist)

    def _halt(self) -> None:
        if self._cmd is not None:
            self._cmd.publish(Twist())

    def _report(self) -> None:
        with self._lock:
            ages = {
                cam: round(time.monotonic() - seen, 1) for cam, (_, seen) in self._frames.items()
            }
            answered, self._answered = self._answered, 0
            skipped = self._skipped
        self.get_logger().info(
            f"{answered} actions in last 5 s; frame age s {ages or '(no frames)'}"
            + (f"; waiting: {skipped}" if answered == 0 and skipped else "")
        )

    def destroy_node(self) -> None:
        self._halt()
        super().destroy_node()


def main() -> None:
    """Entry point for ``ros2 run carma_vla vla_node``."""
    rclpy.init()
    node = VlaNode()
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
