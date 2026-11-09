"""Bridge node.

Owns every ROS import in the project. It serves the ``carma`` library over a
TCP socket (format: ``protocol.py``) and advances the simulation in lockstep:

* ``reset``: stop the robot, reset the world, take one settling step, return
  the first observation.
* ``step``: publish one velocity command, advance the paused world by exactly
  ``round(dt_s / physics_step_s)`` iterations, return the observation.

Nothing here reads the wall clock to decide what a step contained. The world
must be started paused (``lockstep:=true`` on ``row_crop.launch.py``), which is
what lets the same seed replay the same trajectory. Sensor data is matched by
the header stamps Gazebo wrote, not by arrival time.

The image triplet (left, right, depth) is synchronised with an approximate-time
policy whose slop is a parameter; the pose comes from the odometry message at
the end of the step, so it is exact for the returned ``t`` rather than up to a
camera period old.
"""

from __future__ import annotations

import base64
import math
import socket
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

from carma_bridge.protocol import recv_frame, send_frame

_EPS_S = 1e-6
# Messages kept per sensor. Only the newest one at or before the step target is used.
_HISTORY = 8


def _stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class BridgeError(RuntimeError):
    """A request could not be carried out; the message goes back to the library."""


class BridgeNode(Node):
    """Serves lockstep reset/step requests over TCP."""

    def __init__(self) -> None:
        super().__init__("carma_bridge")
        self.declare_parameter("bridge_port", 5555)
        self.declare_parameter("world", "row_crop")
        self.declare_parameter("robot", "carma_ugv")
        self.declare_parameter("spawn_z_m", 0.20)
        self.declare_parameter("physics_step_s", 0.001)
        self.declare_parameter("camera_hz", 15.0)
        self.declare_parameter("quiet_s", 0.15)
        self.declare_parameter("settle_iterations", 2000)
        self.declare_parameter("sync_slop_s", 0.02)
        self.declare_parameter("wait_timeout_s", 60.0)
        self.declare_parameter("command_settle_s", 0.05)

        self._port = int(self.get_parameter("bridge_port").value)
        world = str(self.get_parameter("world").value)
        self._physics_step_s = float(self.get_parameter("physics_step_s").value)
        self._camera_period_s = 1.0 / float(self.get_parameter("camera_hz").value)
        self._quiet_s = float(self.get_parameter("quiet_s").value)
        self._settle_iterations = int(self.get_parameter("settle_iterations").value)
        self._slop_s = float(self.get_parameter("sync_slop_s").value)
        self._timeout_s = float(self.get_parameter("wait_timeout_s").value)
        self._robot = str(self.get_parameter("robot").value)
        self._spawn_z_m = float(self.get_parameter("spawn_z_m").value)
        self._settle_s = float(self.get_parameter("command_settle_s").value)

        self._cv = CvBridge()
        self._cmd_pub = self.create_publisher(Twist, "/carma/cmd_vel", 10)
        self._control = self.create_client(ControlWorld, f"/world/{world}/control")
        self._set_pose = self.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self._start = (0.0, 0.0, 0.0)  # x, y, yaw the episode started at; odometry is relative to it

        self._lock = threading.Condition()
        self._bundles: list[dict[str, Any]] = []  # recent synced image triplets, oldest first
        self._odoms: list[dict[str, float]] = []  # recent odometry, oldest first
        self._clock_s = -1.0  # newest simulation time published on /clock
        self._last_sensor_wall_s = time.monotonic()  # when a sensor message last arrived
        self._base_s = 0.0  # sim time that corresponds to t = 0
        self._now_s = 0.0  # sim time the world was last advanced to

        left = Subscriber(self, Image, "/carma/stereo/left/image_raw")
        right = Subscriber(self, Image, "/carma/stereo/right/image_raw")
        depth = Subscriber(self, Image, "/carma/camera/depth")
        self._sync = ApproximateTimeSynchronizer([left, right, depth], queue_size=10, slop=self._slop_s)
        self._sync.registerCallback(self._on_images)
        self.create_subscription(Odometry, "/carma/odom", self._on_odom, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, 100)

        self._running = True
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(
            f"bridge on port {self._port}: world={world}, physics step {self._physics_step_s} s, "
            f"image sync slop {self._slop_s} s"
        )

    # ---- ROS callbacks (executor thread) --------------------------------------------------

    def _on_images(self, left: Image, right: Image, depth: Image) -> None:
        bundle = {
            "t": _stamp_s(left.header.stamp),
            "left": self._cv.imgmsg_to_cv2(left, desired_encoding="bgr8"),
            "depth": self._cv.imgmsg_to_cv2(depth, desired_encoding="32FC1"),
        }
        with self._lock:
            self._bundles = (self._bundles + [bundle])[-_HISTORY:]
            self._last_sensor_wall_s = time.monotonic()
            self._lock.notify_all()

    def _on_clock(self, msg: Clock) -> None:
        with self._lock:
            self._clock_s = _stamp_s(msg.clock)
            self._lock.notify_all()

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        odom = {
            "t": _stamp_s(msg.header.stamp),
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": yaw,
        }
        with self._lock:
            self._odoms = (self._odoms + [odom])[-_HISTORY:]
            self._last_sensor_wall_s = time.monotonic()
            self._lock.notify_all()

    # ---- requests (server thread) ---------------------------------------------------------

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self._port))
        server.listen(1)
        server.settimeout(1.0)
        while self._running:
            try:
                conn, peer = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self.get_logger().info(f"client connected: {peer}")
            with conn:
                self._handle(conn)
            self.get_logger().info("client disconnected")
        server.close()

    def _handle(self, conn: socket.socket) -> None:
        while self._running:
            try:
                request = recv_frame(conn)
            except (ConnectionError, OSError, ValueError):
                return
            op = request.get("op")
            if op == "close":
                self._publish(0.0, 0.0)
                return
            try:
                if op == "reset":
                    reply = self._reset(request.get("start"))
                elif op == "step":
                    reply = self._step(request)
                else:
                    raise BridgeError(f"unknown op {op!r}")
            except BridgeError as error:
                reply = {"ok": False, "error": str(error)}
            try:
                send_frame(conn, reply)
            except OSError:
                return

    def _publish(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)

    def _call_control(self, **fields: Any) -> None:
        """Call the world-control service and wait for its answer."""
        if not self._control.wait_for_service(timeout_sec=self._timeout_s):
            raise BridgeError("the world control service is not available; was the world launched?")
        request = ControlWorld.Request()
        for name, value in fields.items():
            if name == "reset_all":
                request.world_control.reset.all = value
            else:
                setattr(request.world_control, name, value)
        done = threading.Event()
        future = self._control.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout_s):
            raise BridgeError("the world control service did not answer in time")
        result = future.result()
        if result is None or not result.success:
            raise BridgeError(f"the world control service refused {fields}")

    def _world_pose(self, odom: dict[str, float]) -> dict[str, float]:
        """Odometry (zero at the episode start) expressed in the world frame."""
        sx, sy, syaw = self._start
        cos, sin = math.cos(syaw), math.sin(syaw)
        yaw = math.atan2(math.sin(odom["yaw"] + syaw), math.cos(odom["yaw"] + syaw))
        return {
            "x": sx + cos * odom["x"] - sin * odom["y"],
            "y": sy + sin * odom["x"] + cos * odom["y"],
            "yaw": yaw,
        }

    def _place_robot(self, x: float, y: float, yaw: float) -> None:
        """Move the robot to a world pose with the world paused."""
        if not self._set_pose.wait_for_service(timeout_sec=self._timeout_s):
            raise BridgeError("the set_pose service is not available; was the world launched?")
        request = SetEntityPose.Request()
        request.entity.name = self._robot
        request.entity.type = Entity.MODEL
        request.pose.position.x, request.pose.position.y = x, y
        request.pose.position.z = self._spawn_z_m
        request.pose.orientation.z = math.sin(yaw / 2.0)
        request.pose.orientation.w = math.cos(yaw / 2.0)
        done = threading.Event()
        future = self._set_pose.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self._timeout_s):
            raise BridgeError("the set_pose service did not answer in time")
        result = future.result()
        if result is None or not result.success:
            raise BridgeError(f"could not place {self._robot!r} at ({x}, {y}, {yaw})")

    def _advance(self, iterations: int) -> float:
        """Advance the paused world by ``iterations`` and return the sim time reached."""
        self._call_control(pause=True, multi_step=iterations)
        self._now_s += iterations * self._physics_step_s
        return self._now_s

    def _wait_for(self, target_s: float) -> dict[str, Any]:
        """Wait for the world to reach ``target_s`` and its sensor data to arrive, then reply.

        Gazebo stamps each sensor message with the simulation time it was
        produced at. After a step, the data the robot "has" is the newest
        message of each sensor stamped at or before ``target_s``. The clock
        reaching ``target_s`` says the iterations ran; a ``quiet_s`` pause in
        sensor traffic says the messages published during them have arrived.
        Selection is by stamp, so the reply is a function of simulated
        time, not of arrival order. A sensor silent for longer than two
        periods is an error, not stale data.
        """
        tol = 1.5 * self._physics_step_s
        with self._lock:
            if not self._lock.wait_for(
                lambda: abs(self._clock_s - target_s) <= tol, timeout=self._timeout_s
            ):
                raise BridgeError(
                    f"the simulation clock read {self._clock_s:.3f} s, expected {target_s:.3f} s"
                )
        # The clock says the iterations ran, not that the sensor messages they
        # produced have crossed the bridge (software-rendered images lag).
        # Wait until none has arrived for quiet_s. Wall time only decides when
        # to stop waiting; which message is used is decided by stamp below.
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() - self._last_sensor_wall_s < self._quiet_s:
            if time.monotonic() > deadline:
                raise BridgeError("sensor messages kept arriving after the world stopped")
            time.sleep(0.01)
        with self._lock:
            odoms = [o for o in self._odoms if o["t"] <= target_s + tol]
            bundles = [b for b in self._bundles if b["t"] <= target_s + tol]
            if not odoms or not bundles:
                raise BridgeError(
                    f"no sensor data at or before sim time {target_s:.3f} s "
                    "(is a stereo, depth or odometry publisher running?)"
                )
            odom, bundle = dict(odoms[-1]), dict(bundles[-1])
        stale = max(target_s - odom["t"], target_s - bundle["t"])
        if stale > 2.0 * self._camera_period_s:
            raise BridgeError(
                f"the newest sensor data is {stale:.3f} s old at sim time {target_s:.3f} s: "
                "a stereo, depth or odometry publisher has stalled"
            )
        ok, png = cv2.imencode(".png", bundle["left"])
        if not ok:
            raise BridgeError("could not encode the left image")
        depth = np.ascontiguousarray(bundle["depth"], dtype="<f4")
        return {
            "ok": True,
            "t": round(target_s - self._base_s, 9),
            "pose": self._world_pose(odom),
            "rgb_png": base64.b64encode(png.tobytes()).decode("ascii"),
            "depth_f32": base64.b64encode(depth.tobytes()).decode("ascii"),
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "image_t": round(bundle["t"] - self._base_s, 9),
            "pose_t": round(odom["t"] - self._base_s, 9),
            "sync_slop_s": self._slop_s,
        }

    def _reset(self, start: Any) -> dict[str, Any]:
        # "Reset all" restores poses but not the vehicle's whole dynamic state:
        # with a 0.1 s settle, the first frame of an episode differed from run
        # to run depending on how the previous episode ended (measured: five
        # different frames after episodes of 30, 5, 30, 0 and 12 steps). A 2 s
        # settle lets the suspension come to rest and the frames agree exactly.
        self._publish(0.0, 0.0)
        self._call_control(pause=True, reset_all=True)
        try:
            sx, sy, syaw = (float(start[key]) for key in ("x", "y", "yaw"))
        except (TypeError, KeyError, ValueError) as error:
            raise BridgeError(f"reset needs a start pose with x, y and yaw: {error}") from error
        self._start = (sx, sy, syaw)
        if (sx, sy, syaw) != (0.0, 0.0, 0.0):
            # The world's own start is the origin, facing +x; anything else is a move.
            self._place_robot(sx, sy, syaw)
        # Messages from before the reset may still be in flight; let them land
        # before the histories are cleared so none is mistaken for new data.
        time.sleep(max(0.1, 2.0 * self._quiet_s))
        with self._lock:
            self._bundles, self._odoms, self._clock_s = [], [], -1.0
        # Settle long enough for every sensor to publish once; t = 0 is the end of it.
        self._call_control(pause=True, multi_step=self._settle_iterations)
        # The world restarts at sim time 0, so the settled state is at exactly
        # settle_iterations physics steps.
        self._now_s = self._base_s = self._settle_iterations * self._physics_step_s
        return self._wait_for(self._now_s)

    def _step(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            dt_s = float(request["dt_s"])
            linear = 0.0 if request.get("stop") else float(request["linear"])
            angular = 0.0 if request.get("stop") else float(request["angular"])
        except (KeyError, TypeError, ValueError) as error:
            raise BridgeError(f"malformed step request: {error}") from error
        iterations = round(dt_s / self._physics_step_s)
        if iterations < 1:
            raise BridgeError(f"dt_s={dt_s} is shorter than one physics step")
        self._publish(linear, angular)
        # ros_gz_bridge gives no acknowledgement, and a world stepped before the
        # command arrives would run one period on the previous command. A short
        # wall-clock pause (command_settle_s) covers the hop; it delays the step
        # but never changes what the step contains. docs/simulation.md describes
        # the replay check that guards this.
        time.sleep(self._settle_s)
        return self._wait_for(self._advance(iterations))

    def destroy_node(self) -> None:
        self._running = False
        super().destroy_node()


def main() -> None:
    """Spin the bridge until shutdown."""
    rclpy.init()
    node = BridgeNode()
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
