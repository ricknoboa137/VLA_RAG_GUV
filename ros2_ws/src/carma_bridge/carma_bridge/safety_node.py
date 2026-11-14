"""ROS 2 node for the safety gate: the only publisher of the motor command.

Subscribes:
    /carma/cmd_vel_request (geometry_msgs/Twist): what CARMA, a teleop tool or
        the bridge wants the base to do.
    /carma/estop (std_msgs/Bool): True engages the emergency stop. On the robot
        this is wired to the hardware stop's status, so the gate also refuses
        motion when the stop is pressed.
    /carma/stall (std_msgs/Bool): True reports a stall (see motion.py).

Publishes:
    /carma/cmd_vel (geometry_msgs/Twist): the gated command, on every request
        and at ``rate_hz``, so a silent upstream becomes a zero command within
        ``command_timeout_s``.
    /carma/safety/state (std_msgs/String): the gate's state, one of
        ok, limited, estop, timeout, stalled, idle.

The logic lives in :mod:`carma_bridge.safety` and is tested there. With
``use_sim_time`` the watchdog runs on simulation time, so a paused lockstep
world never trips it.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

from carma_bridge.safety import GateOutput, SafetyGate, SafetyLimits


class SafetyNode(Node):
    """Gates every velocity command on its way to the base."""

    def __init__(self) -> None:
        super().__init__("carma_safety")
        self.declare_parameter("max_linear_mps", 0.6)
        self.declare_parameter("max_angular_radps", 1.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("rate_hz", 20.0)
        limits = SafetyLimits(
            max_linear_mps=float(self.get_parameter("max_linear_mps").value),
            max_angular_radps=float(self.get_parameter("max_angular_radps").value),
            command_timeout_s=float(self.get_parameter("command_timeout_s").value),
        )
        self._gate = SafetyGate(limits)
        self._state = ""
        self._cmd_pub = self.create_publisher(Twist, "/carma/cmd_vel", 10)
        self._state_pub = self.create_publisher(String, "/carma/safety/state", 10)
        self.create_subscription(Twist, "/carma/cmd_vel_request", self._on_request, 10)
        self.create_subscription(Bool, "/carma/estop", self._on_estop, 10)
        self.create_subscription(Bool, "/carma/stall", self._on_stall, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(
            f"safety gate: |v| <= {limits.max_linear_mps} m/s, "
            f"|w| <= {limits.max_angular_radps} rad/s, timeout {limits.command_timeout_s} s"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_request(self, msg: Twist) -> None:
        self._gate.command(self._now_s(), float(msg.linear.x), float(msg.angular.z))
        self._send(self._gate.output(self._now_s()))

    def _on_estop(self, msg: Bool) -> None:
        self._gate.set_estop(bool(msg.data))
        self._send(self._gate.output(self._now_s()))

    def _on_stall(self, msg: Bool) -> None:
        if msg.data:
            self._gate.report_stall()
            self._send(self._gate.output(self._now_s()))

    def _tick(self) -> None:
        self._send(self._gate.output(self._now_s()))

    def _send(self, out: GateOutput) -> None:
        cmd = Twist()
        cmd.linear.x = out.linear_mps
        cmd.angular.z = out.angular_radps
        self._cmd_pub.publish(cmd)
        if out.state != self._state:
            self._state = out.state
            self._state_pub.publish(String(data=out.state))
            self.get_logger().info(f"safety state: {out.state}")


def main() -> None:
    """Spin the gate until shutdown, sending a final zero."""
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
