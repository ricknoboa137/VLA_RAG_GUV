#!/usr/bin/env python3
"""Keep the arm's single ``gripper`` joint, on top of a two-finger gripper.

Why this node exists. The arm in this project has a parallel gripper: a micro
servo turns a pinion that drives two racks, so the two fingers slide in
opposite directions. The simulation models that honestly, as two prismatic
joints. Everything outside the simulation, though, was written against the
SO-101's gripper as LeRobot and the URDF describe it: one joint named
``gripper``, commanded on ``/carma/arm/gripper/cmd_pos`` in radians, between
-0.174533 (closed) and 1.74533 (open). The VLA configs under
``configs/arm_policy/`` map the policies' 0..100 gripper onto exactly that
range, and ``carma_vla`` addresses the joint by that name.

Rewriting every consumer for a change in the simulator's internals would be the
wrong place to pay for it, so this node translates at the boundary:

* ``/carma/arm/gripper/cmd_pos`` (radians, the old contract) becomes a finger
  travel in metres on ``/carma/arm/gripper_plus/cmd_pos`` and
  ``/carma/arm/gripper_minus/cmd_pos``, which the bridge hands to Gazebo.
* The simulator's joint states arrive on ``/carma/arm/gz_joint_states`` with
  the two finger joints; this node republishes them on
  ``/carma/arm/joint_states`` with the pair folded back into one ``gripper``
  entry in radians.

The mapping is linear, so a policy's "50 % open" is a gripper half open, and
the angle is a scale rather than a physical rotation: the servo's real angle
depends on the pinion, which the simulation does not model.

The mapping functions are plain Python so tests can import this file without
a ROS installation; rclpy is only imported when the node actually runs.
"""

from __future__ import annotations

# The contract: the hinged jaw's limits in the SO-101 URDF.
GRIPPER_CLOSED_RAD = -0.174533
GRIPPER_OPEN_RAD = 1.74533

# How far each finger slides from closed to fully open. Must equal the upper
# limit of the prismatic joints in models/so101_follower/model.sdf; a test pins
# the two together.
STROKE_M = 0.014

FINGER_JOINTS = ("gripper_plus", "gripper_minus")


def finger_from_gripper(angle_rad: float, stroke_m: float = STROKE_M) -> float:
    """Finger travel in metres for a ``gripper`` command in radians.

    Out-of-range commands are clamped, not rejected: a policy that overshoots
    its range should still close the gripper, not leave it where it was.
    """
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    fraction = (angle_rad - GRIPPER_CLOSED_RAD) / span
    return stroke_m * min(1.0, max(0.0, fraction))


def gripper_from_finger(travel_m: float, stroke_m: float = STROKE_M) -> float:
    """The ``gripper`` angle in radians that corresponds to a finger travel."""
    fraction = min(1.0, max(0.0, travel_m / stroke_m))
    return GRIPPER_CLOSED_RAD + fraction * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def fold_fingers(
    names: list[str],
    positions: list[float],
    velocities: list[float],
    stroke_m: float = STROKE_M,
) -> tuple[list[str], list[float], list[float]]:
    """Replace the two finger joints by one ``gripper`` joint, appended last.

    The fingers move together by construction, so their mean is the gripper.
    A state message missing one finger yields the other alone rather than
    nothing; a message with neither leaves the joint list as it came.
    """
    index = {name: i for i, name in enumerate(names)}
    present = [j for j in FINGER_JOINTS if j in index]
    kept = [i for i, name in enumerate(names) if name not in FINGER_JOINTS]
    out_names = [names[i] for i in kept]
    out_pos = [positions[i] for i in kept]
    out_vel = [velocities[i] for i in kept] if velocities else []
    if not present:
        return out_names, out_pos, out_vel

    travel = sum(positions[index[j]] for j in present) / len(present)
    out_names.append("gripper")
    out_pos.append(gripper_from_finger(travel, stroke_m))
    if velocities:
        speed = sum(velocities[index[j]] for j in present) / len(present)
        out_vel.append(speed / stroke_m * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD))
    return out_names, out_pos, out_vel


def main() -> None:
    """Spin the relay until shutdown."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64

    class GripperRelay(Node):
        """Translates the gripper command and joint state at the simulator boundary."""

        def __init__(self) -> None:
            super().__init__("carma_gripper_relay")
            self.declare_parameter("stroke_m", STROKE_M)
            self._stroke = float(self.get_parameter("stroke_m").value)

            self._outputs = {
                joint: self.create_publisher(Float64, f"/carma/arm/{joint}/cmd_pos", 10)
                for joint in FINGER_JOINTS
            }
            self._states = self.create_publisher(JointState, "/carma/arm/joint_states", 10)
            self.create_subscription(Float64, "/carma/arm/gripper/cmd_pos", self._command, 10)
            self.create_subscription(JointState, "/carma/arm/gz_joint_states", self._state, 10)
            self.get_logger().info(
                f"gripper {GRIPPER_CLOSED_RAD:+.3f}..{GRIPPER_OPEN_RAD:+.3f} rad "
                f"-> fingers 0..{self._stroke * 1000:.1f} mm"
            )

        def _command(self, msg: Float64) -> None:
            travel = finger_from_gripper(msg.data, self._stroke)
            for publisher in self._outputs.values():
                publisher.publish(Float64(data=travel))

        def _state(self, msg: JointState) -> None:
            names, positions, velocities = fold_fingers(
                list(msg.name), list(msg.position), list(msg.velocity), self._stroke
            )
            out = JointState()
            out.header = msg.header
            out.name = names
            out.position = positions
            out.velocity = velocities
            self._states.publish(out)

    rclpy.init()
    node = GripperRelay()
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
