"""The simulated SO-101 must move exactly as the real one does.

Joint angles recorded on the physical arm, or predicted by a policy trained on
LeRobot data, mean the same thing in Gazebo only if the kinematic chain is the
same one. This pins the model to the official URDF, so an edit that bends the
arm fails here instead of producing a policy that reaches for the wrong place.

The numbers below are copied verbatim from
``TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`` (Apache-2.0),
read on 2026-10-22. They are embedded rather than fetched because the suite
runs offline (AGENTS.md section 2.3), and so a change upstream cannot silently
move the reference this is checked against; re-read the file and update the
constants deliberately if the real arm's definition changes.

Geometry is not compared. The model draws links as primitives, which is a
presentation choice. What is compared is everything that decides where a link
is for a given set of joint angles: origins, axes, limits and masses of the
five arm joints. The gripper is checked separately, because the arm in this
project carries a parallel gripper rather than the URDF's hinged jaw.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_MODEL = Path("ros2_ws/src/carma_sim/models/so101_follower/model.sdf")

# joint name -> (child link, origin xyz, origin rpy, lower, upper), from the URDF.
_URDF_JOINTS = {
    "shoulder_pan": (
        "shoulder_link",
        (0.0388353, -8.97657e-09, 0.0624),
        (3.14159, 4.18253e-17, -3.14159),
        -1.91986,
        1.91986,
    ),
    "shoulder_lift": (
        "upper_arm_link",
        (-0.0303992, -0.0182778, -0.0542),
        (-1.5708, -1.5708, 0.0),
        -1.74533,
        1.74533,
    ),
    "elbow_flex": (
        "lower_arm_link",
        (-0.11257, -0.028, 1.73763e-16),
        (-3.63608e-16, 8.74301e-16, 1.5708),
        -1.69,
        1.69,
    ),
    "wrist_flex": (
        "wrist_link",
        (-0.1349, 0.0052, 3.62355e-17),
        (4.02456e-15, 8.67362e-16, -1.5708),
        -1.65806,
        1.65806,
    ),
    "wrist_roll": (
        "gripper_link",
        (5.55112e-17, -0.0611, 0.0181),
        (1.5708, 0.0486795, 3.14159),
        -2.74385,
        2.84121,
    ),
}
_ARM_CHAIN = tuple(_URDF_JOINTS)

# link -> mass in kilograms, from the URDF.
_URDF_MASSES = {
    "base_link": 0.147,
    "shoulder_link": 0.100006,
    "upper_arm_link": 0.103,
    "lower_arm_link": 0.104,
    "wrist_link": 0.079,
    "gripper_link": 0.087,
}


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF and SDF both use fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(xyz: tuple[float, ...], rpy: tuple[float, ...]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(*rpy)
    matrix[:3, 3] = xyz
    return matrix


def _about_z(angle: float) -> np.ndarray:
    matrix = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:2, :2] = [[c, -s], [s, c]]
    return matrix


class _Model:
    """The few facts about the SDF this test needs, read without Gazebo."""

    def __init__(self) -> None:
        self.root = ET.parse(_MODEL).getroot()
        self.links = {link.get("name"): link for link in self.root.iter("link")}
        self.joints = {joint.get("name"): joint for joint in self.root.iter("joint")}

    def link_pose(self, name: str) -> tuple[tuple[float, ...], tuple[float, ...], str | None]:
        """Origin, rpy and the parent frame a link's pose is given relative to."""
        pose = self.links[name].find("pose")
        assert pose is not None, f"link {name} has no pose"
        values = tuple(float(v) for v in (pose.text or "").split())
        return values[:3], values[3:], pose.get("relative_to")

    def limits(self, joint: str) -> tuple[float, float]:
        node = self.joints[joint].find("axis/limit")
        assert node is not None
        return float(node.findtext("lower", "nan")), float(node.findtext("upper", "nan"))

    def axis(self, joint: str) -> tuple[float, ...]:
        text = self.joints[joint].findtext("axis/xyz", "")
        return tuple(float(v) for v in text.split())

    def mass(self, link: str) -> float:
        return float(self.links[link].findtext("inertial/mass", "nan"))


@pytest.fixture(scope="module")
def model() -> _Model:
    return _Model()


def _forward(
    origins: dict[str, tuple[tuple[float, ...], tuple[float, ...]]], q: np.ndarray
) -> np.ndarray:
    """Pose of the last arm link frame for joint angles ``q``."""
    matrix = np.eye(4)
    for name, angle in zip(_ARM_CHAIN, q, strict=True):
        xyz, rpy = origins[name]
        matrix = matrix @ _transform(xyz, rpy) @ _about_z(float(angle))
    return matrix


def _sdf_origins(model: _Model) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    result = {}
    for joint, (child, *_rest) in _URDF_JOINTS.items():
        xyz, rpy, _ = model.link_pose(child)
        result[joint] = (xyz, rpy)
    return result


def test_each_link_hangs_from_the_link_the_urdf_says(model: _Model) -> None:
    """The chain order matters as much as the numbers: a wrong parent bends the arm."""
    expected_parent = {
        "shoulder_link": "base_link",
        "upper_arm_link": "shoulder_link",
        "lower_arm_link": "upper_arm_link",
        "wrist_link": "lower_arm_link",
        "gripper_link": "wrist_link",
    }
    for joint, (child, *_rest) in _URDF_JOINTS.items():
        _, _, parent = model.link_pose(child)
        assert parent == expected_parent[child], f"{joint}: {child} hangs from {parent}"


@pytest.mark.parametrize("joint", _ARM_CHAIN)
def test_joint_origin_matches_the_urdf(model: _Model, joint: str) -> None:
    child, xyz, rpy, _lo, _hi = _URDF_JOINTS[joint]
    sdf_xyz, sdf_rpy, _ = model.link_pose(child)
    assert np.allclose(sdf_xyz, xyz, atol=1e-6), f"{joint} position"
    # Angles are compared as rotation matrices: 3.14159 and -3.14159 are the
    # same orientation, and a plain comparison of the numbers would not see it.
    assert np.allclose(_rotation(*sdf_rpy), _rotation(*rpy), atol=1e-5), f"{joint} orientation"


@pytest.mark.parametrize("joint", _ARM_CHAIN)
def test_every_joint_turns_about_its_own_z_axis(model: _Model, joint: str) -> None:
    assert model.axis(joint) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("joint", _ARM_CHAIN)
def test_joint_limits_match_the_urdf(model: _Model, joint: str) -> None:
    _child, _xyz, _rpy, lower, upper = _URDF_JOINTS[joint]
    assert model.limits(joint) == pytest.approx((lower, upper), abs=1e-5)


@pytest.mark.parametrize("link", sorted(_URDF_MASSES))
def test_link_mass_matches_the_urdf(model: _Model, link: str) -> None:
    assert model.mass(link) == pytest.approx(_URDF_MASSES[link], rel=1e-6)


def test_the_last_arm_frame_lands_where_the_urdf_puts_it(model: _Model) -> None:
    """The decisive check: for random joint angles, both chains place the wrist alike."""
    sdf = _sdf_origins(model)
    urdf = {joint: (xyz, rpy) for joint, (_c, xyz, rpy, _lo, _hi) in _URDF_JOINTS.items()}
    rng = np.random.default_rng(0)

    worst_position = 0.0
    worst_rotation = 0.0
    for _ in range(500):
        q = np.array(
            [rng.uniform(_URDF_JOINTS[j][3], _URDF_JOINTS[j][4]) for j in _ARM_CHAIN],
        )
        a, b = _forward(sdf, q), _forward(urdf, q)
        worst_position = max(worst_position, float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        worst_rotation = max(worst_rotation, float(np.abs(a[:3, :3] - b[:3, :3]).max()))

    # The URDF prints 1e-8 and 1e-17 noise in some origins; micrometres is far
    # below anything the arm, the servos or a camera could resolve.
    assert worst_position < 1e-6, f"wrist position differs by {worst_position:.2e} m"
    assert worst_rotation < 1e-5, f"wrist orientation differs by {worst_rotation:.2e}"
