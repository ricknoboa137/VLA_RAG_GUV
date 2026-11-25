"""The simulated SO-101 gripper is the standard hinged jaw the follower is built with.

``test_arm_kinematics.py`` pins the ``gripper`` joint's origin, axis, limits and
the jaw's mass to the official URDF along with the five arm joints. This file
covers the rest of the gripper: the vendored meshes, the light visual meshes
and collision boxes made from them by ``tools/build_gripper_geometry.py``, the
grasp frame, the controller and the topics. Nothing here needs Gazebo or ROS:
the SDF is parsed as XML and the meshes are read as binary STL.

The URDF numbers below are copied from
``TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`` (Apache-2.0),
the same file the kinematics test cites.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml

_SIM = Path("ros2_ws/src/carma_sim")
_MODEL_DIR = _SIM / "models/so101_follower"
_MODEL = _MODEL_DIR / "model.sdf"
_MESHES = _MODEL_DIR / "meshes"
_WORLD = _SIM / "worlds/row_crop.sdf"
_BRIDGE = _SIM / "config/bridge.yaml"
_URI_PREFIX = "model://so101_follower/meshes/visual/"

# mesh file -> (link, URDF visual origin xyz, rpy).
_URDF_MESHES = {
    "wrist_roll_follower_so101_v1.stl": (
        "gripper_link",
        (8.32667e-17, -0.000218214, 0.000949706),
        (-3.14159, -5.55112e-17, 0.0),
    ),
    "sts3215_03a_v1.stl": (
        "gripper_link",
        (0.0077, 0.0001, -0.0234),
        (-1.5708, -5.19179e-17, -1.66533e-16),
    ),
    "moving_jaw_so101_v1.stl": (
        "moving_jaw_link",
        (-5.55112e-17, -5.55112e-17, 0.0189),
        (9.53145e-17, 6.93889e-18, 1.24077e-24),
    ),
}
# The URDF's gripper_frame_joint origin: the grasp point between the jaw tips.
_URDF_GRIPPER_FRAME = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))
_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture(scope="module")
def sdf() -> ET.Element:
    return ET.parse(_MODEL).getroot()


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """The geometry builder; plain numpy, imported from its path."""
    spec = importlib.util.spec_from_file_location(
        "build_gripper_geometry", _SIM / "tools/build_gripper_geometry.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pose(node: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([float(v) for v in (node.findtext("pose") or "").split()])
    return values[:3], values[3:]


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _same_pose(xyz: np.ndarray, rpy: np.ndarray, ref: tuple[tuple[float, ...], ...]) -> bool:
    """Positions to a micrometre, orientations as rotation matrices (+pi and -pi agree)."""
    return bool(
        np.allclose(xyz, ref[0], atol=1e-6)
        and np.allclose(_rotation(*rpy), _rotation(*ref[1]), atol=1e-5)
    )


def _stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    assert len(data) == 84 + 50 * count, f"{path.name} is not a complete binary STL"
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data[84:], dtype=record)["vertices"].reshape(-1, 3)


def _mesh_nodes(sdf: ET.Element) -> list[tuple[str, str, ET.Element]]:
    """(link, file, element) for every visual or collision in the model drawn with a mesh."""
    out = []
    for link in sdf.iter("link"):
        for kind in ("visual", "collision"):
            for node in link.iter(kind):
                uri = node.findtext("geometry/mesh/uri")
                if uri is not None:
                    assert uri.startswith(_URI_PREFIX), uri
                    out.append((str(link.get("name")), uri[len(_URI_PREFIX) :], node))
    return out


def test_every_mesh_is_vendored_and_matches_its_recorded_checksum() -> None:
    """The README's table is the provenance record; it must describe the files present."""
    table = dict(
        re.findall(
            r"\| `([\w.]+\.stl)` \|.*?\| `([0-9a-f]{64})` \|", (_MESHES / "README.md").read_text()
        )
    )
    on_disk = sorted(p.name for p in _MESHES.glob("*.stl"))
    assert sorted(table) == on_disk == sorted(_URDF_MESHES)
    for name, digest in table.items():
        assert hashlib.sha256((_MESHES / name).read_bytes()).hexdigest() == digest, name
    assert (_MESHES / "LICENSE").read_text().lstrip().startswith("Apache License")


@pytest.mark.parametrize("name", sorted(_URDF_MESHES))
def test_meshes_are_binary_stl_in_metres(name: str) -> None:
    """A millimetre mesh would draw the gripper a thousand times too big."""
    extent = np.ptp(_stl_vertices(_MESHES / name), axis=0)
    assert 0.02 < float(extent.max()) < 0.12, f"{name} spans {extent} m"


def test_each_part_is_drawn_where_the_urdf_puts_it(sdf: ET.Element) -> None:
    """Each part has one visual, from its light mesh, at the URDF's pose; no mesh collides."""
    seen: dict[str, list[str]] = {}
    for link, name, node in _mesh_nodes(sdf):
        expected_link, xyz, rpy = _URDF_MESHES[name]
        assert link == expected_link, f"{name} is on {link}"
        assert _same_pose(*_pose(node), (xyz, rpy)), f"{name} {node.tag} pose"
        seen.setdefault(name, []).append(node.tag)
    assert seen == {name: ["visual"] for name in _URDF_MESHES}


def test_the_light_visual_meshes_are_what_the_tool_builds(tool: ModuleType) -> None:
    for name in _URDF_MESHES:
        original = tool.read_stl(_MESHES / name)
        built = tool.stl_bytes(tool.simplify(original))
        assert (
            _MESHES / "visual" / name
        ).read_bytes() == built, f"{name}: rerun tools/build_gripper_geometry.py"
        # Simplification must not move the part: its outline stays within one grid cell.
        light = tool.read_stl(_MESHES / "visual" / name).reshape(-1, 3)
        whole = original.reshape(-1, 3)
        assert np.allclose(light.min(axis=0), whole.min(axis=0), atol=tool.GRID_M)
        assert np.allclose(light.max(axis=0), whole.max(axis=0), atol=tool.GRID_M)
        assert len(light) < len(whole) / 5


def test_the_collision_boxes_are_what_the_tool_builds(tool: ModuleType) -> None:
    sdf = _MODEL.read_text(encoding="utf-8")
    originals = {name: tool.read_stl(_MESHES / name) for name in _URDF_MESHES}
    assert tool.render_sdf(sdf, originals) == sdf, "rerun tools/build_gripper_geometry.py"


def test_every_gripper_collision_is_a_generated_box(sdf: ET.Element, tool: ModuleType) -> None:
    links = {link.get("name"): link for link in sdf.iter("link")}
    for name, (part, _axis, slabs) in tool.PLAN.items():
        link = links[_URDF_MESHES[name][0]]
        boxes = [c for c in link.iter("collision") if str(c.get("name")).startswith(part + "_")]
        assert len(boxes) == slabs, part
        assert all(c.find("geometry/box") is not None for c in boxes), part
    for link_name in ("gripper_link", "moving_jaw_link"):
        for collision in links[link_name].iter("collision"):
            assert collision.find("geometry/box") is not None, collision.get("name")


def test_the_grasp_frame_is_the_urdfs(sdf: ET.Element) -> None:
    frames = {f.get("name"): f for f in sdf.iter("frame")}
    frame = frames["gripper_frame"]
    assert frame.get("attached_to") == "gripper_link"
    assert _same_pose(*_pose(frame), _URDF_GRIPPER_FRAME)


def test_one_controller_drives_the_jaw_on_the_clients_topic(sdf: ET.Element) -> None:
    controllers = {
        p.findtext("joint_name"): p.findtext("topic")
        for p in sdf.iter("plugin")
        if p.get("name") == "gz::sim::systems::JointPositionController"
    }
    assert controllers == {j: f"/carma/arm/{j}/cmd_pos" for j in _JOINTS}
    states = [
        p for p in sdf.iter("plugin") if p.get("name") == "gz::sim::systems::JointStatePublisher"
    ]
    assert len(states) == 1
    assert tuple(n.text for n in states[0].iter("joint_name")) == _JOINTS


def test_the_bridge_carries_the_gripper_straight_through() -> None:
    """No relay: ROS and Gazebo use the same names for the command and the states."""
    entries = {e["ros_topic_name"]: e for e in yaml.safe_load(_BRIDGE.read_text())}
    command = entries["/carma/arm/gripper/cmd_pos"]
    assert (command["gz_topic_name"], command["direction"]) == (
        "/carma/arm/gripper/cmd_pos",
        "ROS_TO_GZ",
    )
    states = entries["/carma/arm/joint_states"]
    assert (states["gz_topic_name"], states["direction"]) == (
        "/carma/arm/joint_states",
        "GZ_TO_ROS",
    )


def test_nothing_in_the_simulation_still_drives_sliding_fingers() -> None:
    for path in _SIM.rglob("*"):
        if path.is_file() and path.suffix in {".sdf", ".yaml", ".py", ".txt", ".xacro"}:
            text = path.read_text(encoding="utf-8")
            assert "gripper_plus" not in text and "gripper_minus" not in text, path


def test_the_jaw_controller_is_stable_at_the_worlds_step(sdf: ET.Element) -> None:
    """The jaw is a hundred times lighter than the arm links, so it cannot share their gains.

    The controller's P and D terms are applied explicitly once per physics step.
    With I the jaw's inertia about its hinge, the discrete loop stays stable
    while D.dt/I and P.dt^2/I stay well below 1. (The joint's own damping is
    applied implicitly by the physics engine and does not limit the step.)
    """
    dt = float(ET.parse(_WORLD).getroot().findtext(".//physics/max_step_size") or "nan")
    jaw = next(link for link in sdf.iter("link") if link.get("name") == "moving_jaw_link")
    mass = float(jaw.findtext("inertial/mass") or "nan")
    com = np.array([float(v) for v in (jaw.findtext("inertial/pose") or "").split()[:3]])
    izz = float(jaw.findtext("inertial/inertia/izz") or "nan")
    # The hinge is the link frame's z axis (test_arm_kinematics pins the axis).
    inertia = izz + mass * float(com[0] ** 2 + com[1] ** 2)
    plugin = next(
        p
        for p in sdf.iter("plugin")
        if p.get("name") == "gz::sim::systems::JointPositionController"
        and p.findtext("joint_name") == "gripper"
    )
    p_gain, d_gain = (
        float(plugin.findtext("p_gain") or "nan"),
        float(plugin.findtext("d_gain") or "nan"),
    )
    assert d_gain * dt / inertia < 0.5
    assert p_gain * dt**2 / inertia < 0.5
