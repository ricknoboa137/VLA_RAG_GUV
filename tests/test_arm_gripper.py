"""The arm's parallel gripper, and the relay that keeps its old interface alive.

The arm carries a rack-and-pinion parallel gripper, not the URDF's hinged jaw,
so the gripper is checked on its own terms here; ``test_arm_kinematics.py``
covers the five arm joints against the URDF. Nothing in this file opens a
socket or needs Gazebo or ROS: the SDF is parsed as XML and the relay is
imported for its plain-Python mapping functions only.
"""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SIM = Path("ros2_ws/src/carma_sim")
_MODEL = _SIM / "models/so101_follower/model.sdf"
_BOXES = _SIM / "models/so101_follower/gripper_boxes.json"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def relay() -> ModuleType:
    return _load(_SIM / "scripts/gripper_relay.py", "gripper_relay")


@pytest.fixture(scope="module")
def sdf() -> ET.Element:
    return ET.parse(_MODEL).getroot()


@pytest.fixture(scope="module")
def data() -> dict:  # type: ignore[type-arg]
    return json.loads(_BOXES.read_text(encoding="utf-8"))


def _links(sdf: ET.Element) -> dict[str, ET.Element]:
    return {str(link.get("name")): link for link in sdf.iter("link")}


def _boxes(link: ET.Element, element: str = "collision") -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for node in link.iter(element):
        pose = node.find("pose")
        size = node.find("geometry/box/size")
        if pose is None or size is None or not pose.text or not size.text:
            continue
        out.append(
            (
                np.array([float(v) for v in pose.text.split()[:3]]),
                np.array([float(v) for v in size.text.split()]),
            ),
        )
    return out


# ------------------------------------------------------------ generated model


def test_model_matches_the_box_data_it_is_generated_from() -> None:
    """The SDF is generated; a hand edit to it would silently diverge from the CAD data."""
    tool = _load(_SIM / "tools/build_gripper_sdf.py", "build_gripper_sdf")
    current = _MODEL.read_text(encoding="utf-8")
    assert tool.build() == current, "model.sdf is stale: run tools/build_gripper_sdf.py"


def test_the_box_data_records_where_it_came_from(data: dict) -> None:  # type: ignore[type-arg]
    """Geometry with no stated origin, assumptions or licence status is not trustworthy."""
    assert data["source"]["file"]
    assert (
        "NOT VERIFIED" in data["source"]["licence"]
    ), "update this test when the licence is settled"
    assert len(data["frame"]["assumptions"]) >= 3
    assert data["mechanism"]["density_basis"]


# ------------------------------------------------------------------- joints


def test_the_hinged_jaw_is_gone(sdf: ET.Element) -> None:
    names = {str(j.get("name")) for j in sdf.iter("joint")}
    assert "gripper" not in names, "the single hinged jaw must not coexist with the two fingers"
    assert "moving_jaw_link" not in _links(sdf)


def test_two_prismatic_fingers_open_in_opposite_directions(sdf: ET.Element) -> None:
    joints = {str(j.get("name")): j for j in sdf.iter("joint")}
    plus, minus = joints["gripper_plus"], joints["gripper_minus"]
    for joint in (plus, minus):
        assert joint.get("type") == "prismatic"
        assert joint.findtext("parent") == "gripper_link"
    axis_plus = [float(v) for v in (plus.findtext("axis/xyz") or "").split()]
    axis_minus = [float(v) for v in (minus.findtext("axis/xyz") or "").split()]
    # Positive travel is opening for both, so the axes must point away from each other.
    assert axis_plus == [1.0, 0.0, 0.0]
    assert axis_minus == [-1.0, 0.0, 0.0]


def test_both_fingers_share_one_stroke_that_starts_closed(sdf: ET.Element) -> None:
    joints = {str(j.get("name")): j for j in sdf.iter("joint")}
    for name in ("gripper_plus", "gripper_minus"):
        lower = float(joints[name].findtext("axis/limit/lower") or "nan")
        upper = float(joints[name].findtext("axis/limit/upper") or "nan")
        assert lower == 0.0, "joint zero is the CAD pose, which is the closed limit"
        assert upper == pytest.approx(0.014)


def test_the_relay_stroke_equals_the_joint_limit(sdf: ET.Element, relay: ModuleType) -> None:
    """If these drift apart a fully open command either stops short or is clipped."""
    joints = {str(j.get("name")): j for j in sdf.iter("joint")}
    upper = float(joints["gripper_plus"].findtext("axis/limit/upper") or "nan")
    assert upper == pytest.approx(relay.STROKE_M)


# ----------------------------------------------------------------- geometry


def _stem_faces(link: ET.Element) -> tuple[float, float]:
    """Outermost x faces of the fingertip stem: boxes beyond the racks, past 80 mm."""
    lows, highs = [], []
    for centre, size in _boxes(link):
        if centre[2] < -0.080:
            lows.append(centre[0] - size[0] / 2)
            highs.append(centre[0] + size[0] / 2)
    return min(lows), max(highs)


def test_closed_gap_matches_the_cad(sdf: ET.Element, data: dict) -> None:  # type: ignore[type-arg]
    links = _links(sdf)
    plus_low, _ = _stem_faces(links["gripper_finger_plus"])
    _, minus_high = _stem_faces(links["gripper_finger_minus"])
    gap_mm = (plus_low - minus_high) * 1000.0
    assert gap_mm == pytest.approx(data["mechanism"]["closed_gap_mm"], abs=0.5)


def test_open_gap_clears_a_40_mm_block(sdf: ET.Element, data: dict) -> None:  # type: ignore[type-arg]
    """The platform plan picks up a 40 mm block, so the open gap has to exceed it."""
    links = _links(sdf)
    plus_low, _ = _stem_faces(links["gripper_finger_plus"])
    _, minus_high = _stem_faces(links["gripper_finger_minus"])
    closed_mm = (plus_low - minus_high) * 1000.0
    open_mm = closed_mm + 2 * data["mechanism"]["stroke_mm_per_finger"]
    assert open_mm > 40.0


def test_the_fingers_are_centred_on_the_roll_axis(sdf: ET.Element) -> None:
    links = _links(sdf)
    plus_low, _ = _stem_faces(links["gripper_finger_plus"])
    _, minus_high = _stem_faces(links["gripper_finger_minus"])
    assert abs((plus_low + minus_high) / 2) < 0.002


def test_fingertips_reach_the_documented_depth(sdf: ET.Element) -> None:
    tip = min(c[2] - s[2] / 2 for c, s in _boxes(_links(sdf)["gripper_finger_plus"]))
    assert tip == pytest.approx(-0.111, abs=0.002)


def test_the_grasp_frame_is_between_the_fingertips(sdf: ET.Element) -> None:
    frame = next(f for f in sdf.iter("frame") if f.get("name") == "gripper_frame")
    x, y, z = (float(v) for v in (frame.findtext("pose") or "").split()[:3])
    assert abs(x) < 0.002 and abs(y) < 0.002
    assert -0.111 < z < -0.091, "inside the distal 20 mm of the fingertips"


# ----------------------------------------------------------------- collisions


def test_the_fingers_cannot_collide_with_each_other_or_the_body(sdf: ET.Element) -> None:
    """At joint zero the CAD has the fingers touching; contact there would be pure jitter."""
    masks = {}
    for name in ("gripper_link", "gripper_finger_plus", "gripper_finger_minus"):
        link = _links(sdf)[name]
        found = {
            int(node.text, 16) for node in link.iter("collide_bitmask") if node.text is not None
        }
        masks[name] = found
    assert masks["gripper_link"] == {0x01}
    assert masks["gripper_finger_plus"] == {0x02}
    assert masks["gripper_finger_minus"] == {0x04}
    # Two shapes collide only if their masks share a bit; none of these do.
    assert 0x01 & 0x02 == 0 and 0x01 & 0x04 == 0 and 0x02 & 0x04 == 0


def test_no_finger_box_is_buried_in_the_other_finger(sdf: ET.Element) -> None:
    """A box deeper than 1.5 mm inside the other finger means the pose or frame is wrong."""
    links = _links(sdf)
    plus, minus = _boxes(links["gripper_finger_plus"]), _boxes(links["gripper_finger_minus"])
    deepest = 0.0
    for (c1, s1), (c2, s2) in ((a, b) for a in plus for b in minus):
        depth = (s1 + s2) / 2 - np.abs(c1 - c2)
        if (depth > 0).all():
            deepest = max(deepest, float(depth.min()))
    assert deepest < 0.0015, f"fingers interpenetrate by {deepest * 1000:.2f} mm at joint zero"


def test_the_arm_cameras_are_not_inside_the_gripper(sdf: ET.Element) -> None:
    """A camera in a solid sees nothing useful, and the old mount was inside the new body."""
    links = _links(sdf)
    solids = [
        *_boxes(links["gripper_link"]),
        *_boxes(links["gripper_finger_plus"]),
        *_boxes(links["gripper_finger_minus"]),
    ]
    for sensor in links["gripper_link"].iter("sensor"):
        text = sensor.findtext("pose") or ""
        position = np.array([float(v) for v in text.split()[:3]])
        for centre, size in solids:
            assert not (
                np.abs(position - centre) < size / 2
            ).all(), f"{sensor.get('name')} at {position} is inside a gripper box"


def test_the_camera_baseline_is_still_the_calibrated_60_mm(sdf: ET.Element) -> None:
    poses = {
        str(s.get("name")): [float(v) for v in (s.findtext("pose") or "").split()]
        for s in sdf.iter("sensor")
    }
    left, right = poses["arm_stereo_left"], poses["arm_stereo_right"]
    assert left[1] - right[1] == pytest.approx(0.060, abs=1e-6)
    assert left[1] > right[1], "the left camera must be on the +y (image-left) side"


# ---------------------------------------------------------------- the relay


def test_the_gripper_range_maps_to_the_full_stroke(relay: ModuleType) -> None:
    assert relay.finger_from_gripper(relay.GRIPPER_CLOSED_RAD) == pytest.approx(0.0)
    assert relay.finger_from_gripper(relay.GRIPPER_OPEN_RAD) == pytest.approx(relay.STROKE_M)
    middle = (relay.GRIPPER_CLOSED_RAD + relay.GRIPPER_OPEN_RAD) / 2
    assert relay.finger_from_gripper(middle) == pytest.approx(relay.STROKE_M / 2)


def test_overshooting_commands_are_clamped_not_ignored(relay: ModuleType) -> None:
    assert relay.finger_from_gripper(-3.0) == 0.0
    assert relay.finger_from_gripper(+3.0) == pytest.approx(relay.STROKE_M)


def test_the_mapping_round_trips(relay: ModuleType) -> None:
    for angle in np.linspace(relay.GRIPPER_CLOSED_RAD, relay.GRIPPER_OPEN_RAD, 25):
        back = relay.gripper_from_finger(relay.finger_from_gripper(float(angle)))
        assert back == pytest.approx(float(angle), abs=1e-9)


def test_the_policy_range_maps_onto_the_same_fraction(relay: ModuleType) -> None:
    """configs/arm_policy maps range_0_100 onto this exact radian range: 50 means half open."""
    low, high = relay.GRIPPER_CLOSED_RAD, relay.GRIPPER_OPEN_RAD
    assert (low, high) == (-0.174533, 1.74533), "the VLA configs were written against these"
    for percent in (0, 25, 50, 75, 100):
        angle = low + percent / 100 * (high - low)
        assert relay.finger_from_gripper(angle) == pytest.approx(percent / 100 * relay.STROKE_M)


def test_folding_the_fingers_back_into_one_gripper_joint(relay: ModuleType) -> None:
    names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper_plus",
        "gripper_minus",
    ]
    positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.007, 0.007]
    velocities = [0.0] * 5 + [0.0, 0.0]
    out_names, out_pos, out_vel = relay.fold_fingers(names, positions, velocities)

    assert out_names == [*names[:5], "gripper"], "the pair becomes one entry, last, as before"
    assert out_pos[:5] == positions[:5]
    assert out_pos[5] == pytest.approx(relay.gripper_from_finger(0.007))
    assert len(out_vel) == len(out_names)


def test_folding_survives_a_message_with_only_one_finger(relay: ModuleType) -> None:
    out_names, out_pos, _ = relay.fold_fingers(
        ["wrist_roll", "gripper_plus"], [0.5, 0.014], [0.0, 0.0]
    )
    assert out_names == ["wrist_roll", "gripper"]
    assert out_pos[1] == pytest.approx(relay.GRIPPER_OPEN_RAD)


def test_folding_leaves_a_message_without_fingers_alone(relay: ModuleType) -> None:
    out_names, out_pos, _ = relay.fold_fingers(["wrist_roll"], [0.5], [0.0])
    assert (out_names, out_pos) == (["wrist_roll"], [0.5])
