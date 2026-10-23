"""The arm policy adapter's pure parts: units, spread, uncertainty, cameras, specs.

``scripts/arm_policy.py`` imports torch and LeRobot only when a model is built,
so everything that decides what crosses the wire is tested here without them,
offline, in the ``carma`` environment. The model path itself is exercised by
``policy_server.py --self-check`` in the ``carma-train`` environment.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from carma.backbone.wire import ActRequest, encode_image

_ROOT = Path(__file__).resolve().parents[1]
_SPECS = sorted((_ROOT / "configs" / "arm_policy").glob("*.yaml"))
_ARM_SDF = _ROOT / "ros2_ws" / "src" / "carma_sim" / "models" / "so101_follower" / "model.sdf"
_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.fixture(scope="module")
def arm() -> ModuleType:
    path = _ROOT / "scripts" / "arm_policy.py"
    spec = importlib.util.spec_from_file_location("arm_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(**overrides: object) -> ActRequest:
    base: dict[str, object] = {
        "image": encode_image(np.zeros((4, 6, 3), dtype=np.uint8)),
        "instruction": "pick up the red cube",
        "pose": (0.0, 0.0, 0.0),
        "t": 0.0,
        "stage": "VEGETATIVE",
        "seed": 0,
    }
    base.update(overrides)
    return ActRequest(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ units


def test_range_units_put_the_ends_of_the_range_at_the_joint_limits(arm: ModuleType) -> None:
    lift = arm.JointUnits("shoulder_lift", model=(-100.0, 100.0), rad=(-1.74533, 1.74533))
    assert lift.to_rad(np.array([-100.0, 0.0, 50.0, 100.0])) == pytest.approx(
        [-1.74533, 0.0, 0.872665, 1.74533]
    )
    assert lift.range_rad == pytest.approx(3.49066)


def test_units_round_trip(arm: ModuleType) -> None:
    gripper = arm.JointUnits("gripper", model=(0.0, 100.0), rad=(-0.174533, 1.74533))
    values = np.array([0.0, 12.5, 100.0])
    assert gripper.to_model(gripper.to_rad(values)) == pytest.approx(values)
    # 10 % of a 1.919863 rad range above the closed end.
    assert float(gripper.to_rad(np.float64(10.0))) == pytest.approx(-0.174533 + 0.1919863)


def test_a_degenerate_unit_map_is_rejected(arm: ModuleType) -> None:
    with pytest.raises(ValueError, match="differ"):
        arm.JointUnits("x", model=(1.0, 1.0), rad=(0.0, 1.0))


def test_a_missing_joint_is_an_error_not_a_zero(arm: ModuleType) -> None:
    """Zero is a real pose; filling a missing joint with it would mislead the model."""
    spec = arm.load_spec(_SPECS[0])
    with pytest.raises(ValueError, match="gripper"):
        spec.state_in_model_units(_request(arm_state=tuple((n, 0.0) for n in _JOINTS[:-1])))


def test_state_converts_to_model_units_in_model_order(arm: ModuleType) -> None:
    spec = arm.load_spec(_SPECS[0])
    # Given out of order on purpose: the spec's order decides the vector.
    state = (("gripper", -0.174533), ("shoulder_pan", 0.959930), *((n, 0.0) for n in _JOINTS[1:5]))
    vector = spec.state_in_model_units(_request(arm_state=state))
    assert vector == pytest.approx([50.0, 0.0, 0.0, 0.0, -1.7432, 0.0], abs=1e-3)


# -------------------------------------------------------- spread, uncertainty


def test_spread_is_the_mean_std_across_samples_over_each_range(arm: ModuleType) -> None:
    # Two samples, one step, two joints. Joint 0 differs by 0.2 rad (std 0.1)
    # on a 1.0 rad range; joint 1 agrees. Mean of (0.1, 0.0) = 0.05.
    chunks = np.array([[[0.0, 0.5]], [[0.2, 0.5]]])
    assert arm.chunk_spread(chunks, np.array([1.0, 2.0])) == pytest.approx(0.05)


def test_one_sample_has_no_spread(arm: ModuleType) -> None:
    assert arm.chunk_spread(np.zeros((1, 5, 6)), np.ones(6)) == 0.0


def test_spread_rejects_the_wrong_shape(arm: ModuleType) -> None:
    with pytest.raises(ValueError, match="samples, steps, joints"):
        arm.chunk_spread(np.zeros((5, 6)), np.ones(6))


def test_uncalibrated_uncertainty_is_half_at_its_scale(arm: ModuleType) -> None:
    u = arm.UncertaintyMap(method="uncalibrated", scale=0.02)
    assert u(0.0) == 0.0
    assert u(0.02) == pytest.approx(0.5)
    assert u(0.06) == pytest.approx(0.75)
    assert u.key == "chunk_spread/uncalibrated"


def test_logistic_uncertainty(arm: ModuleType) -> None:
    u = arm.UncertaintyMap(method="logistic", a=2.0, b=1.0, source="fit.json")
    # sigmoid(2 * ln(e^-0.5) + 1) = sigmoid(0) = 0.5
    assert u(math.exp(-0.5)) == pytest.approx(0.5)
    assert u(0.0) == 0.0
    assert u.key == "chunk_spread/logistic"


def test_logistic_fit_recovers_the_hand_computed_maximum_likelihood(arm: ModuleType) -> None:
    # Four chunks at ln(s) = -1 with one exceeding, four at ln(s) = +1 with three.
    # MLE: sigmoid(-a + b) = 1/4 and sigmoid(a + b) = 3/4, so b = 0, a = ln 3.
    spreads = np.array([math.exp(-1.0)] * 4 + [math.exp(1.0)] * 4)
    exceeded = np.array([True, False, False, False, True, True, True, False])
    a, b = arm.fit_logistic(spreads, exceeded, ridge=0.0)
    assert a == pytest.approx(math.log(3.0), abs=1e-8)
    assert b == pytest.approx(0.0, abs=1e-8)


def test_logistic_fit_stays_finite_on_separable_data(arm: ModuleType) -> None:
    spreads = np.array([0.001, 0.002, 0.1, 0.2])
    a, b = arm.fit_logistic(spreads, np.array([False, False, True, True]))
    assert math.isfinite(a) and math.isfinite(b) and a > 0.0


def test_logistic_fit_rejects_a_zero_spread(arm: ModuleType) -> None:
    with pytest.raises(ValueError, match="positive"):
        arm.fit_logistic(np.array([0.0, 0.1]), np.array([False, True]))


def test_an_unreadable_spread_is_full_uncertainty(arm: ModuleType) -> None:
    u = arm.UncertaintyMap()
    assert u(float("nan")) == 1.0
    assert u(-1.0) == 1.0


def test_an_unknown_uncertainty_method_is_rejected(arm: ModuleType) -> None:
    with pytest.raises(ValueError, match="method"):
        arm.UncertaintyMap(method="vibes")


# ---------------------------------------------------------------- cameras


def test_camera_slots_take_their_sources_and_report_the_missing(arm: ModuleType) -> None:
    arm_frame = np.full((4, 6, 3), 10, dtype=np.uint8)
    world_frame = np.full((4, 6, 3), 200, dtype=np.uint8)
    request = _request(
        image=encode_image(arm_frame), views=(("world_left", encode_image(world_frame)),)
    )
    cameras = (("wrist", "image"), ("top", "world_left"), ("side", "image_right"))
    frames, missing = arm.camera_frames(request, cameras)
    assert set(frames) == {"wrist", "top"}
    assert int(frames["wrist"][0, 0, 0]) == 10
    assert int(frames["top"][0, 0, 0]) == 200
    assert missing == ["side"]


def test_frames_reach_the_model_as_rgb_at_the_feature_size(arm: ModuleType) -> None:
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    rgb = arm.to_model_rgb(bgr, 8, 12)
    assert rgb.shape == (8, 12, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


# ------------------------------------------------------------------ specs


@pytest.mark.parametrize("path", _SPECS, ids=lambda p: p.stem)
def test_each_spec_loads_and_pins_a_revision(arm: ModuleType, path: Path) -> None:
    spec = arm.load_spec(path)
    assert spec.joint_names == _JOINTS
    assert re.fullmatch(r"[0-9a-f]{40}", spec.model_revision)
    assert str(spec.checkpoint).replace("\\", "/").startswith("assets/")
    assert spec.cameras[0][1] == "image", "the arm camera, the primary view, comes first"


def _pairs(arm: ModuleType) -> list[list[object]]:
    """The specs grouped into compared pairs: by checkpoint origin, community or fine-tuned here."""
    specs = [arm.load_spec(p) for p in _SPECS]
    groups: dict[bool, list[object]] = {}
    for spec in specs:
        groups.setdefault(spec.uncertainty.method == "logistic", []).append(spec)
    return list(groups.values())


def test_the_compared_models_share_units_cameras_and_uncertainty(arm: ModuleType) -> None:
    """A benchmark of two models must differ only by the model.

    A fitted uncertainty map belongs to its model, so a fine-tuned pair shares
    the method and the tolerance it was fitted against, not the coefficients.
    """
    for pair in _pairs(arm):
        assert sorted(s.kind for s in pair) == ["smolvla", "xvla"]
        first, other = pair
        assert other.joints == first.joints
        assert other.fps == first.fps
        assert sorted(src for _, src in other.cameras) == sorted(src for _, src in first.cameras)
        if first.uncertainty.method == "logistic":
            assert other.uncertainty.method == "logistic"
            assert other.uncertainty.tolerance_rad == first.uncertainty.tolerance_rad > 0.0
            assert other.uncertainty.source == first.uncertainty.source != ""
        else:
            assert other.uncertainty == first.uncertainty


def test_spec_joint_maps_end_at_the_simulated_arms_limits(arm: ModuleType) -> None:
    """The radians a model is mapped to are the ones the Gazebo arm accepts."""
    sdf = _ARM_SDF.read_text(encoding="utf-8")
    pattern = (
        r'<joint name="(\w+)" type="revolute">.*?'
        r"<lower>([-\d.]+)</lower><upper>([-\d.]+)</upper>"
    )
    limits = {
        name: (float(lo), float(hi)) for name, lo, hi in re.findall(pattern, sdf, flags=re.DOTALL)
    }
    for joint in arm.load_spec(_SPECS[0]).joints:
        assert joint.rad == pytest.approx(limits[joint.name]), joint.name
