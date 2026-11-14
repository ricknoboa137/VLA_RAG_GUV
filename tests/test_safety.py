"""The bridge's safety gate, stall detector and pose estimator.

Loaded by path because the bridge package lives in the ROS workspace, not in
``src/``; these modules have no ROS imports, so they run here.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_PKG = Path(__file__).resolve().parents[1] / "ros2_ws/src/carma_bridge/carma_bridge"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bridge_{name}", _PKG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def safety() -> ModuleType:
    return _load("safety")


@pytest.fixture(scope="module")
def motion() -> ModuleType:
    return _load("motion")


# ------------------------------------------------------------------ safety gate


def test_a_fresh_command_passes_and_a_stale_one_stops(safety: ModuleType) -> None:
    gate = safety.SafetyGate(safety.SafetyLimits(command_timeout_s=0.5))
    assert gate.output(0.0).state == "idle"
    gate.command(1.0, 0.4, 0.2)
    out = gate.output(1.3)
    assert (out.linear_mps, out.angular_radps, out.state) == (0.4, 0.2, "ok")
    # 0.6 s after the command, past the 0.5 s timeout: zero.
    out = gate.output(1.6)
    assert (out.linear_mps, out.angular_radps, out.state) == (0.0, 0.0, "timeout")


def test_speeds_are_clamped_to_the_limits(safety: ModuleType) -> None:
    gate = safety.SafetyGate(safety.SafetyLimits(max_linear_mps=0.5, max_angular_radps=0.8))
    gate.command(0.0, 2.0, -3.0)
    out = gate.output(0.1)
    assert (out.linear_mps, out.angular_radps, out.state) == (0.5, -0.8, "limited")


def test_estop_stops_and_release_does_not_resume_the_old_command(safety: ModuleType) -> None:
    gate = safety.SafetyGate(safety.SafetyLimits())
    gate.command(0.0, 0.4, 0.0)
    gate.set_estop(True)
    assert gate.output(0.1).state == "estop"
    gate.command(0.2, 0.4, 0.0)  # commands while engaged are still refused
    assert gate.output(0.25).state == "estop"
    gate.set_estop(False)
    assert gate.output(0.3).state == "idle"  # nothing to resume
    gate.command(0.4, 0.3, 0.0)
    assert gate.output(0.45).linear_mps == 0.3


def test_a_stall_holds_until_a_stop_command_acknowledges_it(safety: ModuleType) -> None:
    gate = safety.SafetyGate(safety.SafetyLimits())
    gate.command(0.0, 0.4, 0.0)
    gate.report_stall()
    gate.command(0.1, 0.4, 0.0)
    assert gate.output(0.15).state == "stalled"
    gate.command(0.2, 0.0, 0.0)
    gate.command(0.3, 0.4, 0.0)
    assert gate.output(0.35).state == "ok"


def test_limits_must_be_positive(safety: ModuleType) -> None:
    with pytest.raises(ValueError):
        safety.SafetyLimits(max_linear_mps=0.0)
    with pytest.raises(ValueError):
        safety.SafetyLimits(command_timeout_s=-1.0)


# --------------------------------------------------------------- stall detector


def _frame(level: int) -> np.ndarray:
    return np.full((120, 160, 3), level, np.uint8)


def test_image_change_is_the_mean_grey_difference(motion: ModuleType) -> None:
    # Uniform frames 10 grey levels apart differ by exactly 10.
    assert motion.image_change(_frame(100), _frame(110)) == pytest.approx(10.0)
    assert motion.image_change(_frame(100), _frame(100)) == 0.0


def test_driven_with_a_static_view_is_a_stall_after_stall_s(motion: ModuleType) -> None:
    det = motion.StallDetector(motion.StallConfig(min_command_mps=0.1, stall_s=1.0))
    assert not det.update(0.0, 0.4, _frame(100))
    assert not det.update(0.5, 0.4, _frame(100))
    assert not det.update(0.9, 0.4, _frame(100))
    assert det.update(1.0, 0.4, _frame(100))


def test_a_changing_view_or_no_command_is_not_a_stall(motion: ModuleType) -> None:
    det = motion.StallDetector(motion.StallConfig(max_change=1.0, stall_s=1.0))
    # Each frame 5 levels brighter: the view is changing, so the robot moves.
    assert not any(det.update(0.1 * k, 0.4, _frame(50 + 5 * k)) for k in range(20))
    det.reset()
    # Standing still on purpose is not a stall.
    assert not any(det.update(0.1 * k, 0.0, _frame(100)) for k in range(30))


def test_a_slow_creep_counts_as_motion_once_it_adds_up(motion: ModuleType) -> None:
    # 0.4 levels per frame is under the 1.0 threshold frame to frame, but the
    # reference is kept, so after 3 frames the change is 1.2 and resets.
    det = motion.StallDetector(motion.StallConfig(max_change=1.0, stall_s=1.0))
    frames = [np.full((120, 160, 3), 100, np.uint8) for _ in range(30)]
    for k, f in enumerate(frames):
        f[:] = 100 + int(0.4 * k * 10) // 10  # 100,100,100,101,... a level every 2-3 frames
    assert not any(det.update(0.1 * k, 0.4, f) for k, f in enumerate(frames))


# ------------------------------------------------------------------- estimator


def test_straight_driving_integrates_distance(motion: ModuleType) -> None:
    est = motion.PlanarEstimator()
    est.reset(motion.PlanarPose(1.0, 2.0, 0.0), 0.0)
    for k in range(1, 11):
        est.update(0.1 * k, 0.5, 0.0)
    # 1 s at 0.5 m/s along +x.
    assert est.pose.x == pytest.approx(1.5)
    assert est.pose.y == pytest.approx(2.0)


def test_heading_comes_from_the_gyro(motion: ModuleType) -> None:
    est = motion.PlanarEstimator()
    est.reset(motion.PlanarPose(0.0, 0.0, 0.0), 0.0)
    # A quarter turn at pi/2 rad/s over 1 s, standing still.
    for k in range(1, 101):
        est.update(0.01 * k, 0.0, math.pi / 2.0)
    assert est.pose.yaw == pytest.approx(math.pi / 2.0)
    assert (est.pose.x, est.pose.y) == (0.0, 0.0)


def test_a_constant_arc_ends_on_the_circle(motion: ModuleType) -> None:
    # Speed 1 m/s, yaw rate 1 rad/s: radius 1 m. After pi/2 s the robot is at
    # (1, 1) facing +y, from (0, 0) facing +x.
    est = motion.PlanarEstimator()
    est.reset(motion.PlanarPose(0.0, 0.0, 0.0), 0.0)
    n = 1000
    for k in range(1, n + 1):
        est.update(k * (math.pi / 2.0) / n, 1.0, 1.0)
    assert est.pose.x == pytest.approx(1.0, abs=1e-5)
    assert est.pose.y == pytest.approx(1.0, abs=1e-5)
    assert est.pose.yaw == pytest.approx(math.pi / 2.0)


def test_a_command_from_before_a_clock_reset_is_not_used(safety: ModuleType) -> None:
    gate = safety.SafetyGate(safety.SafetyLimits())
    gate.command(8.0, 0.4, 0.0)
    # The simulation reset to t = 0: the held command is from the old world.
    assert gate.output(2.0).state == "timeout"
