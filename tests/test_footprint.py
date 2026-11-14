"""The bridge's plan-view collision test, on a hand-written world.

Loaded by path because the bridge package lives in the ROS workspace, not in
``src/``; it has no ROS imports, so it runs here.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_WORLD = """<sdf version='1.10'><world name='w'>
  <model name='ground'><link name='l'><collision name='c'>
    <geometry><plane><normal>0 0 1</normal></plane></geometry></collision></link></model>
  <model name='bot'>
    <link name='body'><collision name='c'><pose>0.2 0 0.1 0 0 0</pose>
      <geometry><box><size>0.6 0.4 0.2</size></box></geometry></collision></link>
    <link name='wheel'><pose>0 0.3 0 -1.5707963 0 0</pose><collision name='c'>
      <geometry><cylinder><radius>0.2</radius><length>0.1</length></cylinder></geometry>
    </collision></link>
  </model>
  <model name='row'><pose>5 1 0 0 0 0</pose><link name='seg'><collision name='c'>
    <geometry><box><size>2 0.4 0.5</size></box></geometry></collision></link></model>
  <model name='post'><pose>3 -1 0 0 0 0</pose><link name='l'><collision name='c'>
    <geometry><cylinder><radius>0.1</radius><length>1</length></cylinder></geometry>
    </collision></link></model>
</world></sdf>"""


def _footprint() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "ros2_ws/src/carma_bridge/carma_bridge/footprint.py"
    spec = importlib.util.spec_from_file_location("bridge_footprint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def test_shapes_are_read_and_the_ground_is_left_out() -> None:
    f = _footprint()
    robot, obstacles = f.parse_world(_WORLD, "bot", "ground")
    assert [s.name for s in obstacles] == ["row::seg::c", "post::l::c"]
    wheel = next(s for s in robot if s.name == "bot::wheel::c")
    # A wheel on its side: diameter along x (half 0.2), width along y (half 0.05).
    assert (wheel.kind, wheel.hx, wheel.hy, wheel.y) == ("rect", 0.2, 0.05, pytest.approx(0.3))
    post = obstacles[1]
    assert (post.kind, post.hx, post.x, post.y) == ("circle", 0.1, 3.0, -1.0)


def test_hand_placed_poses_hit_or_clear() -> None:
    f = _footprint()
    robot, obstacles = f.parse_world(_WORLD, "bot", "ground")
    assert f.first_hit(robot, obstacles, 0.0, 0.0, 0.0) is None
    # Row spans y 0.8..1.2; the wheel's outer edge is at y + 0.35.
    assert f.first_hit(robot, obstacles, 5.0, 0.40, 0.0) is None  # edge at 0.75
    assert f.first_hit(robot, obstacles, 5.0, 0.50, 0.0) == "row::seg::c"  # edge at 0.85
    # Post at (3, -1), radius 0.1, so its top edge is y = -0.9. The body spans
    # y +/-0.2 about the robot: at y = -0.65 its edge is -0.85 (gap 0.05), at
    # y = -0.75 it is -0.95 (overlap 0.05).
    assert f.first_hit(robot, obstacles, 3.0, -0.65, 0.0) is None
    assert f.first_hit(robot, obstacles, 3.0, -0.75, 0.0) == "post::l::c"
    # Turned 90 degrees left at the post: the body now spans y -0.1..0.5 from the origin.
    assert f.first_hit(robot, obstacles, 3.0, -1.4, math.pi / 2) == "post::l::c"


def test_a_world_without_the_robot_is_an_error() -> None:
    f = _footprint()
    with pytest.raises(ValueError, match="robot"):
        f.parse_world(_WORLD, "nobody", "ground")
