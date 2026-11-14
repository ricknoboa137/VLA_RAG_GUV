"""Plan-view collision test between the robot and the world, from the world's own SDF.

Gazebo's contact sensors cannot be used here: the bridge resets the world with
"reset all" before every episode, and that tears down the contact system, so
after the first episode no contact is ever published (checked on gz-sim 8: the
``/carma/contact`` publisher was present before a reset and gone after it).

Instead the bridge reads the world as Gazebo holds it (the
``generate_world_sdf`` service), keeps every collision shape in plan view, and
tests the robot's collision shapes, placed at its ground-truth pose, against
everything except the ground. Shapes are reduced to oriented rectangles and
circles in the x-y plane:

* box -> rectangle of its x and y size, rotated by the composed yaw;
* sphere -> circle of its radius;
* cylinder -> circle when its axis is vertical, rectangle (diameter by length)
  when it lies on its side (a wheel).

Heights are ignored: every obstacle in the row-crop world is taller than the
robot's ground clearance. The front wheels are taken at zero steer, which
misplaces their corners by at most a few centimetres.

No ROS and no Gazebo imports, so the unit suite tests it on a plain machine.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_UPRIGHT_TOL_RAD = 0.1


@dataclass(frozen=True)
class Shape:
    """One plan-view collision shape.

    Attributes:
        name: Scoped name, ``model::link::collision``.
        kind: ``"rect"`` or ``"circle"``.
        x: Centre x in metres, in the frame the shape was built in.
        y: Centre y in metres.
        yaw: Rotation in radians (rectangles only).
        hx: Half length along the shape's own x, or the radius for a circle.
        hy: Half width along the shape's own y (rectangles only).
    """

    name: str
    kind: str
    x: float
    y: float
    yaw: float
    hx: float
    hy: float = 0.0


def _pose(element: ET.Element | None) -> tuple[float, float, float, float, float, float]:
    if element is None or element.find("pose") is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    text = (element.find("pose").text or "").split()  # type: ignore[union-attr]
    values = [float(v) for v in text] + [0.0] * (6 - len(text))
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def _compose(
    parent: tuple[float, float, float], child: tuple[float, float, float]
) -> tuple[float, float, float]:
    px, py, pyaw = parent
    cx, cy, cyaw = child
    c, s = math.cos(pyaw), math.sin(pyaw)
    return (px + c * cx - s * cy, py + s * cx + c * cy, pyaw + cyaw)


def _shape(
    name: str, where: tuple[float, float, float], roll: float, pitch: float, geometry: ET.Element
) -> Shape | None:
    x, y, yaw = where
    box = geometry.find("box/size")
    if box is not None and box.text:
        sx, sy, _sz = (float(v) for v in box.text.split())
        return Shape(name, "rect", x, y, yaw, sx / 2.0, sy / 2.0)
    sphere = geometry.find("sphere/radius")
    if sphere is not None and sphere.text:
        return Shape(name, "circle", x, y, 0.0, float(sphere.text))
    radius = geometry.find("cylinder/radius")
    length = geometry.find("cylinder/length")
    if radius is not None and radius.text and length is not None and length.text:
        r, h = float(radius.text), float(length.text)
        if abs(roll) < _UPRIGHT_TOL_RAD and abs(pitch) < _UPRIGHT_TOL_RAD:
            return Shape(name, "circle", x, y, 0.0, r)
        if abs(abs(roll) - math.pi / 2.0) < _UPRIGHT_TOL_RAD:
            # Axis along the link's y: diameter along x, length along y.
            return Shape(name, "rect", x, y, yaw, r, h / 2.0)
        if abs(abs(pitch) - math.pi / 2.0) < _UPRIGHT_TOL_RAD:
            return Shape(name, "rect", x, y, yaw, h / 2.0, r)
    return None  # planes and meshes: the ground, or nothing a robot can hit in plan view


def _model_shapes(model: ET.Element, frame: tuple[float, float, float]) -> list[Shape]:
    shapes: list[Shape] = []
    name = model.get("name", "")
    for link in model.findall("link"):
        lx, ly, _lz, lroll, lpitch, lyaw = _pose(link)
        link_frame = _compose(frame, (lx, ly, lyaw))
        for collision in link.findall("collision"):
            cx, cy, _cz, croll, cpitch, cyaw = _pose(collision)
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            shape = _shape(
                f"{name}::{link.get('name')}::{collision.get('name')}",
                _compose(link_frame, (cx, cy, cyaw)),
                # A wheel's sideways rotation may sit on the link or on the
                # collision; only the sum matters for its plan-view shape.
                lroll + croll,
                lpitch + cpitch,
                geometry,
            )
            if shape is not None:
                shapes.append(shape)
    return shapes


def parse_world(sdf: str, robot: str, ground: str) -> tuple[list[Shape], list[Shape]]:
    """Split a world SDF into the robot's shapes (robot frame) and the obstacles (world frame).

    Args:
        sdf: The world as Gazebo generated it, includes expanded.
        robot: Model name of the robot.
        ground: Model name of the ground, which is never an obstacle.

    Returns:
        ``(robot_shapes, obstacles)``.
    """
    world = ET.fromstring(sdf).find("world")
    if world is None:
        raise ValueError("the SDF has no <world>")
    robot_shapes: list[Shape] = []
    obstacles: list[Shape] = []
    for model in world.findall("model"):
        name = model.get("name", "")
        if name == ground:
            continue
        if name == robot:
            robot_shapes = _model_shapes(model, (0.0, 0.0, 0.0))
        else:
            mx, my, _mz, _mr, _mp, myaw = _pose(model)
            obstacles += _model_shapes(model, (mx, my, myaw))
    if not robot_shapes:
        raise ValueError(f"no collision shapes found for the robot {robot!r}")
    return robot_shapes, obstacles


def place(shapes: list[Shape], x: float, y: float, yaw: float) -> list[Shape]:
    """The robot's shapes moved to the world pose ``(x, y, yaw)``."""
    placed = []
    for s in shapes:
        sx, sy, syaw = _compose((x, y, yaw), (s.x, s.y, s.yaw))
        placed.append(Shape(s.name, s.kind, sx, sy, syaw, s.hx, s.hy))
    return placed


def _corners(r: Shape) -> list[tuple[float, float]]:
    c, s = math.cos(r.yaw), math.sin(r.yaw)
    return [
        (r.x + c * dx - s * dy, r.y + s * dx + c * dy)
        for dx, dy in ((r.hx, r.hy), (-r.hx, r.hy), (-r.hx, -r.hy), (r.hx, -r.hy))
    ]


def _rects_overlap(a: Shape, b: Shape) -> bool:
    """Separating-axis test for two oriented rectangles."""
    ca, cb = _corners(a), _corners(b)
    for shape in (a, b):
        for angle in (shape.yaw, shape.yaw + math.pi / 2.0):
            ax, ay = math.cos(angle), math.sin(angle)
            pa = [ax * px + ay * py for px, py in ca]
            pb = [ax * px + ay * py for px, py in cb]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _rect_circle_overlap(r: Shape, c: Shape) -> bool:
    cos, sin = math.cos(-r.yaw), math.sin(-r.yaw)
    dx, dy = c.x - r.x, c.y - r.y
    lx, ly = cos * dx - sin * dy, sin * dx + cos * dy
    nx, ny = max(-r.hx, min(r.hx, lx)), max(-r.hy, min(r.hy, ly))
    return (lx - nx) ** 2 + (ly - ny) ** 2 < c.hx**2


def overlap(a: Shape, b: Shape) -> bool:
    """True when two plan-view shapes intersect (touching edges do not count)."""
    if a.kind == "rect" and b.kind == "rect":
        return _rects_overlap(a, b)
    if a.kind == "circle" and b.kind == "circle":
        return math.hypot(a.x - b.x, a.y - b.y) < a.hx + b.hx
    rect, circle = (a, b) if a.kind == "rect" else (b, a)
    return _rect_circle_overlap(rect, circle)


def first_hit(
    robot: list[Shape], obstacles: list[Shape], x: float, y: float, yaw: float
) -> str | None:
    """Name of the first obstacle the robot overlaps at ``(x, y, yaw)``, or None."""
    for part in place(robot, x, y, yaw):
        reach = math.hypot(part.hx, part.hy)
        for obstacle in obstacles:
            far = reach + math.hypot(obstacle.hx, obstacle.hy)
            if abs(part.x - obstacle.x) > far or abs(part.y - obstacle.y) > far:
                continue
            if overlap(part, obstacle):
                return obstacle.name
    return None
