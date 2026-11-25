#!/usr/bin/env python3
"""Build the SO-101 gripper's light simulation geometry from the URDF's meshes.

    python ros2_ws/src/carma_sim/tools/build_gripper_geometry.py

The three meshes in ``models/so101_follower/meshes/`` are the URDF's, 76 146
triangles in all. Drawn and collided as they are, they cost the row-crop world
about a third of its speed with the arm on (real-time factor 0.29 against
0.45, measured 2026-11-25). So the simulation uses two lighter derivatives,
both made here and both reproducible from the vendored originals:

* Visual meshes, ``meshes/visual/<name>.stl``: the original simplified by vertex
  clustering. Vertices are snapped to a 1.5 mm grid anchored at the mesh's
  minimum corner, each cell's vertices merge into their mean, and triangles that
  collapse or repeat are dropped. About 7 000 triangles in all, and at camera
  distance indistinguishable from the original.
* Collision boxes: each part is cut into slabs along its long axis, and each
  slab becomes the box that bounds the part's surface inside it. Box contacts
  are cheap and stable for grasping; triangle meshes are neither. The boxes are
  written into ``model.sdf`` between ``BEGIN generated`` / ``END generated``
  markers, posed with the same transform the URDF gives the part's mesh.

``tests/test_arm_gripper.py`` rebuilds both and fails if either differs from
what is committed, so regenerate with this script, never by hand. Needs numpy
only.
"""

from __future__ import annotations

import itertools
import re
import struct
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "so101_follower"
MESHES = MODEL_DIR / "meshes"
VISUAL = MESHES / "visual"
SDF = MODEL_DIR / "model.sdf"

GRID_M = 0.0015
SAMPLE_EDGE_M = 0.001
# mesh -> (SDF collision name, axis to slice along in the mesh frame, number of slabs).
PLAN = {
    "wrist_roll_follower_so101_v1.stl": ("wrist_roll_follower", 2, 8),
    "moving_jaw_so101_v1.stl": ("moving_jaw", 1, 8),
    "sts3215_03a_v1.stl": ("gripper_servo", 0, 1),
}
_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL, shape (n, 3, 3), metres."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a complete binary STL")
    return np.frombuffer(data[84:], dtype=_RECORD)["vertices"].astype(np.float64)


def stl_bytes(triangles: np.ndarray) -> bytes:
    """A binary STL with unit face normals and an all-zero header, so output is reproducible."""
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    out = np.zeros(len(triangles), dtype=_RECORD)
    out["normal"], out["vertices"] = normal, triangles
    return b"\0" * 80 + struct.pack("<I", len(triangles)) + out.tobytes()


def simplify(triangles: np.ndarray, grid_m: float = GRID_M) -> np.ndarray:
    """Vertex clustering on a grid of ``grid_m`` metres; keeps triangle order and winding."""
    vertices = triangles.reshape(-1, 3)
    cells = np.floor((vertices - vertices.min(axis=0)) / grid_m).astype(np.int64)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    centres = np.zeros((len(keys), 3))
    np.add.at(centres, inverse, vertices)
    centres /= np.bincount(inverse)[:, None]
    ids = inverse.reshape(-1, 3)
    ids = ids[(ids[:, 0] != ids[:, 1]) & (ids[:, 1] != ids[:, 2]) & (ids[:, 0] != ids[:, 2])]
    _, first = np.unique(np.sort(ids, axis=1), axis=0, return_index=True)
    return centres[ids[np.sort(first)]]


def _surface_points(triangles: np.ndarray, max_edge_m: float = SAMPLE_EDGE_M) -> np.ndarray:
    """Split triangles four ways until no edge is longer than ``max_edge_m``; their corners.

    A CAD export draws a long flat face with a few long triangles, which leave no
    vertex inside most slabs, so the slabs are bounded by these samples instead.
    """
    tri = triangles
    while True:
        edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
        big = edges > max_edge_m
        if not big.any():
            return tri.reshape(-1, 3)
        a, b, c = tri[big, 0], tri[big, 1], tri[big, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        parts = [np.stack(p, axis=1) for p in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
        tri = np.concatenate([tri[~big], *parts])


def slab_boxes(triangles: np.ndarray, axis: int, slabs: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(centre, size) of each slab's bounding box, in the mesh frame, metres."""
    points = _surface_points(triangles)
    edges = np.linspace(points[:, axis].min(), points[:, axis].max(), slabs + 1)
    boxes = []
    for low, high in itertools.pairwise(edges):
        inside = points[(points[:, axis] >= low) & (points[:, axis] <= high)]
        lo, hi = inside.min(axis=0), inside.max(axis=0)
        boxes.append(((lo + hi) / 2, hi - lo))
    return boxes


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _visual_pose(sdf: str, mesh: str) -> tuple[np.ndarray, str]:
    """The pose of the visual that draws ``mesh``: the URDF's transform for that part."""
    match = re.search(
        rf"<visual name=\"\w+\"><pose>([^<]*)</pose>\s*<geometry><mesh><uri>"
        rf"model://so101_follower/meshes/visual/{re.escape(mesh)}</uri>",
        sdf,
    )
    if match is None:
        raise ValueError(f"model.sdf has no visual drawing {mesh}")
    values = match.group(1).split()
    return np.array([float(v) for v in values[:3]]), " ".join(values[3:])


def collision_block(sdf: str, mesh: str, triangles: np.ndarray, indent: str = "      ") -> str:
    """The generated SDF collision elements for one part."""
    name, axis, slabs = PLAN[mesh]
    origin, rpy = _visual_pose(sdf, mesh)
    rotation = _rotation(*(float(v) for v in rpy.split()))
    lines = [f"{indent}<!-- BEGIN generated: {name} collision -->"]
    for i, (centre, size) in enumerate(slab_boxes(triangles, axis, slabs)):
        x, y, z = origin + rotation @ centre
        sx, sy, sz = size
        lines.append(
            f'{indent}<collision name="{name}_{i}"><pose>{x:.5f} {y:.5f} {z:.5f} {rpy}</pose>'
            f"<geometry><box><size>{sx:.5f} {sy:.5f} {sz:.5f}</size></box></geometry></collision>"
        )
    lines.append(f"{indent}<!-- END generated: {name} collision -->")
    return "\n".join(lines)


def render_sdf(sdf: str, originals: dict[str, np.ndarray]) -> str:
    """``sdf`` with every generated collision region rebuilt."""
    for mesh, (name, _axis, _slabs) in PLAN.items():
        pattern = re.compile(
            rf"( *)<!-- BEGIN generated: {name} collision -->.*?"
            rf"<!-- END generated: {name} collision -->",
            flags=re.S,
        )
        match = pattern.search(sdf)
        if match is None:
            raise ValueError(f"model.sdf has no generated region for {name}")
        block = collision_block(sdf, mesh, originals[mesh], match.group(1))
        sdf = sdf[: match.start()] + block + sdf[match.end() :]
    return sdf


def main() -> None:
    """Write the visual meshes and the collision boxes."""
    originals = {mesh: read_stl(MESHES / mesh) for mesh in PLAN}
    VISUAL.mkdir(exist_ok=True)
    for mesh, triangles in originals.items():
        light = simplify(triangles)
        (VISUAL / mesh).write_bytes(stl_bytes(light))
        print(f"{mesh}: {len(triangles)} -> {len(light)} triangles")
    sdf = SDF.read_text(encoding="utf-8")
    SDF.write_text(render_sdf(sdf, originals), encoding="utf-8")
    print(f"collision boxes written to {SDF.name}")


if __name__ == "__main__":
    main()
