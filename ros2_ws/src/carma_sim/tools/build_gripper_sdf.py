#!/usr/bin/env python3
"""Generate the parallel gripper's SDF from ``gripper_boxes.json``.

    python ros2_ws/src/carma_sim/tools/build_gripper_sdf.py            # rewrite model.sdf
    python ros2_ws/src/carma_sim/tools/build_gripper_sdf.py --check    # fail if it is stale

``model.sdf`` holds two generated regions between ``BEGIN``/``END`` markers: the
fixed body of the gripper, inside ``gripper_link``, and the two finger links
with their prismatic joints. Everything else in the file is written by hand.
Editing a generated region by hand is a mistake that ``--check`` (and
``tests/test_arm_gripper.py``) catches, so the JSON stays the single source of
truth for the shape and the SDF can always be rebuilt from it.

Frames. The boxes are in the CAD gripper assembly's frame, in millimetres. The
SDF wants them in the URDF's ``gripper_link`` frame, in metres, with the
approach direction along -z. The mapping and its assumptions are recorded in
the JSON's ``frame`` block; this script only applies it:

    x_link = x - roll_axis_x          (along the rail: the fingers open along it)
    y_link = z - roll_axis_z
    z_link = -(y + mount_plane_y)     (the fingers point along -z_link)

Mass. The CAD carries none. Each finger gets its measured volume times an
assumed PLA density, spread over its boxes uniformly, which also fixes its
centre of mass and inertia tensor. These are estimates and are labelled so in
the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "so101_follower"
_BOXES = _MODEL_DIR / "gripper_boxes.json"
_SDF = _MODEL_DIR / "model.sdf"

# Fixed-body groups drawn on gripper_link, and the finger groups with the
# joint each one rides on. outward is the sign of the opening direction in x.
_BODY_GROUPS = ("wrist_block", "base_plate", "rail")
_FINGERS = (
    # group, link, joint, outward x sign, collide bitmask
    ("finger_plus_x", "gripper_finger_plus", "gripper_plus", +1.0, "0x02"),
    ("finger_minus_x", "gripper_finger_minus", "gripper_minus", -1.0, "0x04"),
)
_BODY_BITMASK = "0x01"
_BODY_COLOUR = "<ambient>0.45 0.45 0.45 1</ambient><diffuse>0.92 0.92 0.90 1</diffuse>"
_FINGER_COLOUR = "<ambient>0.55 0.25 0.05 1</ambient><diffuse>0.95 0.55 0.15 1</diffuse>"

# Estimates, not datasheet values: the SDF says so beside them.
_FINGER_EFFORT_N = 6.0
_FINGER_SPEED_M_S = 1.0


def _metres(box_mm: list[float], frame: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size, in metres, of a CAD box expressed in the gripper_link frame."""
    x0, y0, z0, x1, y1, z1 = box_mm
    lo = np.array(
        [
            x0 - frame["roll_axis_x_mm"],
            z0 - frame["roll_axis_z_mm"],
            -(y1 + frame["mount_plane_y_mm"]),
        ],
    )
    hi = np.array(
        [
            x1 - frame["roll_axis_x_mm"],
            z1 - frame["roll_axis_z_mm"],
            -(y0 + frame["mount_plane_y_mm"]),
        ],
    )
    return (lo + hi) / 2000.0, (hi - lo) / 1000.0


def _num(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(v):.6f}".rstrip("0").rstrip(".") or "0" for v in values)


def _shape_xml(kind: str, name: str, centre: np.ndarray, size: np.ndarray, extra: str) -> str:
    geometry = f"<geometry><box><size>{_num(size)}</size></box></geometry>"
    pose = f"<pose>{_num(centre)} 0 0 0</pose>"
    if kind == "visual":
        return f'<visual name="{name}">{pose}{geometry}<material>{extra}</material></visual>'
    return (
        f'<collision name="{name}">{pose}{geometry}'
        f"<surface><contact><collide_bitmask>{extra}</collide_bitmask></contact></surface>"
        "</collision>"
    )


def _body(data: dict) -> str:
    lines = []
    for group in _BODY_GROUPS:
        for index, box in enumerate(data["groups"][group]["boxes"]):
            centre, size = _metres(box, data["frame"])
            name = f"gb_{group}_{index:02d}"
            lines.append("      " + _shape_xml("visual", name, centre, size, _BODY_COLOUR))
            lines.append("      " + _shape_xml("collision", name, centre, size, _BODY_BITMASK))
    return "\n".join(lines)


def _inertia(centres: np.ndarray, sizes: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and inertia tensor about it, for uniform boxes of total mass ``mass``."""
    volumes = sizes.prod(axis=1)
    masses = mass * volumes / volumes.sum()
    com = (masses[:, None] * centres).sum(axis=0) / mass
    tensor = np.zeros((3, 3))
    for m, c, s in zip(masses, centres, sizes, strict=True):
        own = (
            m
            / 12.0
            * np.diag(
                [s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2],
            )
        )
        r = c - com
        tensor += own + m * ((r @ r) * np.eye(3) - np.outer(r, r))
    return com, tensor


def _fingers(data: dict) -> str:
    mech = data["mechanism"]
    mass = mech["finger_volume_cm3"] * 1e-6 * mech["pla_density_kg_m3"]
    stroke = mech["stroke_mm_per_finger"] / 1000.0
    out = []
    for group, link, joint, sign, bitmask in _FINGERS:
        boxes = data["groups"][group]["boxes"]
        centres, sizes = zip(*(_metres(b, data["frame"]) for b in boxes), strict=True)
        centres, sizes = np.array(centres), np.array(sizes)
        com, tensor = _inertia(centres, sizes, mass)
        out.append(f'    <link name="{link}">')
        out.append('      <pose relative_to="gripper_link">0 0 0 0 0 0</pose>')
        out.append(
            f"      <inertial><pose>{_num(com)} 0 0 0</pose><mass>{mass:.6f}</mass>"
            f"<inertia><ixx>{tensor[0, 0]:.4e}</ixx><ixy>{tensor[0, 1]:.4e}</ixy>"
            f"<ixz>{tensor[0, 2]:.4e}</ixz><iyy>{tensor[1, 1]:.4e}</iyy>"
            f"<iyz>{tensor[1, 2]:.4e}</iyz><izz>{tensor[2, 2]:.4e}</izz></inertia></inertial>",
        )
        for index, (centre, size) in enumerate(zip(centres, sizes, strict=True)):
            name = f"gf_{group}_{index:02d}"
            out.append("      " + _shape_xml("visual", name, centre, size, _FINGER_COLOUR))
            out.append("      " + _shape_xml("collision", name, centre, size, bitmask))
        out.append("    </link>")
        out.append("")
        out.append(f'    <joint name="{joint}" type="prismatic">')
        out.append("      <parent>gripper_link</parent>")
        out.append(f"      <child>{link}</child>")
        out.append("      <axis>")
        out.append(f"        <xyz>{sign:g} 0 0</xyz>")
        out.append(
            f"        <limit><lower>0</lower><upper>{stroke:g}</upper>"
            f"<effort>{_FINGER_EFFORT_N:g}</effort>"
            f"<velocity>{_FINGER_SPEED_M_S:g}</velocity></limit>",
        )
        out.append("        <dynamics><damping>0.5</damping><friction>0.02</friction></dynamics>")
        out.append("      </axis>")
        out.append("    </joint>")
        out.append("")
    return "\n".join(out).rstrip("\n")


def _splice(text: str, name: str, body: str) -> str:
    begin = f"<!-- BEGIN generated: {name} -->"
    end = f"<!-- END generated: {name} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        msg = f"model.sdf has no {name!r} markers"
        raise SystemExit(msg)
    note = "generated by tools/build_gripper_sdf.py from gripper_boxes.json; do not edit"
    replacement = f"{begin}\n    <!-- {note} -->\n{body}\n    {end}"
    return pattern.sub(lambda _m: replacement, text)


def build() -> str:
    """The model.sdf text with both generated regions rebuilt."""
    data = json.loads(_BOXES.read_text(encoding="utf-8"))
    text = _SDF.read_text(encoding="utf-8")
    text = _splice(text, "gripper-body", _body(data))
    return _splice(text, "gripper-fingers", _fingers(data))


def main() -> int:
    """Rewrite model.sdf, or with --check report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if model.sdf is stale")
    args = parser.parse_args()

    rebuilt = build()
    current = _SDF.read_text(encoding="utf-8")
    if args.check:
        if rebuilt != current:
            print("model.sdf is stale: run tools/build_gripper_sdf.py", file=sys.stderr)
            return 1
        print("model.sdf matches gripper_boxes.json")
        return 0
    _SDF.write_text(rebuilt, encoding="utf-8")
    print(f"wrote {_SDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
