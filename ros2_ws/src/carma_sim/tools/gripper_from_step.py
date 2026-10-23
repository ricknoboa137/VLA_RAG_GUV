#!/usr/bin/env python3
"""Sample the project's parallel gripper from its CAD into ``gripper_boxes.json``.

    pip install cadquery-ocp            # OpenCASCADE bindings; NOT a project dependency
    python ros2_ws/src/carma_sim/tools/gripper_from_step.py \\
        --step "SO101 Assembly new2.stp"
    python ros2_ws/src/carma_sim/tools/build_gripper_sdf.py

This is a development tool. The STEP file is not in the repository (it is the
project's own CAD, kept with the platform design, and its licence is
unverified), and cadquery-ocp is not pinned in ``pyproject.toml`` because
nothing in ``src/`` or the test suite imports it. It exists so the committed
JSON is not an opaque blob: anyone with the same CAD can regenerate it and see
the result is identical.

What it does, in order:

1. Reads the STEP assembly and finds the ``gripper v3`` sub-assembly.
2. Places every part of it in that sub-assembly's own frame.
3. Triangulates the parts the simulation draws: the wrist block, base plate and
   rail (fixed), and the two racks and two fingertips (moving).
4. Voxelises each solid by even-odd ray casting through its triangles along x,
   on a grid aligned to the solid's own bounding box, so a face on the bounding
   box lands exactly on a cell boundary. That matters most for the fingertips,
   whose inner faces set the grip gap, and which therefore use 1 mm cells where
   the racks and fixed body use 2 to 3 mm.
5. Merges the filled cells greedily into axis-aligned boxes.

Only the ``groups`` entry of the JSON is rewritten. The provenance, frame
mapping and mechanism notes are written by hand and kept as they are.

Every ray column crossing the mesh an even number of times is the check that a
solid is watertight; the tool refuses to continue if any is odd, because a
leak would fill or empty cells silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.collections import Sequence_TDF_Label
    from OCP.gp import gp_Trsf
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.exit(
        f"{exc}\nThis tool needs OpenCASCADE's Python bindings: pip install cadquery-ocp "
        "(OCP 8 module layout; older releases name some modules differently)."
    )

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models/so101_follower/gripper_boxes.json"
_ASSEMBLY = "gripper v3"

# Cell size per solid, in millimetres. Fingertips stay at 1 mm.
_RAIL, _BLOCK, _PLATE = 2.0, 3.0, 3.0
_RACK, _FINGERTIP = 2.0, 1.0

# STEP part names, as the file spells them.
_N_BLOCK = "blocco_giunto_polso"
_N_PLATE = "gripper_base_plate v1"
_N_RAIL = "gripper_rail_base v1"
_N_RACK = "gripper_finger v1"
_N_TIP = "dito1"


# ----------------------------------------------------------------------- STEP


def _load(path: Path) -> TDocStd_Document:
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        sys.exit(f"could not read {path}")
    reader.Transfer(doc)
    return doc


def _name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else "?"


def _matrix(location: TopLoc_Location) -> np.ndarray:
    trsf = location.Transformation()
    out = np.eye(4)
    for r in range(3):
        for c in range(4):
            out[r, c] = trsf.Value(r + 1, c + 1)
    return out


def _walk(label: TDF_Label, world: np.ndarray, found: list) -> None:
    """Depth-first over the assembly, recording ``(name, world, shape)`` for every leaf."""
    ref = TDF_Label()
    target = ref if XCAFDoc_ShapeTool.GetReferredShape_s(label, ref) else label
    children = Sequence_TDF_Label()
    if XCAFDoc_ShapeTool.IsAssembly_s(target):
        XCAFDoc_ShapeTool.GetComponents_s(target, children)
    if children.Length() == 0:
        shape = TopoDS_Shape()
        XCAFDoc_ShapeTool.GetShape_s(target, shape)
        found.append((_name(target), world, shape))
        return
    found.append((_name(target), world, None))
    for i in range(1, children.Length() + 1):
        child = children.Value(i)
        _walk(child, world @ _matrix(XCAFDoc_ShapeTool.GetLocation_s(child)), found)


def _moved(shape: TopoDS_Shape, matrix: np.ndarray) -> TopoDS_Shape:
    trsf = gp_Trsf()
    trsf.SetValues(*[float(matrix[r, c]) for r in range(3) for c in range(4)])
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _bbox(shape: TopoDS_Shape) -> np.ndarray:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)  # exact bounds, not the loose default
    lo, hi = box.CornerMin(), box.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()])


def _parts(path: Path) -> dict[str, list[TopoDS_Shape]]:
    """Leaf parts of the gripper sub-assembly, placed in that sub-assembly's own frame."""
    doc = _load(path)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free = Sequence_TDF_Label()
    tool.GetFreeShapes(free)
    found: list = []
    _walk(free.Value(1), np.eye(4), found)

    # The sub-assembly's own placement is the frame everything is expressed in.
    origin = next((w for n, w, s in found if s is None and n.lower().startswith(_ASSEMBLY)), None)
    if origin is None:
        sys.exit(f"no assembly named {_ASSEMBLY!r} in {path}")
    inverse = np.linalg.inv(origin)

    # Everything after the sub-assembly node in the walk and below it belongs to it; the
    # leaf names are unique enough within this file that the name filter is sufficient.
    wanted = {_N_BLOCK, _N_PLATE, _N_RAIL, _N_RACK, _N_TIP}
    out: dict[str, list[TopoDS_Shape]] = {}
    for name, world, shape in found:
        if shape is not None and name in wanted:
            out.setdefault(name, []).append(_moved(shape, inverse @ world))
    return out


# --------------------------------------------------------------------- voxels


def _triangles(shape: TopoDS_Shape) -> np.ndarray:
    BRepMesh_IncrementalMesh(shape, 0.04, False, 0.2, True)
    tris = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            trsf = location.Transformation()
            nodes = np.array(
                [
                    (p.X(), p.Y(), p.Z())
                    for p in (mesh.Node(i).Transformed(trsf) for i in range(1, mesh.NbNodes() + 1))
                ],
            )
            for i in range(1, mesh.NbTriangles() + 1):
                a, b, c = mesh.Triangle(i).Get()
                tris.append(nodes[[a - 1, b - 1, c - 1]])
        explorer.Next()
    return np.array(tris)


def _voxelise(tris: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean grid of the solid and its origin, by even-odd rays along x."""
    points = tris.reshape(-1, 3)
    lo, hi = points.min(axis=0), points.max(axis=0)
    n = np.maximum(1, np.ceil((hi - lo) / size - 1e-9).astype(int))
    # Rays through the middle of each (y, z) cell, nudged off any mesh edge.
    ys = lo[1] + (np.arange(n[1]) + 0.5) * size + 1.7e-4
    zs = lo[2] + (np.arange(n[2]) + 0.5) * size + 2.3e-4
    hits: dict[tuple[int, int], list[float]] = {}
    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in tris:
        det = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
        if abs(det) < 1e-12:
            continue  # parallel to the ray
        j0 = max(0, int(np.floor((min(y0, y1, y2) - lo[1]) / size - 0.5)))
        j1 = min(n[1] - 1, int(np.ceil((max(y0, y1, y2) - lo[1]) / size - 0.5)))
        k0 = max(0, int(np.floor((min(z0, z1, z2) - lo[2]) / size - 0.5)))
        k1 = min(n[2] - 1, int(np.ceil((max(z0, z1, z2) - lo[2]) / size - 0.5)))
        if j1 < j0 or k1 < k0:
            continue
        yy, zz = np.meshgrid(ys[j0 : j1 + 1], zs[k0 : k1 + 1], indexing="ij")
        a = ((yy - y0) * (z2 - z0) - (y2 - y0) * (zz - z0)) / det
        b = ((y1 - y0) * (zz - z0) - (yy - y0) * (z1 - z0)) / det
        inside = (a >= 0) & (b >= 0) & (a + b <= 1)
        xs = x0 + a * (x1 - x0) + b * (x2 - x0)
        for jj, kk in zip(*np.nonzero(inside), strict=True):
            hits.setdefault((j0 + int(jj), k0 + int(kk)), []).append(float(xs[jj, kk]))

    grid = np.zeros(n, dtype=bool)
    centres = lo[0] + (np.arange(n[0]) + 0.5) * size
    for (j, k), crossings in hits.items():
        if len(crossings) % 2:
            sys.exit(
                "a ray column crossed the mesh an odd number of times: the solid is not watertight"
            )
        ordered = np.sort(crossings)
        for enter, leave in zip(ordered[0::2], ordered[1::2], strict=True):
            grid[(centres >= enter) & (centres <= leave), j, k] = True
    return grid, lo


def _merge(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Greedy decomposition into runs along x, then y, then z."""
    seen = np.zeros_like(grid)
    out = []
    nx, ny, nz = grid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not grid[i, j, k] or seen[i, j, k]:
                    continue
                i2 = i
                while i2 + 1 < nx and grid[i2 + 1, j, k] and not seen[i2 + 1, j, k]:
                    i2 += 1
                j2 = j
                while (
                    j2 + 1 < ny
                    and grid[i : i2 + 1, j2 + 1, k].all()
                    and not seen[i : i2 + 1, j2 + 1, k].any()
                ):
                    j2 += 1
                k2 = k
                while (
                    k2 + 1 < nz
                    and grid[i : i2 + 1, j : j2 + 1, k2 + 1].all()
                    and not seen[i : i2 + 1, j : j2 + 1, k2 + 1].any()
                ):
                    k2 += 1
                seen[i : i2 + 1, j : j2 + 1, k : k2 + 1] = True
                out.append((i, j, k, i2, j2, k2))
    return out


def _boxes(shapes: list[tuple[TopoDS_Shape, float]]) -> dict[str, object]:
    boxes: list[list[float]] = []
    filled = 0
    for shape, size in shapes:
        grid, lo = _voxelise(_triangles(shape), size)
        filled += int(grid.sum())
        for i, j, k, i2, j2, k2 in _merge(grid):
            a = lo + np.array([i, j, k]) * size
            b = lo + np.array([i2 + 1, j2 + 1, k2 + 1]) * size
            boxes.append([round(float(v), 3) for v in (*a, *b)])
    return {"filled": filled, "boxes": boxes}


def _by_x(shapes: list[TopoDS_Shape], key: int) -> list[TopoDS_Shape]:
    """Order two parts by one bounding-box coordinate, smallest first."""
    return sorted(shapes, key=lambda s: _bbox(s)[key])


def build(step: Path) -> dict[str, object]:
    """The ``groups`` block for ``gripper_boxes.json``."""
    parts = _parts(step)
    racks = _by_x(parts[_N_RACK], 3)  # by x max: the +x rack ends further out
    tips = _by_x(parts[_N_TIP], 0)  # by x min: the +x fingertip starts further out
    if len(racks) != 2 or len(tips) != 2:
        sys.exit(f"expected two racks and two fingertips, found {len(racks)} and {len(tips)}")
    return {
        "wrist_block": _boxes([(parts[_N_BLOCK][0], _BLOCK)]),
        "base_plate": _boxes([(parts[_N_PLATE][0], _PLATE)]),
        "rail": _boxes([(parts[_N_RAIL][0], _RAIL)]),
        "finger_plus_x": _boxes([(racks[1], _RACK), (tips[1], _FINGERTIP)]),
        "finger_minus_x": _boxes([(racks[0], _RACK), (tips[0], _FINGERTIP)]),
    }


def main() -> int:
    """Regenerate the groups block, or with --check say whether it would change."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=Path, required=True, help="SO101 Assembly new2.stp")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if the JSON would change")
    args = parser.parse_args()

    data = json.loads(args.out.read_text(encoding="utf-8"))
    groups = build(args.step)
    for name, group in groups.items():
        print(f"{name:16s} {len(group['boxes']):4d} boxes")  # type: ignore[arg-type]

    if args.check:
        if groups != data["groups"]:
            print("gripper_boxes.json would change", file=sys.stderr)
            return 1
        print("gripper_boxes.json matches the CAD")
        return 0
    data["groups"] = groups
    args.out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
