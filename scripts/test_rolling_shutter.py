"""Decide whether a camera has a rolling or a global shutter.

A rolling shutter exposes the sensor row by row, so a straight vertical edge
moving sideways is captured at a different horizontal position in each row and
comes out slanted. A global shutter exposes every row at once and keeps it
vertical however fast it moves. The test is that slant.

    python scripts/test_rolling_shutter.py static --camera 1
    # then wave a straight vertical object across the view, fast
    python scripts/test_rolling_shutter.py moving --camera 1
    python scripts/test_rolling_shutter.py report

This matters more than depth accuracy for a robot that moves: a rolling
shutter breaks the assumption that one image is one instant, which is what
visual-inertial odometry and SLAM are built on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_NEAR_VERTICAL_DEG = 25.0
_MIN_EDGE_FRACTION = 0.25


def _open(camera: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        msg = f"camera {camera} could not be opened"
        raise SystemExit(msg)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _edge_tilts(grey: np.ndarray) -> list[float]:
    """Signed tilt from vertical, in degrees, of the long near-vertical edges.

    Positive means the top leans right of the bottom. A rolling shutter gives
    a consistent sign for a given direction of motion; noise does not.
    """
    height = grey.shape[0]
    edges = cv2.Canny(grey, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=60,
        minLineLength=int(height * _MIN_EDGE_FRACTION),
        maxLineGap=8,
    )
    if lines is None:
        return []

    tilts: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dy) < 1e-6:
            continue
        # Angle away from vertical. Normalise so the line always runs downward.
        if dy < 0:
            dx, dy = -dx, -dy
        tilt = np.degrees(np.arctan2(dx, dy))
        if abs(tilt) <= _NEAR_VERTICAL_DEG:
            tilts.append(float(tilt))
    return tilts


def _collect(
    camera: int,
    width: int,
    height: int,
    frames: int,
    min_motion: float = 0.0,
) -> dict[str, object]:
    """Capture frames and summarise the tilt of their near-vertical edges.

    Frames whose mean absolute difference from the previous frame is below
    ``min_motion`` are skipped. Without that gate the moving condition
    silently measures the static background: a fast subject motion-blurs, its
    edges vanish from the edge detector, and what survives is the stationary
    scene, which cannot show readout skew whatever the shutter does.
    """
    cap = _open(camera, width, height)
    for _ in range(15):
        cap.read()

    all_tilts: list[float] = []
    per_frame: list[float] = []
    motions: list[float] = []
    skipped = 0
    previous: np.ndarray | None = None
    attempts = 0
    while len(per_frame) < frames and attempts < frames * 6:
        attempts += 1
        ok, frame = cap.read()
        if not ok:
            break
        # Left eye only: the two halves of a stereo frame read out separately.
        grey = cv2.cvtColor(frame[:, : frame.shape[1] // 2], cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if previous is not None:
            motion = float(cv2.absdiff(grey, previous).mean())
        previous = grey
        if min_motion > 0.0 and motion < min_motion:
            skipped += 1
            continue
        motions.append(motion)
        tilts = _edge_tilts(grey)
        if tilts:
            all_tilts.extend(tilts)
            per_frame.append(float(np.median(tilts)))
    cap.release()

    if not all_tilts:
        msg = "no long vertical edges found; aim at a doorway, a table leg or a held ruler"
        raise SystemExit(msg)
    array = np.array(all_tilts)
    return {
        "edges": int(array.size),
        "frames_with_edges": len(per_frame),
        "mean_interframe_motion": float(np.mean(motions)) if motions else 0.0,
        "frames_skipped_as_static": skipped,
        "tilt_spread_deg": float(np.percentile(array, 90) - np.percentile(array, 10)),
        "median_tilt_deg": float(np.median(array)),
        "mean_abs_tilt_deg": float(np.mean(np.abs(array))),
        "p95_abs_tilt_deg": float(np.percentile(np.abs(array), 95)),
        "per_frame_median_deg": per_frame,
    }


def main() -> int:
    """Capture a condition, or compare the two."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("static", "moving", "report"))
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.0,
        help="skip frames whose mean inter-frame difference is below this (use ~4 when moving)",
    )
    parser.add_argument("--out", type=Path, default=Path("shutter_test.json"))
    args = parser.parse_args()

    stored: dict[str, object] = {}
    if args.out.exists():
        stored = json.loads(args.out.read_text(encoding="utf-8"))

    if args.action in ("static", "moving"):
        result = _collect(args.camera, args.width, args.height, args.frames, args.min_motion)
        stored[args.action] = result
        args.out.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        print(
            f"{args.action}: {result['edges']} edges over {result['frames_with_edges']} frames "
            f"(skipped {result['frames_skipped_as_static']} as too static), "
            f"motion {result['mean_interframe_motion']:.1f}, "
            f"median tilt {result['median_tilt_deg']:+.2f} deg, "
            f"mean |tilt| {result['mean_abs_tilt_deg']:.2f} deg, "
            f"spread {result['tilt_spread_deg']:.2f} deg"
        )
        return 0

    if "static" not in stored or "moving" not in stored:
        print("need both a static and a moving capture first", file=sys.stderr)
        return 1

    static = stored["static"]
    moving = stored["moving"]
    assert isinstance(static, dict)
    assert isinstance(moving, dict)
    rest = float(static["mean_abs_tilt_deg"])
    waved = float(moving["mean_abs_tilt_deg"])
    excess = waved - rest
    rest_spread = float(static.get("tilt_spread_deg", 0.0))
    waved_spread = float(moving.get("tilt_spread_deg", 0.0))
    motion = float(moving.get("mean_interframe_motion", 0.0))

    print(f"still  : mean |tilt| {rest:5.2f} deg, spread {rest_spread:5.2f} deg")
    print(
        f"moving : mean |tilt| {waved:5.2f} deg, spread {waved_spread:5.2f} deg "
        f"(inter-frame motion {motion:.1f})"
    )
    print(f"excess : {excess:+5.2f} deg mean, {waved_spread - rest_spread:+5.2f} deg spread\n")

    if motion < 2.0:
        print("INCONCLUSIVE: the moving capture barely moved, so nothing was tested.")
        print("Re-run the moving pass while panning the camera quickly.")
        return 1

    # A global shutter cannot turn motion into slant, so any consistent excess
    # is readout skew. Below half a degree the test cannot tell them apart.
    if excess < 0.5:
        print("GLOBAL SHUTTER (or a very fast readout): motion adds no measurable slant.")
        print("Tight visual-inertial odometry is on the table.")
    elif excess < 2.0:
        print("PROBABLY ROLLING, with a fast readout. Some skew under motion.")
        print("Usable for VIO if the robot moves slowly; verify before relying on it.")
    else:
        print("ROLLING SHUTTER: motion clearly skews the image.")
        print("One frame is not one instant, so tight VIO needs a rolling-shutter")
        print("model or a different camera. Loose coupling (attitude, gravity) is fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
