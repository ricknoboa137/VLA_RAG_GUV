"""Depth map and point cloud from a side-by-side stereo camera.

    pip install -e ".[stream]"
    python scripts/stereo_depth.py --camera 1 --width 1280 --height 480

Writes a depth preview and a coloured ``.ply`` point cloud that MeshLab or
CloudCompare will open. ``--preview`` shows the disparity live instead.

Without ``--calibration`` the distances come from the baseline and field of
view alone, assuming the sensors are perfectly rectified. That is good enough
to see the shape of a scene and to tell near from far, but the absolute
numbers will be off, more so towards the edges of the frame. Run
``scripts/calibrate_stereo.py`` for measurements worth trusting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from carma.perception.stereo_depth import (
    DisparityParams,
    StereoCalibration,
    approximate_calibration,
    calibration_from_known_distance,
    colourise_depth,
    depth_from_disparity,
    disparity_map,
    looks_side_by_side,
    median_disparity,
    point_cloud,
    rgbd,
    split_side_by_side,
    write_ply,
)


def _open(index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        msg = f"camera {index} could not be opened"
        raise SystemExit(msg)
    if width and height:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> int:
    """Capture one frame (or stream), and write depth and a point cloud."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=1, help="side-by-side stereo camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.06,
        help="distance between the lens centres in metres (default 0.06)",
    )
    parser.add_argument("--fov", type=float, default=60.0, help="horizontal field of view, degrees")
    parser.add_argument("--calibration", type=Path, default=None, help="calibration JSON")
    parser.add_argument("--out", type=Path, default=Path("stereo_out"))
    parser.add_argument("--max-depth", type=float, default=20.0, help="drop points beyond, metres")
    parser.add_argument("--num-disparities", type=int, default=96, help="multiple of 16")
    parser.add_argument("--block-size", type=int, default=7, help="odd")
    parser.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="start of the search; negative for toed-in sensors (see the fit output)",
    )
    parser.add_argument(
        "--calibrate-at",
        type=float,
        default=None,
        help="metres to an object filling the centre of the view; solves the focal length",
    )
    parser.add_argument(
        "--save-calibration", type=Path, default=None, help="write the calibration JSON here"
    )
    parser.add_argument("--preview", action="store_true", help="live window instead of one shot")
    args = parser.parse_args()

    cap = _open(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        print("camera read failed", file=sys.stderr)
        return 1
    if not looks_side_by_side(frame):
        cap.release()
        h, w = frame.shape[:2]
        print(f"{w}x{h} is not a side-by-side pair (aspect {w / h:.2f})", file=sys.stderr)
        return 1

    left, right = split_side_by_side(frame)
    eye_size = (left.shape[1], left.shape[0])
    if args.calibration is not None:
        calibration = StereoCalibration.from_json(args.calibration)
        print(f"calibration from {args.calibration}")
    else:
        calibration = approximate_calibration(eye_size, args.baseline, args.fov)
        print(
            f"approximate calibration: baseline {args.baseline * 1000:.0f} mm, "
            f"fov {args.fov:.0f}deg -> focal {calibration.focal_px:.0f} px. "
            "Distances are indicative only.",
        )
    params = DisparityParams(
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        min_disparity=args.min_disparity,
    )
    print(f"per eye {eye_size[0]}x{eye_size[1]}")

    if args.calibrate_at is not None:
        for _ in range(10):
            ok, frame = cap.read()
        if not ok:
            cap.release()
            print("camera read failed during calibration", file=sys.stderr)
            return 1
        left, right = split_side_by_side(frame)
        centre = median_disparity(disparity_map(left, right, params))
        if not np.isfinite(centre):
            cap.release()
            print(
                "nothing matched in the centre of the view; point the camera at a "
                "textured object at the measured distance",
                file=sys.stderr,
            )
            return 1
        calibration = calibration_from_known_distance(
            eye_size, args.baseline, args.calibrate_at, centre
        )
        implied_fov = 2.0 * np.rad2deg(np.arctan((eye_size[0] / 2.0) / calibration.focal_px))
        print(
            f"calibrated at {args.calibrate_at:.2f} m: centre disparity {centre:.2f} px "
            f"-> focal {calibration.focal_px:.0f} px (implies {implied_fov:.0f}deg horizontal FOV)"
        )
        if args.save_calibration is not None:
            calibration.to_json(args.save_calibration)
            print(f"calibration written to {args.save_calibration}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.preview:
        print("q or Esc to quit")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                left, right = split_side_by_side(frame)
                lr, rr = calibration.rectify(left, right)
                disparity = disparity_map(lr, rr, params)
                depth = depth_from_disparity(disparity, calibration)
                view = np.hstack((lr, colourise_depth(depth, args.max_depth / 2)))
                cv2.imshow("left | depth (near bright)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return 0

    # Let exposure settle; the first frames off a USB camera are often dark.
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        print("camera read failed", file=sys.stderr)
        return 1

    left, right = split_side_by_side(frame)
    lr, rr = calibration.rectify(left, right)

    started = time.monotonic()
    disparity = disparity_map(lr, rr, params)
    depth = depth_from_disparity(disparity, calibration)
    elapsed = time.monotonic() - started

    points, colours = point_cloud(disparity, lr, calibration, args.max_depth)

    cv2.imwrite(str(args.out / "left.png"), lr)
    cv2.imwrite(str(args.out / "right.png"), rr)
    cv2.imwrite(str(args.out / "depth.png"), colourise_depth(depth, args.max_depth / 2))
    colour, depth_mm = rgbd(lr, depth, args.max_depth)
    cv2.imwrite(str(args.out / "rgbd_colour.png"), colour)
    # 16-bit millimetres, the RealSense/Kinect/ROS 16UC1 convention. 0 = no reading.
    cv2.imwrite(str(args.out / "rgbd_depth_mm.png"), depth_mm)
    write_ply(args.out / "cloud.ply", points, colours)

    matched = int(np.isfinite(disparity).sum())
    total = disparity.size
    print(
        f"disparity in {elapsed * 1000:.0f} ms, {matched}/{total} pixels matched "
        f"({100.0 * matched / total:.0f}%)"
    )
    if matched:
        finite = depth[np.isfinite(depth)]
        print(
            f"depth: min {finite.min():.2f} m, median {np.median(finite):.2f} m, "
            f"max {finite.max():.2f} m"
        )
    print(f"{points.shape[0]} points -> {args.out / 'cloud.ply'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
