"""Calibrate stereo depth scale from a box at tape-measured distances.

No chessboard needed. Place a textured box squarely in front of the camera so
it fills the middle of the view, measure the distance from the front of the
camera to the box with a tape, and take a sample:

    python scripts/calibrate_stereo.py sample --distance 0.60
    python scripts/calibrate_stereo.py sample --distance 1.00
    python scripts/calibrate_stereo.py sample --distance 1.50
    python scripts/calibrate_stereo.py fit --out config/stereo_3dusb.json

Samples accumulate in a JSON file between runs, so each distance is its own
command and nothing has to be held still while a prompt waits.

Three or more distances, spread as widely as the camera's useful range allows,
give a fit worth trusting. Two is the minimum and leaves nothing to check the
result against.

This recovers the depth scale — the focal length and a constant disparity
offset. It does not correct lens distortion or sensor misalignment, which need
a chessboard; expect accuracy to fall off towards the edges of the frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from carma.perception.stereo_depth import (
    DisparityParams,
    calibration_with_focal,
    disparity_map,
    fit_focal_and_offset,
    looks_side_by_side,
    median_disparity,
    split_side_by_side,
)


def _capture_disparity(
    camera: int,
    width: int,
    height: int,
    params: DisparityParams,
    frames: int,
    patch: int,
) -> tuple[float, float, tuple[int, int]]:
    """Median centre disparity over several frames, with its spread."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        msg = f"camera {camera} could not be opened"
        raise SystemExit(msg)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Discard the first frames: exposure and gain are still settling.
    for _ in range(15):
        cap.read()

    values: list[float] = []
    eye_size = (0, 0)
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if not looks_side_by_side(frame):
            cap.release()
            h, w = frame.shape[:2]
            msg = f"{w}x{h} is not a side-by-side pair"
            raise SystemExit(msg)
        left, right = split_side_by_side(frame)
        eye_size = (left.shape[1], left.shape[0])
        value = median_disparity(disparity_map(left, right, params), patch=patch)
        if np.isfinite(value):
            values.append(value)
    cap.release()

    if not values:
        msg = "nothing matched in the centre; aim at a textured object filling the middle"
        raise SystemExit(msg)
    array = np.array(values, dtype=np.float64)
    return float(np.median(array)), float(array.max() - array.min()), eye_size


def main() -> int:
    """Collect a sample, or fit a calibration from the samples collected."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("sample", "fit", "list", "clear"))
    parser.add_argument("--distance", type=float, default=None, help="tape distance, metres")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--baseline", type=float, default=0.06, help="lens separation, metres")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--patch", type=int, default=120, help="central window, pixels")
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=9)
    parser.add_argument(
        "--min-disparity",
        type=int,
        default=0,
        help="start of the search; negative for toed-in sensors (see the fit output)",
    )
    parser.add_argument("--samples", type=Path, default=Path("stereo_samples.json"))
    parser.add_argument("--out", type=Path, default=None, help="calibration JSON to write")
    args = parser.parse_args()

    stored: list[dict[str, float]] = []
    if args.samples.exists():
        stored = json.loads(args.samples.read_text(encoding="utf-8"))

    if args.action == "clear":
        args.samples.unlink(missing_ok=True)
        print(f"cleared {args.samples}")
        return 0

    if args.action == "list":
        if not stored:
            print("no samples yet")
            return 0
        for entry in stored:
            print(f"  {entry['distance_m']:.3f} m -> {entry['disparity_px']:.2f} px")
        return 0

    if args.action == "sample":
        if args.distance is None:
            print("--distance is required when sampling", file=sys.stderr)
            return 1
        params = DisparityParams(
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            min_disparity=args.min_disparity,
        )
        disparity, spread, eye_size = _capture_disparity(
            args.camera, args.width, args.height, params, args.frames, args.patch
        )
        stored.append(
            {
                "distance_m": args.distance,
                "disparity_px": disparity,
                "spread_px": spread,
                "eye_width": eye_size[0],
                "eye_height": eye_size[1],
            }
        )
        args.samples.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        print(
            f"{args.distance:.3f} m -> {disparity:.2f} px (spread {spread:.2f} px). "
            f"{len(stored)} sample(s) in {args.samples}"
        )
        return 0

    # fit
    if len(stored) < 2:
        print(f"need at least two samples to fit, have {len(stored)}", file=sys.stderr)
        return 1
    pairs = [(float(e["distance_m"]), float(e["disparity_px"])) for e in stored]
    eye_width = int(stored[0]["eye_width"])
    eye_height = int(stored[0]["eye_height"])
    fit = fit_focal_and_offset(pairs, args.baseline, eye_width)

    print(f"fitted from {len(pairs)} distances:")
    print(f"  focal      {fit.focal_px:.1f} px")
    print(f"  offset     {fit.offset_px:+.2f} px")
    print(f"  implies    {fit.implied_fov_deg:.0f}deg horizontal field of view")
    print("  residuals:")
    for (distance, disparity), residual in zip(pairs, fit.residuals_m, strict=True):
        print(
            f"    {distance:.3f} m ({disparity:.2f} px): {residual * 1000:+.0f} mm "
            f"({100 * residual / distance:+.1f}%)"
        )
    print(f"  worst error {fit.worst_error_m * 1000:.0f} mm")

    if fit.offset_px < -1.0:
        convergence_m = fit.focal_px * args.baseline / -fit.offset_px
        suggested = -int(np.ceil(abs(fit.offset_px) / 16.0)) * 16
        print(
            f"\nThe sensors are toed in: disparity reaches zero at {convergence_m:.2f} m and "
            f"goes negative beyond it. To see past {convergence_m:.2f} m the matcher has to "
            f"search negative disparity, e.g. DisparityParams(min_disparity={suggested}).",
        )

    if args.out is not None:
        calibration = calibration_with_focal(
            (eye_width, eye_height),
            args.baseline,
            fit.focal_px,
            fit.implied_fov_deg,
            fit.offset_px,
        )
        calibration.to_json(args.out)
        print(f"\ncalibration written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
