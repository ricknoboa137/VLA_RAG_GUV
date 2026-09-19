"""Recompute every published stereo-depth number from the recorded measurements.

    python scripts/report_stereo_validation.py

Reads ``experiments/stereo_depth_validation/measurements.json`` and prints the
fit, the held-out errors and the range table. It touches no camera, so anyone
can reproduce the published figures from the repository alone; running it is
the check that the paper and the data still agree.

``--markdown`` emits the tables ready to paste into a paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carma.perception.stereo_depth import fit_focal_and_offset

_DEFAULT = Path("experiments/stereo_depth_validation/measurements.json")
# One pixel of disparity error costs Z^2 / (focal * baseline) metres; half a
# pixel is what sub-pixel interpolation realistically achieves.
_SUBPIXEL = 0.5
_RANGE_M = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0)


def main() -> int:
    """Print the validation report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measurements", type=Path, default=_DEFAULT)
    parser.add_argument("--markdown", action="store_true", help="emit paper-ready tables")
    args = parser.parse_args()

    data = json.loads(args.measurements.read_text(encoding="utf-8"))
    baseline = float(data["hardware"]["baseline_m"])
    eye_width = 640

    samples = [
        (float(s["distance_m"]), float(s["disparity_px"])) for s in data["calibration_samples"]
    ]
    fit = fit_focal_and_offset(samples, baseline, eye_width)
    convergence = fit.focal_px * baseline / -fit.offset_px if fit.offset_px < 0 else float("inf")
    focal_baseline = fit.focal_px * baseline

    bullet = "| " if args.markdown else "  "
    print(f"# {data['title']}  ({data['date']})\n")
    print(
        f"{data['hardware']['camera']}, baseline {baseline * 1000:.0f} mm, "
        f"{data['hardware']['capture_mode']}\n"
    )

    print("## Calibration")
    if args.markdown:
        print("\n| Distance (m) | Disparity (px) | Frame spread (px) | Residual (mm) |")
        print("|---|---|---|---|")
    for (distance, disparity), residual in zip(samples, fit.residuals_m, strict=True):
        spread = next(
            s["frame_spread_px"]
            for s in data["calibration_samples"]
            if abs(float(s["distance_m"]) - distance) < 1e-9
        )
        if args.markdown:
            print(f"| {distance:.3f} | {disparity:+.2f} | {spread:.2f} | {residual * 1000:+.0f} |")
        else:
            print(
                f"{bullet}{distance:.3f} m  {disparity:+7.2f} px  "
                f"spread {spread:.2f}  residual {residual * 1000:+.0f} mm"
            )
    print(
        f"\nfocal {fit.focal_px:.1f} px, offset {fit.offset_px:+.2f} px, "
        f"implied horizontal FOV {fit.implied_fov_deg:.0f} deg "
        f"(quoted {data['hardware']['quoted_horizontal_fov_deg']} deg)"
    )
    print(
        f"convergence distance {convergence:.2f} m; worst fit residual "
        f"{fit.worst_error_m * 1000:.0f} mm\n"
    )

    print("## Held-out distances (not used in the fit)")
    if args.markdown:
        print("\n| Tape (m) | Disparity (px) | Depth (m) | Error (mm) | Error (%) |")
        print("|---|---|---|---|---|")
    worst = 0.0
    for sample in data["held_out_samples"]:
        distance = float(sample["distance_m"])
        disparity = float(sample["disparity_px"])
        depth = focal_baseline / (disparity - fit.offset_px)
        error = depth - distance
        worst = max(worst, abs(error))
        if args.markdown:
            print(
                f"| {distance:.3f} | {disparity:+.2f} | {depth:.3f} | "
                f"{error * 1000:+.0f} | {100 * error / distance:+.1f} |"
            )
        else:
            print(
                f"{bullet}{distance:.3f} m  {disparity:+7.2f} px -> {depth:.3f} m  "
                f"({error * 1000:+.0f} mm, {100 * error / distance:+.1f}%)"
            )
    print(f"\nworst held-out error {worst * 1000:.0f} mm\n")

    print("## Why the disparity search must go negative")
    comparison = data["search_floor_comparison"]
    true_m = float(comparison["true_distance_m"])
    for condition in comparison["conditions"]:
        error = float(condition["reported_depth_m"]) - true_m
        print(
            f"{bullet}min_disparity {condition['min_disparity']:+4d}: "
            f"{condition['disparity_px']:+6.2f} px -> {condition['reported_depth_m']:.3f} m "
            f"({error * 1000:+.0f} mm), coverage {condition['coverage_percent']}%"
        )
    print(f"\ntrue distance {true_m:.2f} m\n")

    print("## Depth resolution")
    if args.markdown:
        print("\n| Distance (m) | Disparity (px) | Error at 0.5 px (mm) | Error (%) |")
        print("|---|---|---|---|")
    for distance in _RANGE_M:
        disparity = focal_baseline / distance + fit.offset_px
        error = _SUBPIXEL * distance * distance / focal_baseline
        if args.markdown:
            print(
                f"| {distance:.2f} | {disparity:+.2f} | {error * 1000:.0f} | "
                f"{100 * error / distance:.1f} |"
            )
        else:
            print(
                f"{bullet}{distance:.2f} m  {disparity:+7.2f} px  "
                f"{error * 1000:6.0f} mm  {100 * error / distance:4.1f}%"
            )

    print("\n## Limitations")
    for limitation in data["limitations"]:
        print(f"- {limitation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
