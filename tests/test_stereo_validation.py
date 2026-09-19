"""The published stereo-depth figures must stay reproducible from the record.

These are not unit tests of the algorithms — ``test_stereo_depth.py`` covers
those. They pin the numbers that appear in ``docs/stereo_depth.md`` to the
measurements in ``experiments/stereo_depth_validation/measurements.json``, so
a change to the depth model that would invalidate a published claim fails here
rather than in review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carma.perception.stereo_depth import StereoCalibration, fit_focal_and_offset

_MEASUREMENTS = Path("experiments/stereo_depth_validation/measurements.json")
_CALIBRATION = Path("config/stereo_3dusb.json")
_EYE_WIDTH = 640


@pytest.fixture(scope="module")
def record() -> dict:  # type: ignore[type-arg]
    return json.loads(_MEASUREMENTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit(record: dict):  # type: ignore[type-arg, no-untyped-def]
    baseline = float(record["hardware"]["baseline_m"])
    samples = [
        (float(s["distance_m"]), float(s["disparity_px"])) for s in record["calibration_samples"]
    ]
    return fit_focal_and_offset(samples, baseline, _EYE_WIDTH)


def test_fit_reproduces_the_published_parameters(fit) -> None:  # type: ignore[no-untyped-def]
    assert fit.focal_px == pytest.approx(567.5, abs=0.5)
    assert fit.offset_px == pytest.approx(-36.5, abs=0.5)


def test_the_sensors_are_toed_in(fit, record: dict) -> None:  # type: ignore[type-arg, no-untyped-def]
    # The published claim: disparity reaches zero at a finite distance, which
    # is why the matcher has to search below zero.
    baseline = float(record["hardware"]["baseline_m"])
    assert fit.offset_px < 0
    convergence_m = fit.focal_px * baseline / -fit.offset_px
    assert convergence_m == pytest.approx(0.93, abs=0.02)


def test_worst_fit_residual_is_within_the_published_bound(fit) -> None:  # type: ignore[no-untyped-def]
    assert fit.worst_error_m * 1000 <= 25.0


def test_held_out_distances_are_within_two_percent(fit, record: dict) -> None:  # type: ignore[type-arg, no-untyped-def]
    """The strongest claim: distances never used in the fit still come out right."""
    baseline = float(record["hardware"]["baseline_m"])
    focal_baseline = fit.focal_px * baseline

    for sample in record["held_out_samples"]:
        distance = float(sample["distance_m"])
        depth = focal_baseline / (float(sample["disparity_px"]) - fit.offset_px)
        relative = abs(depth - distance) / distance
        assert relative < 0.02, f"{distance} m predicted as {depth:.3f} m"


def test_shipped_calibration_matches_the_recorded_fit(fit) -> None:  # type: ignore[no-untyped-def]
    """config/ must hold the calibration these measurements produce, not a stale one."""
    calibration = StereoCalibration.from_json(_CALIBRATION)
    assert calibration.focal_px == pytest.approx(fit.focal_px, rel=1e-3)
    assert calibration.disparity_offset_px == pytest.approx(fit.offset_px, rel=1e-3)


def test_a_zero_search_floor_would_have_been_badly_wrong(record: dict) -> None:  # type: ignore[type-arg]
    """The published before/after: searching only positive disparity fails past convergence."""
    comparison = record["search_floor_comparison"]
    true_m = float(comparison["true_distance_m"])
    by_floor = {int(c["min_disparity"]): c for c in comparison["conditions"]}

    naive_error = abs(float(by_floor[0]["reported_depth_m"]) - true_m)
    fixed_error = abs(float(by_floor[-48]["reported_depth_m"]) - true_m)
    assert naive_error > 0.3, "the failure it documents must be a large one"
    assert fixed_error < 0.06
    assert by_floor[-48]["coverage_percent"] > by_floor[0]["coverage_percent"]


def test_contrast_equalisation_does_not_shift_distance(record: dict) -> None:  # type: ignore[type-arg]
    """Sharper edges must not be bought with a biased distance."""
    conditions = record["contrast_equalisation_bias_check"]["conditions"]
    disparities = {bool(c["equalise_contrast"]): float(c["disparity_px"]) for c in conditions}
    assert disparities[True] == pytest.approx(disparities[False], abs=0.01)


def test_contrast_equalisation_is_the_best_edge_result(record: dict) -> None:  # type: ignore[type-arg]
    """The reason it is the default, and that the expensive alternative is not."""
    conditions = record["matcher_comparison"]["conditions"]
    by_name = {c["name"]: c for c in conditions}
    clahe = by_name["CLAHE contrast equalisation"]
    baseline = by_name["baseline SGBM_3WAY, block 9"]
    eight_way = by_name["MODE_HH, 8 directions"]

    assert clahe["edge_agreement"] > 1.5 * baseline["edge_agreement"]
    assert clahe["milliseconds"] <= baseline["milliseconds"]
    # The expensive mode buys nothing, which is why it is not the default.
    assert eight_way["edge_agreement"] <= baseline["edge_agreement"]
    assert eight_way["milliseconds"] > 4 * baseline["milliseconds"]


def test_the_record_keeps_its_provenance_and_limitations(record: dict) -> None:  # type: ignore[type-arg]
    """A measurement without its method and its caveats is not evidence."""
    for key in ("title", "date", "hardware", "method", "limitations"):
        assert key in record, f"missing {key}"
    assert record["method"]["distance_measurement"]
    assert record["hardware"]["baseline_source"]
    assert len(record["limitations"]) >= 3
