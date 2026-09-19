"""Stereo depth: splitting, disparity, metric depth and the point cloud."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from carma.perception.stereo_depth import (
    DisparityParams,
    approximate_calibration,
    calibration_with_focal,
    colourise_depth,
    depth_from_disparity,
    disparity_map,
    fit_focal_and_offset,
    looks_side_by_side,
    point_cloud,
    rgbd,
    split_side_by_side,
    write_ply,
)

_RNG = np.random.default_rng(0)


def _textured(height: int, width: int) -> np.ndarray:
    """Blob texture, the way a real scene looks to a block matcher.

    Per-pixel noise is not a substitute: neighbouring pixels are independent,
    so every block is ambiguous and the uniqueness check rejects nearly
    everything. Upscaling coarse noise gives detail that is locally distinct
    and spatially smooth, which is what block matching relies on.
    """
    coarse = _RNG.integers(
        0, 255, size=(max(2, height // 8), max(2, width // 8), 3), dtype=np.uint8
    )
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def _shifted_pair(shift: int, height: int = 240, width: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """A left/right pair where the right image is the left shifted by ``shift`` pixels.

    A constant shift is the same as a plane at one constant depth, which makes
    the expected disparity exactly ``shift``.

    The direction matters: disparity is how far left a feature moves going
    from the left image to the right one, so the right image samples the
    scene ``shift`` pixels further along. Building it the other way round
    gives a negative disparity, which the matcher cannot represent and which
    shows up as plausible-looking nonsense.
    """
    base = _textured(height, width + shift)
    left = base[:, :width]
    right = base[:, shift : shift + width]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def test_split_side_by_side_halves_the_frame() -> None:
    frame = _textured(48, 128)
    left, right = split_side_by_side(frame)
    assert left.shape == (48, 64, 3)
    assert right.shape == (48, 64, 3)
    assert np.array_equal(left, frame[:, :64])
    assert np.array_equal(right, frame[:, 64:])


def test_split_rejects_an_odd_width() -> None:
    with pytest.raises(ValueError, match="even width"):
        split_side_by_side(_textured(10, 11))


def test_looks_side_by_side_separates_the_layouts() -> None:
    assert looks_side_by_side(_textured(480, 1280))  # two 4:3 eyes
    assert not looks_side_by_side(_textured(480, 640))  # one 4:3 camera
    assert not looks_side_by_side(_textured(1080, 1920))  # one 16:9 camera


def test_disparity_recovers_a_known_shift() -> None:
    shift = 12
    left, right = _shifted_pair(shift)
    disparity = disparity_map(left, right, DisparityParams(num_disparities=32, block_size=7))

    matched = disparity[np.isfinite(disparity)]
    assert matched.size > 0.5 * disparity.size, "most of a textured pair should match"
    # Sub-pixel interpolation means the result is near, not exactly, the shift.
    assert abs(float(np.median(matched)) - shift) < 1.0


def test_unmatched_pixels_are_nan_not_zero() -> None:
    # Two unrelated images have no correspondence to find.
    left, right = _textured(240, 320), _textured(240, 320)
    disparity = disparity_map(left, right, DisparityParams(num_disparities=32))
    assert np.isnan(disparity).any(), "no-match must be NaN so it cannot read as far away"


def test_depth_follows_focal_times_baseline_over_disparity() -> None:
    calibration = approximate_calibration((640, 480), baseline_m=0.06, horizontal_fov_deg=100.0)
    disparity = np.array([[calibration.focal_px * calibration.baseline_m]], dtype=np.float32)
    depth = depth_from_disparity(disparity, calibration)
    assert depth[0, 0] == pytest.approx(1.0, rel=1e-5)


def test_depth_is_nan_where_disparity_is_unknown() -> None:
    calibration = approximate_calibration((64, 48), baseline_m=0.06)
    disparity = np.array([[np.nan, 10.0]], dtype=np.float32)
    depth = depth_from_disparity(disparity, calibration)
    assert np.isnan(depth[0, 0])
    assert np.isfinite(depth[0, 1])


def test_approximate_calibration_rejects_a_zero_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        approximate_calibration((640, 480), baseline_m=0.0)


def test_disparity_params_reject_invalid_settings() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        DisparityParams(num_disparities=100)
    with pytest.raises(ValueError, match="odd"):
        DisparityParams(block_size=8)


def test_point_cloud_drops_far_and_unmatched_points() -> None:
    calibration = approximate_calibration((320, 240), baseline_m=0.06, horizontal_fov_deg=100.0)
    left, right = _shifted_pair(10, height=240, width=320)
    disparity = disparity_map(left, right, DisparityParams(num_disparities=32))

    near, colours = point_cloud(disparity, left, calibration, max_depth_m=100.0)
    far, _ = point_cloud(disparity, left, calibration, max_depth_m=0.01)
    assert near.shape[0] > 0
    assert near.shape[1] == 3
    assert colours.shape[0] == near.shape[0]
    assert far.shape[0] < near.shape[0], "a tight depth limit must remove points"
    assert np.isfinite(near).all()


def test_write_ply_round_trips_the_vertex_count(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    colours = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    write_ply(path, points, colours)

    head = path.read_bytes()[:200].decode("ascii", errors="replace")
    assert head.startswith("ply")
    assert "element vertex 2" in head
    # 2 vertices x (3 floats + 3 bytes) after the header.
    assert path.stat().st_size == len(head.split("end_header\n")[0]) + len("end_header\n") + 2 * 15


def test_write_ply_rejects_mismatched_colours(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must match"):
        write_ply(
            tmp_path / "bad.ply",
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.uint8),
        )


def test_colourise_depth_blacks_out_unknown_pixels() -> None:
    depth = np.array([[np.nan, 1.0]], dtype=np.float32)
    image = colourise_depth(depth, max_depth_m=10.0)
    assert image.shape == (1, 2, 3)
    assert tuple(image[0, 0]) == (0, 0, 0)
    assert tuple(image[0, 1]) != (0, 0, 0)


def test_fit_recovers_focal_and_offset_from_a_toed_in_pair() -> None:
    # A camera whose sensors converge: disparity hits zero at a finite range.
    focal, baseline, offset = 567.5, 0.06, -36.5
    samples = [(d, focal * baseline / d + offset) for d in (0.31, 0.5, 0.72, 1.26)]

    fit = fit_focal_and_offset(samples, baseline, image_width=640)

    assert fit.focal_px == pytest.approx(focal, rel=1e-6)
    assert fit.offset_px == pytest.approx(offset, abs=1e-6)
    assert fit.worst_error_m < 1e-6


def test_fit_needs_two_distances() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_focal_and_offset([(0.5, 30.0)], 0.06, image_width=640)


def test_fit_accepts_negative_disparity() -> None:
    # Beyond the convergence distance disparity is negative, and those samples
    # are exactly the ones that pin down the offset.
    samples = [(0.31, 73.9), (1.26, -9.0)]
    fit = fit_focal_and_offset(samples, 0.06, image_width=640)
    assert fit.focal_px > 0
    assert fit.offset_px < 0


def test_depth_accounts_for_the_disparity_offset() -> None:
    plain = approximate_calibration((640, 480), baseline_m=0.06, horizontal_fov_deg=100.0)
    toed_in = calibration_with_focal((640, 480), 0.06, 567.5, 59.0, disparity_offset_px=-36.5)

    # At the convergence distance the toed-in pair reads zero disparity, so a
    # model that ignores the offset puts a 0.93 m object at infinity.
    disparity = np.array([[0.0]], dtype=np.float32)
    assert not np.isfinite(depth_from_disparity(disparity, plain)[0, 0])

    measured = np.array([[-9.0]], dtype=np.float32)
    depth = depth_from_disparity(measured, toed_in)[0, 0]
    assert depth == pytest.approx(1.24, abs=0.05)


def test_q_matrix_encodes_the_offset_consistently_with_depth() -> None:
    # reprojectImageTo3D must agree with depth_from_disparity, or the point
    # cloud and the depth map describe different scenes.
    calibration = calibration_with_focal((640, 480), 0.06, 567.5, 59.0, disparity_offset_px=-36.5)
    disparity = np.full((8, 8), -9.0, dtype=np.float32)
    expected = depth_from_disparity(disparity, calibration)[4, 4]

    xyz = cv2.reprojectImageTo3D(disparity, calibration.q)
    assert float(xyz[4, 4, 2]) == pytest.approx(float(expected), rel=1e-4)


def test_negative_min_disparity_is_searched() -> None:
    # The matcher must be able to return disparities below zero, and the
    # no-match sentinel must follow the start of the search rather than zero.
    shift = 10
    left, right = _shifted_pair(shift)
    params = DisparityParams(num_disparities=64, block_size=7, min_disparity=-32)
    disparity = disparity_map(left, right, params)

    matched = disparity[np.isfinite(disparity)]
    assert matched.size > 0
    assert abs(float(np.median(matched)) - shift) < 1.5


def test_rgbd_returns_aligned_millimetres() -> None:
    colour = _textured(16, 16)
    depth = np.full((16, 16), 1.5, dtype=np.float32)
    depth[0, 0] = np.nan

    aligned, millimetres = rgbd(colour, depth, max_depth_m=5.0)

    assert aligned.shape[:2] == millimetres.shape
    assert millimetres.dtype == np.uint16
    assert millimetres[0, 0] == 0, "unknown depth must be 0, the RGB-D convention"
    assert millimetres[8, 8] == 1500


def test_rgbd_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must match"):
        rgbd(_textured(8, 8), np.zeros((4, 4), dtype=np.float32))
