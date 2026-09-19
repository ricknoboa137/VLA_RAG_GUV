"""Depth and point clouds from a side-by-side stereo frame.

A "3D USB" camera delivers both eyes in one synchronised frame, so the pair is
split here rather than captured twice. Everything in this module is a pure
function of its inputs: no capture, no network, no model.

Depth is only metric once the camera is calibrated. :class:`StereoCalibration`
carries the rectification maps and the reprojection matrix ``Q`` produced by
``scripts/calibrate_stereo.py``; :func:`approximate_calibration` stands in
before that, from the baseline and field of view, and is honest about being an
estimate — it assumes the sensors are already rectified, which no real camera
is exactly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from carma.types.core import BgrImage, DepthImage

# Values OpenCV writes into a disparity map where it found no match. Disparity
# comes back fixed-point with 4 fractional bits, hence the division by 16.
_DISPARITY_SCALE = 16.0
_MIN_VALID_DISPARITY = 1e-3


def split_side_by_side(frame: BgrImage) -> tuple[BgrImage, BgrImage]:
    """Split one side-by-side frame into ``(left, right)``.

    Raises if the width is odd, which means the frame is not a pair.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        msg = f"frame must be HxWx3, got {frame.shape}"
        raise ValueError(msg)
    width = frame.shape[1]
    if width % 2 != 0:
        msg = f"side-by-side frame needs an even width, got {width}"
        raise ValueError(msg)
    half = width // 2
    return frame[:, :half].copy(), frame[:, half:].copy()


def looks_side_by_side(frame: BgrImage, threshold: float = 2.2) -> bool:
    """Whether a frame is wide enough to be a stereo pair.

    Two 4:3 views give 8:3, so the default threshold sits well above any
    single-sensor aspect ratio short of a specialist panoramic camera.
    """
    if frame.ndim < 2 or frame.shape[0] == 0:
        return False
    return frame.shape[1] / frame.shape[0] >= threshold


@dataclass(frozen=True)
class StereoCalibration:
    """Rectification maps and the reprojection matrix for one camera.

    ``Q`` maps ``(x, y, disparity)`` to metric ``(X, Y, Z)``. Its units follow
    the calibration target: a chessboard measured in metres yields metres.
    """

    image_size: tuple[int, int]
    left_map_x: npt.NDArray[np.float32]
    left_map_y: npt.NDArray[np.float32]
    right_map_x: npt.NDArray[np.float32]
    right_map_y: npt.NDArray[np.float32]
    q: npt.NDArray[np.float64]
    baseline_m: float
    focal_px: float
    rectified: bool = True
    # Disparity where an infinitely distant point lands. Zero for parallel
    # sensors; negative for a toed-in pair, which converges at a finite
    # distance and reads negative disparity beyond it.
    disparity_offset_px: float = 0.0

    def rectify(self, left: BgrImage, right: BgrImage) -> tuple[BgrImage, BgrImage]:
        """Apply the rectification maps so matching rows correspond."""
        lr = cv2.remap(left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        rr = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        return cast("BgrImage", lr), cast("BgrImage", rr)

    def to_json(self, path: Path) -> None:
        """Write the calibration.

        Identity rectification maps are not written: they are two float arrays
        the size of the image, which turn a small file into megabytes of JSON
        for no information at all. :meth:`from_json` rebuilds them.
        """
        payload: dict[str, object] = {
            "image_size": list(self.image_size),
            "q": self.q.tolist(),
            "baseline_m": self.baseline_m,
            "focal_px": self.focal_px,
            "disparity_offset_px": self.disparity_offset_px,
            "rectified": self.rectified,
        }
        if not self._maps_are_identity():
            payload["left_map_x"] = self.left_map_x.tolist()
            payload["left_map_y"] = self.left_map_y.tolist()
            payload["right_map_x"] = self.right_map_x.tolist()
            payload["right_map_y"] = self.right_map_y.tolist()
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _maps_are_identity(self) -> bool:
        """Whether rectification is a no-op, so the maps need not be stored."""
        width, height = self.image_size
        if self.left_map_x.shape != (height, width):
            return False
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        return all(
            np.array_equal(a, b)
            for a, b in (
                (self.left_map_x, xs),
                (self.left_map_y, ys),
                (self.right_map_x, xs),
                (self.right_map_y, ys),
            )
        )

    @staticmethod
    def from_json(path: Path) -> StereoCalibration:
        """Read a calibration written by :meth:`to_json`."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        width, height = (int(v) for v in raw["image_size"])
        identity_x, identity_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        def _map(key: str, fallback: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            if key not in raw:
                return fallback
            return np.asarray(raw[key], dtype=np.float32)

        return StereoCalibration(
            image_size=(width, height),
            left_map_x=_map("left_map_x", identity_x),
            left_map_y=_map("left_map_y", identity_y),
            right_map_x=_map("right_map_x", identity_x.copy()),
            right_map_y=_map("right_map_y", identity_y.copy()),
            q=np.asarray(raw["q"], dtype=np.float64),
            baseline_m=float(raw["baseline_m"]),
            focal_px=float(raw["focal_px"]),
            rectified=bool(raw.get("rectified", True)),
            disparity_offset_px=float(raw.get("disparity_offset_px", 0.0)),
        )


def approximate_calibration(
    image_size: tuple[int, int],
    baseline_m: float,
    horizontal_fov_deg: float = 60.0,
) -> StereoCalibration:
    """A calibration guessed from the baseline and field of view.

    Useful to see the shape of a scene before a chessboard calibration exists,
    and wrong in two ways that matter: it assumes the two sensors are perfectly
    rectified — parallel, coplanar and aligned to the pixel row — and it takes
    the field of view from the data sheet rather than from measurement. Treat
    the distances as indicative, not as measurements. ``rectified`` is False so
    callers can tell the two apart.
    """
    if baseline_m <= 0:
        msg = f"baseline must be positive, got {baseline_m}"
        raise ValueError(msg)
    width = image_size[0]
    focal_px = (width / 2.0) / float(np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    return calibration_with_focal(image_size, baseline_m, focal_px, horizontal_fov_deg)


def calibration_from_known_distance(
    image_size: tuple[int, int],
    baseline_m: float,
    known_distance_m: float,
    measured_disparity_px: float,
) -> StereoCalibration:
    """Solve the focal length from one object at a measured distance.

    ``depth = focal * baseline / disparity`` has a single unknown once the
    baseline is known and one true distance has been measured, so pointing the
    camera at something a tape measure away fixes the scale. This is much
    better than trusting a quoted field of view, which is often the diagonal
    or simply optimistic, and it needs no chessboard.

    It corrects scale only. Lens distortion and any misalignment between the
    two sensors remain, so accuracy still degrades towards the frame edges;
    ``rectified`` stays False to say so. A chessboard calibration fixes those.
    """
    if known_distance_m <= 0:
        msg = f"known distance must be positive, got {known_distance_m}"
        raise ValueError(msg)
    if measured_disparity_px <= 0:
        msg = f"measured disparity must be positive, got {measured_disparity_px}"
        raise ValueError(msg)
    focal_px = known_distance_m * measured_disparity_px / baseline_m
    implied_fov = 2.0 * np.rad2deg(np.arctan((image_size[0] / 2.0) / focal_px))
    return calibration_with_focal(image_size, baseline_m, focal_px, float(implied_fov))


def calibration_with_focal(
    image_size: tuple[int, int],
    baseline_m: float,
    focal_px: float,
    _implied_fov_deg: float,
    disparity_offset_px: float = 0.0,
) -> StereoCalibration:
    """Build an identity-rectified calibration around a known focal length.

    Rectification is the identity: this fixes the depth scale only, and leaves
    lens distortion and sensor misalignment in place.
    """
    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    # The last row's constant is where OpenCV puts a difference in principal
    # points, which is exactly how a disparity offset enters the reprojection:
    # Z = focal * baseline / (disparity - offset).
    q = np.array(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, focal_px],
            [0.0, 0.0, 1.0 / baseline_m, -disparity_offset_px / baseline_m],
        ],
        dtype=np.float64,
    )
    identity_x, identity_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return StereoCalibration(
        image_size=image_size,
        left_map_x=identity_x,
        left_map_y=identity_y,
        right_map_x=identity_x.copy(),
        right_map_y=identity_y.copy(),
        q=q,
        baseline_m=baseline_m,
        focal_px=focal_px,
        rectified=False,
        disparity_offset_px=disparity_offset_px,
    )


@dataclass(frozen=True)
class DistanceFit:
    """Focal length and disparity offset fitted from measured distances."""

    focal_px: float
    offset_px: float
    residuals_m: tuple[float, ...]
    implied_fov_deg: float

    @property
    def worst_error_m(self) -> float:
        """Largest absolute distance error over the samples used to fit."""
        return max((abs(r) for r in self.residuals_m), default=0.0)


def fit_focal_and_offset(
    samples: Sequence[tuple[float, float]],
    baseline_m: float,
    image_width: int,
) -> DistanceFit:
    """Fit ``disparity = focal * baseline / distance + offset`` by least squares.

    Two unknowns, so at least two distances are needed and three or more make
    the fit worth believing. The offset matters: a pair of sensors that is
    very slightly toed in, or whose principal points differ, adds a constant
    to every disparity, and forcing it to zero bends the focal length to
    compensate — which is how a single-distance calibration can agree with
    itself and still be wrong everywhere else.

    ``samples`` are ``(distance_m, disparity_px)`` pairs.
    """
    if len(samples) < 2:
        msg = f"need at least two distances to fit, got {len(samples)}"
        raise ValueError(msg)
    for distance, _ in samples:
        if distance <= 0:
            msg = f"distances must be positive, got {distance}"
            raise ValueError(msg)
    # Disparity itself may be negative: a toed-in pair reads below zero beyond
    # its convergence distance, and those samples are the most informative
    # ones for separating the focal length from the offset.

    inverse = np.array([1.0 / d for d, _ in samples], dtype=np.float64)
    measured = np.array([p for _, p in samples], dtype=np.float64)
    design = np.column_stack((inverse * baseline_m, np.ones_like(inverse)))
    (focal_px, offset_px), *_ = np.linalg.lstsq(design, measured, rcond=None)

    residuals: list[float] = []
    for distance, disparity in samples:
        corrected = disparity - float(offset_px)
        predicted_m = float(focal_px) * baseline_m / corrected if corrected > 0 else float("inf")
        residuals.append(predicted_m - distance)

    fov = 2.0 * np.rad2deg(np.arctan((image_width / 2.0) / float(focal_px)))
    return DistanceFit(
        focal_px=float(focal_px),
        offset_px=float(offset_px),
        residuals_m=tuple(residuals),
        implied_fov_deg=float(fov),
    )


def median_disparity(disparity: DepthImage, patch: int = 120) -> float:
    """Median disparity over a central patch, or NaN when nothing matched.

    Used to read off the disparity of an object placed at a measured distance
    in front of the camera, for :func:`calibration_from_known_distance`.
    """
    height, width = disparity.shape[:2]
    half = max(1, patch // 2)
    cy, cx = height // 2, width // 2
    window = disparity[
        max(0, cy - half) : cy + half,
        max(0, cx - half) : cx + half,
    ]
    values = window[np.isfinite(window)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def rgbd(
    colour_image: BgrImage,
    depth_m: DepthImage,
    max_depth_m: float = 20.0,
) -> tuple[BgrImage, npt.NDArray[np.uint16]]:
    """An aligned colour image and a 16-bit depth image in millimetres.

    Millimetres in uint16 is the convention RealSense, Kinect and the ROS
    ``16UC1`` encoding all use, so the result drops into existing RGB-D tools.
    Zero means "no reading", which those tools already treat as invalid — the
    NaN used internally has no representation in an integer image.

    The pair is aligned by construction: depth is computed in the rectified
    left camera's frame, so pixel (x, y) is the same ray in both.
    """
    if colour_image.shape[:2] != depth_m.shape[:2]:
        msg = f"colour {colour_image.shape[:2]} and depth {depth_m.shape[:2]} must match"
        raise ValueError(msg)
    millimetres = depth_m * 1000.0
    valid = np.isfinite(millimetres) & (millimetres > 0) & (millimetres <= max_depth_m * 1000.0)
    out = np.zeros(depth_m.shape[:2], dtype=np.uint16)
    out[valid] = millimetres[valid].astype(np.uint16)
    return colour_image, out


@dataclass(frozen=True)
class DisparityParams:
    """Semi-global block matching settings.

    ``num_disparities`` must be a multiple of 16 and sets the closest distance
    the camera can resolve: larger means nearer objects are matched, at the
    cost of a wider blind band down the left edge and more time per frame.
    """

    num_disparities: int = 96
    block_size: int = 7
    # Where the search starts. Zero assumes parallel sensors; a toed-in pair
    # reads negative disparity beyond its convergence distance, and with the
    # default of zero everything past that point simply fails to match.
    min_disparity: int = 0
    # Local contrast equalisation before matching. Block matching keys on
    # local contrast, so evening it out puts depth edges on image edges
    # instead of smearing them; measured here to roughly double the agreement
    # between the two, at no cost in time. It slightly lowers raw coverage,
    # trading a few ambiguous pixels for sharper shapes, which is the right
    # trade for mapping.
    equalise_contrast: bool = True
    clahe_clip_limit: float = 2.0
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1

    def __post_init__(self) -> None:
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            msg = f"num_disparities must be a positive multiple of 16, got {self.num_disparities}"
            raise ValueError(msg)
        if self.block_size < 1 or self.block_size % 2 == 0:
            msg = f"block_size must be a positive odd number, got {self.block_size}"
            raise ValueError(msg)


def disparity_map(
    left: BgrImage,
    right: BgrImage,
    params: DisparityParams | None = None,
) -> DepthImage:
    """Disparity in pixels as float32, with unmatched pixels set to NaN.

    NaN rather than a sentinel so that arithmetic downstream cannot silently
    treat "no match" as "zero disparity", which reads as infinitely far away.
    """
    if left.shape != right.shape:
        msg = f"left and right must match, got {left.shape} and {right.shape}"
        raise ValueError(msg)
    settings = params or DisparityParams()

    grey_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grey_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if settings.equalise_contrast:
        # The same operator on both eyes, so a matched pair stays matched.
        clahe = cv2.createCLAHE(clipLimit=settings.clahe_clip_limit, tileGridSize=(8, 8))
        grey_l = clahe.apply(grey_l)
        grey_r = clahe.apply(grey_r)
    channels = 1
    matcher = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
        minDisparity=settings.min_disparity,
        numDisparities=settings.num_disparities,
        blockSize=settings.block_size,
        # Smoothness penalties, from the OpenCV documentation's suggested form.
        P1=8 * channels * settings.block_size**2,
        P2=32 * channels * settings.block_size**2,
        disp12MaxDiff=settings.disp12_max_diff,
        uniquenessRatio=settings.uniqueness_ratio,
        speckleWindowSize=settings.speckle_window_size,
        speckleRange=settings.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(grey_l, grey_r).astype(np.float32) / _DISPARITY_SCALE
    # OpenCV marks "no match" as one step below the search range, so the test
    # has to follow min_disparity rather than assume the range starts at zero.
    floor = float(settings.min_disparity) + _MIN_VALID_DISPARITY
    disparity = np.where(raw > floor, raw, np.nan)
    return disparity.astype(np.float32)


def depth_from_disparity(disparity: DepthImage, calibration: StereoCalibration) -> DepthImage:
    """Metres per pixel, NaN where disparity is unknown.

    ``depth = focal * baseline / (disparity - offset)``: precision falls off
    with the square of distance, so far readings from a short baseline are
    coarse. The offset is what a toed-in pair reads at infinity, and ignoring
    it makes every distance wrong in a way that still looks self-consistent.
    """
    corrected = disparity - calibration.disparity_offset_px
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = calibration.focal_px * calibration.baseline_m / corrected
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)


def point_cloud(
    disparity: DepthImage,
    colour_image: BgrImage,
    calibration: StereoCalibration,
    max_depth_m: float = 20.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """``(points_xyz, colours_rgb)`` for pixels with a usable match.

    Points beyond ``max_depth_m`` are dropped: at long range the disparity is
    a pixel or two and the position is mostly quantisation noise.
    """
    filled = np.nan_to_num(disparity, nan=0.0)
    xyz = cv2.reprojectImageTo3D(filled, calibration.q)
    valid = np.isfinite(disparity) & np.isfinite(xyz).all(axis=2)
    valid &= (xyz[:, :, 2] > 0) & (xyz[:, :, 2] <= max_depth_m)

    points = xyz[valid].astype(np.float32)
    colours = cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)[valid].astype(np.uint8)
    return points, colours


def write_ply(path: Path, points: npt.NDArray[np.float32], colours: npt.NDArray[np.uint8]) -> None:
    """Write a coloured point cloud as binary PLY, which MeshLab and CloudCompare read."""
    if points.shape[0] != colours.shape[0]:
        msg = f"points and colours must match, got {points.shape[0]} and {colours.shape[0]}"
        raise ValueError(msg)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    rows = np.empty(points.shape[0], dtype=dtype)
    rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
    rows["red"], rows["green"], rows["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(rows.tobytes())


def colourise_depth(depth_m: DepthImage, max_depth_m: float = 10.0) -> BgrImage:
    """A viewable BGR image of a depth map; unknown pixels are black."""
    known = np.isfinite(depth_m)
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    if known.any():
        clipped = np.clip(depth_m[known], 0.0, max_depth_m) / max_depth_m
        # Near is bright, far is dark, which reads more naturally than the reverse.
        scaled[known] = ((1.0 - clipped) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    coloured[~known] = 0
    return cast("BgrImage", coloured)
