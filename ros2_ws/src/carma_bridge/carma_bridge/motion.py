"""Motion evidence that does not trust the wheels, without ROS.

Two problems seen in simulation, both of which a real field robot has worse:

* Wheel odometry keeps counting while the robot is stuck against a crop row.
* Heading integrated from steering drifts badly: one alley episode ended with
  odometry at -0.02 rad and the true heading at -0.86 rad.

This module answers each with a sensor the robot will carry.

:class:`StallDetector` watches the camera. If the wheels are being driven
forward and the image has barely changed for ``stall_s`` seconds, the robot is
not moving, whatever the encoders say.

:class:`PlanarEstimator` dead-reckons the pose from the wheel speed (distance
only) and the gyro's yaw rate (heading). It removes the heading drift; it does
not fix distance while the wheels slip, which is why it is paired with the
stall detector, and why a field run adds GNSS. Its error against ground truth
is measured in simulation, not assumed.

Images are BGR ``uint8`` HWC, the project convention. No ROS imports; OpenCV
only, which the bridge image already carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

_THUMB_W = 64
_THUMB_H = 48


def _thumbnail(bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Grey, area-averaged to 64x48, so noise and small sway average out."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def image_change(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    """Mean absolute difference of two frames' thumbnails, in grey levels (0..255)."""
    return float(np.abs(_thumbnail(a) - _thumbnail(b)).mean())


@dataclass(frozen=True)
class StallConfig:
    """Thresholds for :class:`StallDetector`.

    Defaults are set from measurements in the alley world (2026-11-14): driving
    at 0.5 m/s, the thumbnail changed by at least 0.98 grey levels per second
    (median 1.2 to 2.7); parked, by exactly 0.0. A first threshold of 1.0
    sat on top of normal driving and latched the safety gate on a moving
    robot. The simulated camera has no noise, so on real images ``max_change``
    must be re-measured against a parked robot before it is trusted.

    Attributes:
        min_command_mps: Forward speed above which the robot should be moving.
        max_change: Image change (grey levels) below which the view is static.
        stall_s: How long both must hold before a stall is declared, seconds.
            At ``min_command_mps`` the robot covers 0.4 m in this time, enough
            to change the view well past ``max_change``.
    """

    min_command_mps: float = 0.2
    max_change: float = 0.3
    stall_s: float = 2.0


class StallDetector:
    """Declares a stall when the robot is driven but its camera view is static."""

    def __init__(self, config: StallConfig) -> None:
        self._config = config
        self._reference: npt.NDArray[np.uint8] | None = None
        self._since_s: float | None = None

    def reset(self) -> None:
        """Forget history, for a new episode."""
        self._reference = None
        self._since_s = None

    def update(self, t_s: float, commanded_mps: float, frame: npt.NDArray[np.uint8]) -> bool:
        """Feed one frame; True once the robot has been stalled for ``stall_s``.

        The reference frame is the one in view when the static stretch began,
        so a slow creep that changes the view a little each frame still counts
        as motion once it adds up.
        """
        driven = abs(commanded_mps) >= self._config.min_command_mps
        if not driven or self._reference is None:
            self._reference = frame
            self._since_s = t_s if driven else None
            return False
        if image_change(self._reference, frame) > self._config.max_change:
            self._reference = frame
            self._since_s = t_s
            return False
        if self._since_s is None:
            self._since_s = t_s
        return t_s - self._since_s >= self._config.stall_s


@dataclass(frozen=True)
class PlanarPose:
    """x, y in metres and yaw in radians, in the frame the estimate started in."""

    x: float
    y: float
    yaw: float


class PlanarEstimator:
    """Pose from wheel speed for distance and the gyro for heading."""

    def __init__(self) -> None:
        self._pose = PlanarPose(0.0, 0.0, 0.0)
        self._t_s: float | None = None

    @property
    def pose(self) -> PlanarPose:
        """The current estimate."""
        return self._pose

    def reset(self, start: PlanarPose, t_s: float) -> None:
        """Start from a known pose at time ``t_s``."""
        self._pose = start
        self._t_s = t_s

    def update(self, t_s: float, wheel_speed_mps: float, yaw_rate_radps: float) -> PlanarPose:
        """Advance to ``t_s`` with the latest wheel speed and gyro yaw rate.

        Integrates at the midpoint heading, which is exact for a constant-rate
        arc over the interval up to second order in the turned angle.
        """
        if self._t_s is None:
            self._t_s = t_s
            return self._pose
        dt = t_s - self._t_s
        self._t_s = t_s
        if dt <= 0.0:
            return self._pose
        dyaw = yaw_rate_radps * dt
        mid = self._pose.yaw + 0.5 * dyaw
        distance = wheel_speed_mps * dt
        yaw = math.atan2(math.sin(self._pose.yaw + dyaw), math.cos(self._pose.yaw + dyaw))
        self._pose = PlanarPose(
            self._pose.x + distance * math.cos(mid),
            self._pose.y + distance * math.sin(mid),
            yaw,
        )
        return self._pose
