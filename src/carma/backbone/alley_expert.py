"""A scripted expert for the row-crop alley, used to make demonstrations.

Not a learned policy and not a contribution: it is a lane-keeping controller
that reads the pose and the instruction, the same two inputs a learned policy
gets over the wire besides the image. It has two jobs. It generates the
demonstrations a navigation VLA is fine-tuned on, and it gives the ceiling that
learned policies are compared against in the same world and routes.

It drives toward the alley centreline (y = 0, heading 0), at constant speed,
and stops when the odometer reaches the distance the instruction names. The
instruction must contain one, e.g. "drive 14 metres down the row and stop".
"""

from __future__ import annotations

import re

import numpy as np

from carma.backbone.registry import BACKBONES
from carma.types import BackboneError
from carma.types.core import Action, Observation

_DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:metres|meters|m)\b", re.IGNORECASE)


@BACKBONES.register("alley_expert")
class AlleyExpert:
    """Proportional lane keeping with a distance-triggered stop.

    Args:
        linear_mps: Forward speed in metres per second.
        k_lateral: Yaw rate in rad/s per metre of lateral error.
        k_heading: Yaw rate in rad/s per radian of heading error.
        max_angular: Yaw-rate cap in rad/s.
        stop_margin_m: Stop this far before the named distance, to allow for
            the vehicle coasting; the success radius is larger than this.
    """

    def __init__(
        self,
        linear_mps: float = 0.5,
        k_lateral: float = 1.0,
        k_heading: float = 1.5,
        max_angular: float = 0.8,
        stop_margin_m: float = 0.3,
    ) -> None:
        self._linear = linear_mps
        self._k_lateral = k_lateral
        self._k_heading = k_heading
        self._max_angular = max_angular
        self._stop_margin = stop_margin_m

    def reset(self) -> None:
        """Nothing is carried between episodes."""

    def act(
        self,
        obs: Observation,
        context: tuple[str, ...] = (),
        *,
        rng: np.random.Generator,
    ) -> tuple[Action, float]:
        """Return the lane-keeping action and an uncertainty of 0 (the expert is sure)."""
        match = _DISTANCE.search(obs.instruction)
        if match is None:
            msg = f"the instruction names no distance in metres: {obs.instruction!r}"
            raise BackboneError(msg)
        if obs.pose.x >= float(match.group(1)) - self._stop_margin:
            return Action(linear=0.0, angular=0.0, stop=True), 0.0
        turn = -self._k_lateral * obs.pose.y - self._k_heading * obs.pose.yaw
        angular = float(np.clip(turn, -self._max_angular, self._max_angular))
        return Action(linear=self._linear, angular=angular), 0.0
