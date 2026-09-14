"""A dependency-free world for tests and CI.

Renders a crude field scene from the robot's pose so that image-derived cues
vary sensibly as it moves. It exists so the full loop is exercised in under a
second with no simulator installed. No scientific claim may rest on it.
"""

from __future__ import annotations

import numpy as np

from carma.sim.registry import SIMS
from carma.types.core import Action, BgrImage, Observation, PhenologyStage, Pose

_H, _W = 120, 160


@SIMS.register("synthetic")
class SyntheticSim:
    """Kinematic unicycle in a rendered field.

    Args:
        dt: Control period in seconds.
        instruction: Instruction handed to the backbone each step.
        stage: Crop stage the episode is set in.
        seed: Unused at construction; ``reset`` takes the effective seed.
    """

    def __init__(
        self,
        dt: float = 0.1,
        instruction: str = "follow the row to the end and stop at the headland",
        stage: str = "VEGETATIVE",
        seed: int = 0,
    ) -> None:
        self._dt = dt
        self._instruction = instruction
        self._stage = PhenologyStage[stage]
        self._pose = Pose(0.0, 0.0, 0.0)
        self._t = 0.0
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int) -> Observation:
        """Start a new episode at the origin."""
        self._rng = np.random.default_rng(seed)
        self._pose = Pose(0.0, 0.0, 0.0)
        self._t = 0.0
        return self._observe()

    def step(self, action: Action) -> Observation:
        """Integrate one control period of unicycle kinematics."""
        yaw = self._pose.yaw + action.angular * self._dt
        x = self._pose.x + action.linear * float(np.cos(yaw)) * self._dt
        y = self._pose.y + action.linear * float(np.sin(yaw)) * self._dt
        self._pose = Pose(x, y, float(np.arctan2(np.sin(yaw), np.cos(yaw))))
        self._t += self._dt
        return self._observe()

    def close(self) -> None:
        """Nothing to release."""

    def _observe(self) -> Observation:
        return Observation(
            rgb=self._render(),
            depth=None,
            pose=self._pose,
            t=self._t,
            instruction=self._instruction,
            stage=self._stage,
        )

    def _render(self) -> BgrImage:
        """Render a deterministic scene that varies with pose.

        Sky in the upper third, soil below, and two vegetation bands whose
        position shifts with heading, so ``traversability_cue`` responds to
        where the robot is looking.
        """
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        img[: _H // 3] = (200, 190, 175)
        img[_H // 3 :] = (70, 90, 120)

        offset = int((self._pose.yaw / np.pi) * (_W // 2))
        for side in (-1, 1):
            centre = _W // 2 + side * (_W // 5) - offset
            lo = max(centre - _W // 12, 0)
            hi = min(centre + _W // 12, _W)
            if lo < hi:
                img[_H // 3 :, lo:hi] = (40, 130, 60)
        return img
