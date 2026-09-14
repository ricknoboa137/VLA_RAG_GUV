"""A model-free backbone used to exercise the harness.

This is scaffolding, not a contribution. It produces plausible actions and a
plausible uncertainty from OpenCV cues alone, so the episode loop, the metrics
and the manifests can be validated before a real VLA checkpoint is integrated.

Replacing it means implementing :class:`~carma.types.NavigationBackbone` in a
sibling module and changing one line of config. Whatever replaces it must
return uncertainty in ``[0, 1]``; calibrate inside the adapter and document how.
"""

from __future__ import annotations

import numpy as np

from carma.backbone.registry import BACKBONES
from carma.perception import traversability_cue
from carma.types.core import Action, Observation

_TURN_KEYWORDS = {"left": 1.0, "right": -1.0}


@BACKBONES.register("heuristic")
class HeuristicBackbone:
    """Drives forward where the ground looks traversable, turning otherwise.

    Args:
        max_linear: Forward speed cap in metres per second.
        max_angular: Yaw rate cap in radians per second.
        noise: Standard deviation of action noise, in the same units. Draws
            come from the caller's generator, never a module-level RNG.
    """

    def __init__(
        self,
        max_linear: float = 0.6,
        max_angular: float = 0.8,
        noise: float = 0.02,
    ) -> None:
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._noise = noise
        self._steps = 0

    def reset(self) -> None:
        """Clear per-episode state."""
        self._steps = 0

    def act(
        self,
        obs: Observation,
        context: tuple[str, ...] = (),
        *,
        rng: np.random.Generator,
    ) -> tuple[Action, float]:
        """Return an action and an uncertainty in ``[0, 1]``.

        Retrieved context biases the turn direction when it mentions one, which
        is a deliberately crude stand-in for context conditioning: it makes the
        effect of retrieval visible in the harness without pretending to be a
        learned mechanism.
        """
        self._steps += 1
        traversable = traversability_cue(obs.rgb)

        bias = 0.0
        for text in context:
            lowered = text.lower()
            for word, direction in _TURN_KEYWORDS.items():
                if word in lowered:
                    bias += direction
        bias = float(np.clip(bias, -1.0, 1.0))

        linear = self._max_linear * traversable
        angular = self._max_angular * (bias if bias else (0.5 - traversable))

        linear += float(rng.normal(0.0, self._noise))
        angular += float(rng.normal(0.0, self._noise))

        # Confident where the ground reads clearly one way or the other;
        # uncertain in the ambiguous middle, which is where a field robot
        # actually needs help.
        uncertainty = float(np.clip(1.0 - 2.0 * abs(traversable - 0.5), 0.0, 1.0))
        if context:
            uncertainty *= 0.7

        action = Action(
            linear=float(np.clip(linear, 0.0, self._max_linear)),
            angular=float(np.clip(angular, -self._max_angular, self._max_angular)),
            stop=False,
        )
        return action, uncertainty
