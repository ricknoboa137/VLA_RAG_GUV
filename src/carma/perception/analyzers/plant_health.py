"""Model-free plant-health cue.

A colour heuristic, not a trained model and not calibrated against ground
truth. It exists for two reasons: it gives the live pipeline something real to
publish before a learned model is trained, and it is the smallest complete
example of the :class:`~carma.types.SceneAnalyzer` contract. Replace it with an
``onnx_classifier`` entry once a model exists.

Hue windows are in OpenCV units, where hue spans ``[0, 180)``: yellow is near
30 and green near 60.
"""

from __future__ import annotations

import cv2
import numpy as np

from carma.perception.registry import ANALYZERS
from carma.types import AnalyzerError
from carma.types.core import AnalysisResult, BgrImage, Observation

_HUE_CHLOROTIC = (18, 34)
_HUE_GREEN = (34, 90)


def plant_health_scores(
    rgb: BgrImage,
    min_saturation: int = 40,
    min_value: int = 30,
) -> dict[str, float]:
    """Colour statistics of the vegetation in an image.

    Returns:
        ``vegetation_fraction``: share of pixels read as foliage, in ``[0, 1]``.
        ``green_fraction_of_vegetation``: share of foliage that is green.
        ``chlorotic_fraction_of_vegetation``: share of foliage that is yellow.
        ``excess_green_mean``: mean chromatic excess-green index over the
        whole image, ``2g - r - b`` on sum-normalised channels, in ``[-1, 2]``.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= min_saturation) & (val >= min_value)
    green = lit & (hue >= _HUE_GREEN[0]) & (hue <= _HUE_GREEN[1])
    chlorotic = lit & (hue >= _HUE_CHLOROTIC[0]) & (hue < _HUE_CHLOROTIC[1])

    n_pixels = float(hue.size)
    n_green = float(np.count_nonzero(green))
    n_chlorotic = float(np.count_nonzero(chlorotic))
    n_vegetation = n_green + n_chlorotic

    channels = rgb.astype(np.float32)
    total = channels.sum(axis=2)
    total[total == 0.0] = 1.0
    b, g, r = (channels[..., i] / total for i in range(3))
    exg = float((2.0 * g - r - b).mean())

    return {
        "vegetation_fraction": n_vegetation / n_pixels,
        "green_fraction_of_vegetation": n_green / n_vegetation if n_vegetation else 0.0,
        "chlorotic_fraction_of_vegetation": n_chlorotic / n_vegetation if n_vegetation else 0.0,
        "excess_green_mean": exg,
    }


@ANALYZERS.register("plant_health")
class PlantHealthAnalyzer:
    """Labels a scene ``healthy``, ``stressed`` or ``no_vegetation``.

    Args:
        min_saturation: Pixels less saturated than this (0-255) are not foliage.
        min_value: Pixels darker than this (0-255) are not foliage.
        min_vegetation: Below this vegetation fraction the label is
            ``no_vegetation``.
        stress_threshold: Above this chlorotic fraction of foliage the label is
            ``stressed``.
    """

    def __init__(
        self,
        min_saturation: int = 40,
        min_value: int = 30,
        min_vegetation: float = 0.02,
        stress_threshold: float = 0.25,
    ) -> None:
        for key, value in (
            ("min_vegetation", min_vegetation),
            ("stress_threshold", stress_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"plant_health {key} must be in [0, 1], got {value}"
                raise AnalyzerError(msg)
        self._min_saturation = min_saturation
        self._min_value = min_value
        self._min_vegetation = min_vegetation
        self._stress_threshold = stress_threshold

    @property
    def name(self) -> str:
        """Registry key."""
        return "plant_health"

    def analyze(self, obs: Observation) -> AnalysisResult:
        """Score the left image of ``obs``."""
        scores = plant_health_scores(obs.rgb, self._min_saturation, self._min_value)
        if scores["vegetation_fraction"] < self._min_vegetation:
            label = "no_vegetation"
        elif scores["chlorotic_fraction_of_vegetation"] > self._stress_threshold:
            label = "stressed"
        else:
            label = "healthy"
        return AnalysisResult(
            analyzer=self.name,
            t=obs.t,
            pose=obs.pose,
            label=label,
            scores=scores,
        )
