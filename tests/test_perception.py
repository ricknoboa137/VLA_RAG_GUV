"""OpenCV cues are bounded, deterministic and responsive."""

from __future__ import annotations

import numpy as np
import pytest

from carma.perception import green_fraction, scene_descriptor, texture_energy, traversability_cue
from carma.types.core import Observation


def test_green_fraction_detects_vegetation(observation: Observation) -> None:
    """The fixture is vegetated over its lower half."""
    assert 0.4 < green_fraction(observation.rgb) < 0.6


def test_cues_are_bounded(observation: Observation) -> None:
    """Every cue is in [0, 1] so downstream weighting stays interpretable."""
    for value in (
        green_fraction(observation.rgb),
        texture_energy(observation.rgb),
        traversability_cue(observation.rgb),
    ):
        assert 0.0 <= value <= 1.0


def test_descriptor_is_unit_norm(observation: Observation) -> None:
    """Descriptors are normalised so cosine similarity is meaningful."""
    vec = scene_descriptor(observation.rgb)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_cues_are_deterministic(observation: Observation) -> None:
    """Same image, same number, every time."""
    assert traversability_cue(observation.rgb) == traversability_cue(observation.rgb)
