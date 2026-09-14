"""OpenCV-based scene description.

Nothing here learns. These are cheap, deterministic cues the backbone and the
hygiene policy use to describe a scene without pulling in a heavy model, and
they double as the visual descriptor attached to memory entries.
"""

from __future__ import annotations

from carma.perception.features import (
    green_fraction,
    scene_descriptor,
    texture_energy,
    traversability_cue,
)

__all__ = ["green_fraction", "scene_descriptor", "texture_energy", "traversability_cue"]
