"""Scene analyzer registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import SceneAnalyzer

ANALYZERS: Registry[SceneAnalyzer] = Registry("analyzer")
