"""Scene analyzers: the extension point for your own perception models.

Importing this package registers every shipped analyzer. To add one, see
``docs/analyzers.md``.
"""

from __future__ import annotations

from carma.perception.analyzers import onnx_classifier, plant_health
from carma.perception.registry import ANALYZERS

__all__ = ["ANALYZERS", "onnx_classifier", "plant_health"]
