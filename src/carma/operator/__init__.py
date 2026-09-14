"""The channel to the human operator.

Every implementation must report honest latencies. Those latencies are not
telemetry: they are the interaction-economy measurements the project exists to
make, and a channel that under-reports them invalidates the headline result.
"""

from __future__ import annotations

from carma.operator.registry import OPERATORS
from carma.operator.scripted import ScriptedOperator

__all__ = ["OPERATORS", "ScriptedOperator"]
