"""The scientific core: pricing and choosing between act, retrieve and ask.

This is the only package that may import both ``carma.memory`` and
``carma.backbone``. Everything that makes one experimental condition differ
from another lives here.
"""

from __future__ import annotations

from carma.arbiter.cost_model import COST_MODELS, LinearCostModel
from carma.arbiter.policies import ARBITERS

__all__ = ["ARBITERS", "COST_MODELS", "LinearCostModel"]
