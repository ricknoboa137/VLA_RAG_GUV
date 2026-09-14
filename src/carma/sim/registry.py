"""Simulator registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import SimAdapter

SIMS: Registry[SimAdapter] = Registry("sim")
