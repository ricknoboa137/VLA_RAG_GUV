"""Simulator and robot adapters behind one interface.

``synthetic`` needs nothing installed and runs in CI. ``gazebo`` targets Gazebo
Harmonic through ROS 2 and is the platform the experiments actually use.
"""

from __future__ import annotations

from carma.sim.gazebo import GazeboAdapter
from carma.sim.registry import SIMS
from carma.sim.synthetic import SyntheticSim

__all__ = ["SIMS", "GazeboAdapter", "SyntheticSim"]
