"""Gazebo Harmonic adapter, driven over ROS 2.

The Python process here holds no ROS dependency of its own: the ROS 2 node
lives in ``ros2_ws/src/carma_bridge`` and exchanges messages with this adapter
over a local transport. Keeping ``rclpy`` out of the library is what lets the
unit suite run in a plain virtualenv with no ROS installation.

Bring-up, topic names and the world files are documented in
``docs/simulation.md``.
"""

from __future__ import annotations

from carma.sim.registry import SIMS
from carma.types import SimulatorError
from carma.types.core import Action, Observation


@SIMS.register("gazebo")
class GazeboAdapter:
    """Talks to a running Gazebo Harmonic instance through the ROS 2 bridge.

    Args:
        world: World file name under ``ros2_ws/src/carma_sim/worlds``.
        robot: Robot description name.
        control_hz: Rate at which actions are published.
        timeout_s: How long to wait for the bridge before failing.
    """

    def __init__(
        self,
        world: str = "row_crop.sdf",
        robot: str = "carma_ugv",
        control_hz: float = 10.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._world = world
        self._robot = robot
        self._control_hz = control_hz
        self._timeout_s = timeout_s

    def reset(self, *, seed: int) -> Observation:
        """Reset the world and return the first observation."""
        raise SimulatorError(self._todo())

    def step(self, action: Action) -> Observation:
        """Publish an action and return the next synchronised observation."""
        raise SimulatorError(self._todo())

    def close(self) -> None:
        """Shut the bridge down."""

    def _todo(self) -> str:
        return (
            f"Gazebo adapter is not implemented yet (world={self._world!r}, "
            f"robot={self._robot!r}). Implement it against the bridge described "
            "in docs/simulation.md; until then run with sim.kind=synthetic."
        )
