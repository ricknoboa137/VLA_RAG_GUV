"""Bridge node skeleton.

Owns every ROS import in the project. Subscribes to the robot's sensors,
publishes velocity commands, and exposes synchronised observations to the
``carma`` library.

Implement against the topic contract in this package's README. Keep the
synchronisation slop as a declared parameter and record it in the run manifest
once the adapter lands: a loose slop silently degrades every pose-conditioned
memory entry, and that failure is invisible in the task metrics.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for ``ros2 run carma_bridge bridge_node``."""
    msg = (
        "carma_bridge is a skeleton. Implement it against the topic contract "
        "in ros2_ws/src/carma_bridge/README.md. Until then run experiments "
        "with sim.kind=synthetic."
    )
    raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
