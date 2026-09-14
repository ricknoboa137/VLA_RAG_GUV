# Simulation

## Why Gazebo Harmonic

Open source end to end, runs on modest hardware, and scriptable in CI. Its
weaker photorealism matters for a learned VLA backbone, and that is a known
limitation to state in the paper rather than to hide: the field campaigns are
what establish real-world validity, and the simulation study establishes the
arbitration result under controlled conditions.

## Two adapters

`carma.sim` exposes one interface with two implementations:

- **`synthetic`** — no dependencies, renders a crude field scene from the pose.
  Runs the full loop in under a second. Used by the unit suite and CI. No
  scientific claim may rest on it.
- **`gazebo`** — the real target. Talks to a running Gazebo Harmonic instance
  through the ROS 2 bridge in `ros2_ws/src/carma_bridge`.

## Why `rclpy` is not a library dependency

The ROS node lives in `ros2_ws/`, not in `src/carma/`. The library talks to it
over a local transport. This keeps the unit suite runnable in a plain
virtualenv with no ROS installation, which is what makes CI cheap and what lets
a reviewer run the tests without a robotics stack.

## Bring-up

```bash
# terminal 1 — simulator
ros2 launch carma_sim row_crop.launch.py

# terminal 2 — bridge
ros2 run carma_bridge bridge_node

# terminal 3 — experiment
carma run --config configs/experiment/sim_study.yaml
```

## Status

The Gazebo adapter is a stub that raises with a clear message. Implement it
against the topic contract in `ros2_ws/src/carma_bridge/README.md`; until then
run with `sim.kind=synthetic`.
