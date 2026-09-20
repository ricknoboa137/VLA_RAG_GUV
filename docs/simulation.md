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

## The world

`ros2_ws/src/carma_sim` holds the row-crop field and the UGV description. The
full geometry — stereo baseline, intrinsics, depth range, wheel parameters — is
tabulated in that package's README; it is not repeated here, so that there is
one place to change it.

Two things about it are load-bearing for the research design:

**Crop stage is a world argument, not a forked file.** The drift study runs the
same routes at two phenological stages. `worlds/row_crop.sdf` is a xacro
template expanded at launch:

```bash
ros2 launch carma_sim row_crop.launch.py stage:=VEGETATIVE
ros2 launch carma_sim row_crop.launch.py stage:=CANOPY_CLOSURE
```

`stage` takes the names of `carma.types.PhenologyStage` and is validated
against them, so the two runs differ by one recorded parameter rather than by
an untracked edit. Row spacing, row count, obstacle placement and lighting are
identical across stages; only the canopy changes. The expanded world is written
to `~/.ros/log/carma_sim/row_crop_<stage>.sdf` and kept, so the exact scene an
episode ran against is recoverable after the fact. When `GazeboAdapter` lands
it records `stage` in the run manifest alongside the config hash.

**Nothing is fetched from Fuel.** Ground, lighting, crop and obstacles are all
inline in the world file. AGENTS.md section 2.3 forbids network access at run
time, and a world that silently downloads a mesh produces different numbers on
an offline machine.

## Bring-up

```bash
# terminal 1 — simulator (world + ros_gz bridge, headless)
ros2 launch carma_sim row_crop.launch.py stage:=VEGETATIVE

# terminal 2 — bridge
ros2 run carma_bridge bridge_node

# terminal 3 — experiment
carma run --config configs/experiment/sim_study.yaml
```

Or, entirely in the container that already has Gazebo:

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

`headless:=false` opens the Gazebo GUI instead of running server-only; it needs
a display, which the container does not have by default.

## Status

The world, the UGV description and the launch file are in place. Headless
bring-up publishes all four robot→bridge topics steadily and a `/carma/cmd_vel`
command moves the robot in `/carma/odom`.

Nominal rates are 15 Hz for the cameras and 30 Hz for odometry. On a CPU-only
host with software rendering they settle around 10 Hz and 20 Hz, because the
render loop cannot hold real-time factor 1; a GPU host reaches nominal. That
rate is worth recording per run, since it sets how many decision points an
episode had.

Not yet implemented, and both needed before an episode can run against Gazebo:

- `GazeboAdapter` in `src/carma/sim/gazebo.py` is a stub that raises with a
  clear message. Implement it against the topic contract in
  `ros2_ws/src/carma_bridge/README.md`.
- `carma_bridge`'s approximate-time synchroniser, which turns the four topics
  into an `Observation`.

Until both land, run with `sim.kind=synthetic`.
