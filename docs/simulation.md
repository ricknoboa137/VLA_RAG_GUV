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

The vehicle is a tractor: rear-wheel drive, Ackermann front steering, matching
the physical platform. Two things follow that the arbitration study has to
respect. It cannot turn on the spot — minimum turning radius is 0.83 m against
a narrowest alley of 1.00 m, so turns happen at the headland. And `angular.z`
does nothing without `linear.x`, because steering authority comes from forward
motion. A backbone that emits pivot-in-place actions will appear to fail for
reasons that have nothing to do with arbitration.

The SO-101 arm can ride on the front (`arm:=true`, off by default). It is not
part of any experimental condition; see `docs/arm.md`.

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

## Running `carma run` against Gazebo

`GazeboAdapter` (`sim.kind: gazebo`) drives the world in **lockstep**, which is
what makes a Gazebo run reproducible (AGENTS.md section 2.2). The world starts
paused (`lockstep:=true`); each `step` sends one velocity command, `carma_bridge`
advances the world by exactly `1/control_hz` simulated seconds through Gazebo's
world-control service, and only then returns the observation. Nothing depends
on wall-clock time, so a slow software renderer costs time, not correctness.

Bring-up, in three shells (or `LOCKSTEP=true docker compose -f docker/compose.yaml
--profile sim --profile bridge up`):

```bash
ros2 launch carma_sim row_crop.launch.py lockstep:=true stage:=VEGETATIVE
ros2 run carma_bridge bridge_node            # TCP 5555; published by the bridge compose service
carma run --config configs/experiment/gazebo_smoke.yaml
```

`stage:=` must match `sim.params.stage` in the config: the world's crop stage is
fixed at launch and the config only labels the observations.

What the adapter and bridge guarantee, and how it was checked:

- **Stamps, not arrival order.** The reply holds the newest camera frame and
  odometry message stamped at or before the step's target time; the bridge
  waits for the clock to reach the target and for sensor traffic to go quiet
  (`quiet_s`, 0.15 s of wall time) before choosing. Wall time decides only when
  to stop waiting. `image_t` and `pose_t` in the reply say which stamps were used.
- **A 2 s settle after reset** (`settle_iterations`, 2000). "Reset all" restores
  poses but not the vehicle's whole dynamic state: with a 0.1 s settle the first
  frame of an episode depended on how the previous episode ended (five distinct
  frames after episodes of 30, 5, 30, 0 and 12 steps). After the longer settle
  all five were identical. Episode time `t = 0` is the end of the settle.
- **Replay check, run on this commit** (software rendering, 640x480): three
  episodes of 30 steps with identical actions were bit-identical in pose, left
  image and depth (31 frames each); two complete `gazebo_smoke` experiments
  (200 steps each) produced identical step records. Repeat the first check with
  `python scripts/gazebo_replay_check.py --episodes 3 --steps 30` after changing
  the bridge, the world, the physics step, `command_settle_s` or the sensor
  rates; it exits 1 and lists the differing frames if replay breaks.
- **Known limit.** `ros_gz_bridge` gives no acknowledgement for the velocity
  command, so the bridge pauses `command_settle_s` (0.05 s of wall time) before
  stepping. Too short a pause would let a step run on the previous command;
  none was seen in the replays above, which is evidence, not a proof.

Cost: about 0.4 s of wall time per 0.1 s control step with software rendering
(77 s for a 200-step episode), plus 2 s of simulated time at each reset.

## Routes, start poses and the expert

An experiment config may carry an optional `route:` block. Without it, episodes
come from `synthetic_episodes` (goals on a ring, for the synthetic simulator);
`route: {kind: row_alley}` generates down-the-alley episodes for this world
instead (`carma.runtime.episodes.row_alley_episodes`): the goal is a whole
number of metres along the clear y = 0 alley, the instruction names it ("drive
13 metres down the row and stop"), and the robot starts with a seeded lateral
offset (default +/-0.25 m) and heading error (+/-0.2 rad). The goal is used for
scoring only and never reaches the policy. A config without `route` hashes as it
did before the field existed.

`SimAdapter.reset` now takes the episode's start pose, and the episode loop
stamps the spec's instruction onto every observation, because both belong to the
episode, not the simulator. In Gazebo the bridge moves the robot there with the
`set_pose` service and reports poses in the world frame (odometry starts at zero
at the start pose, so the bridge composes the two). Placing the robot at the
origin took the same path as before; starting at (0.0, 0.3, 0.25) and driving
straight moved the robot along that heading, and a later reset to the origin
reproduced the first episode's trajectory exactly.

`backbone: {kind: alley_expert}` is a scripted lane keeper that reads the pose
and the instruction. It is the ceiling for learned policies and the source of
navigation demonstrations, not a contribution. Evaluated in Gazebo
(`configs/experiment/gazebo_alley_expert.yaml`, VEGETATIVE, four episodes of
10 to 14 m): 4 of 4 reached the goal radius, lateral excursion at most 0.19 m,
about 80 s of wall time per episode. An episode ends on entering the success
radius, so the final distance to the goal is just under it (1.48 m mean).
`configs/experiment/gazebo_smoke.yaml`, with the heuristic backbone, reaches
about 3 m of a 10 to 16 m route and succeeds on none, which is the floor to beat.

## Driving the platform by hand and by voice

`scripts/drive_sim.py` opens a window with the platform's left-camera view and
steps the lockstep world from the keyboard and from spoken commands:

```bash
python scripts/drive_sim.py              # keyboard + speech
python scripts/drive_sim.py --no-voice   # keyboard only
```

Keys (click the window first): `W`/`S` forward and back, `A`/`D` turn, `Q`/`E`
forward-and-turn, `Space` or `X` stop, `R` back to the start pose, `Esc` quit.
Speech comes from the `carma_voice` utterance topic (feed it with
`scripts/mic_mqtt.py`): "go" latches forward, "turn left/right" is a timed turn,
"stop" clears everything. Longer remarks are suggestions: they do not drive,
are shown on screen, and are appended to `assets/drive_suggestions.jsonl`.

A held key beats speech; stop beats everything. Timing is counted in control
steps, so a slow renderer makes the platform respond later in wall time, not
differently in simulated time. The bridge serves one client, so close any other
run or recording first. The mapping lives in `carma.sim.teleop` and is covered
by `tests/test_teleop.py`; the window itself is not tested.
