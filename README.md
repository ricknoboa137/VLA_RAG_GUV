# CARMA

**Cost-Aware Retrieval Memory Arbitration for field robots.**

CARMA wraps a vision-language-action navigation backbone with a memory of past
human corrections and an arbitration policy that decides, at each step, whether
to act, to retrieve from that memory, or to ask the operator. Operator attention
is treated as a measured cost rather than a free oracle.

The system targets a wheeled UGV in outdoor agricultural
environments, with Gazebo Harmonic and ROS 2 for simulation.

**New here? Start with [HOW_TO_USE.md](HOW_TO_USE.md)**: how to run every part
of the system, alone and together, and what every file in the repository does
and why it is needed.

## Status

Research prototype, running end to end in Gazebo with a learned backbone.

- **Backbone:** a navigation SmolVLA fine-tuned on alley demonstrations, served
  by `scripts/policy_server.py` (`docs/vla-integration.md`, `docs/backbones.md`).
- **Memory and arbitration:** text and spatial retrieval, stage-aware hygiene,
  and the cost model of `docs/cost-model.md`, including remembered warnings as
  risk evidence (`memory_risk`).
- **The five conditions** run in Gazebo from config files; results and the
  commits that produced them are in `training/results/RESULTS.md`.
- **Hidden hazards** (`docs/hazards.md`): the scenario in which the choice
  matters. In Gazebo, carma matches unconditional retrieval (7/8) with 17%
  fewer retrievals; the backbone alone and ask-only enter the hazard every time.
- **Robot readiness** (`docs/robot-readiness.md`): safety gate, pose from the
  robot's own sensors and stall detection are built; the robot interface and
  on-demand operator questions are not.

The synthetic simulator (`configs/experiment/smoke.yaml`) remains for fast
tests; its placeholder backbone is not goal-directed, so a success rate of zero
there is expected.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Run an episode

```bash
carma run --config configs/experiment/smoke.yaml --seed 0
carma report runs/
```

Every run writes `runs/<run_id>/manifest.yaml` recording config hash, git
commit, seed, platform and dependency versions. `carma report` refuses to
aggregate runs whose manifests disagree.

## Run the simulation

A row-crop field with the tractor UGV, in Docker, no hardware required:

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

It comes up headless and publishes the robot side of the topic contract —
stereo pair, depth, odometry, IMU — and takes drive requests on
`/carma/cmd_vel_request`, which the safety gate passes to `/carma/cmd_vel`. Step-by-step
instructions, including how to drive it, view the cameras and open the Gazebo
window, are in section 8 of `HOW_TO_USE.md`. The world and robot geometry are
documented in `ros2_ws/src/carma_sim/README.md`. `arm:=true` adds the SO-101
pick-up arm on the front of the vehicle (`docs/arm.md`).

Experiments run against it with `sim.kind: gazebo` in lockstep mode; see
`docs/simulation.md` and `configs/experiment/gazebo_study/` and
`configs/experiment/hazard_study/`.

## Conditions

| Key | What it isolates |
|---|---|
| `no_memory` | Backbone alone — the floor |
| `unconditional_retrieval` | Retrieval on every step |
| `ask_only` | Uncertainty-gated asking, no memory |
| `carma` | Full arbitration |
| `carma_no_hygiene` | Arbitration without staleness handling |

## Repository layout

```
src/carma/        library — see AGENTS.md for module boundaries
ros2_ws/          ROS 2 packages for the physical and simulated robot
docker/           container image and compose file for the ROS 2 side
vr/unity/         Meta Quest 2 client (Unity 6, Meta XR SDK) — see vr/unity/README.md
configs/          every experiment is a config file, not a CLI flag
config/           stereo camera calibration
experiments/      recorded measurements behind published figures
docs/             architecture, cost model, hazards, simulation, robot readiness
training/         fine-tuning, dataset builders, splits and recorded results
tests/            unit suite, plus architecture and determinism gates
scripts/          host-side tools: camera, microphone, policy server, stereo, gates
```

## On the robot

The ROS 2 side runs in Docker on any machine (`docker/README.md`). Section 4
of `HOW_TO_USE.md` starts the camera, analysis, viewer and speech together:

```bash
MQTT_CAMERA=true docker compose -f docker/compose.yaml up -d vision voice viewer
```

`carma_vision` runs scene analyzers on the live camera — your own models plug in
through `configs/analyzers/` (`docs/analyzers.md`) — and streams the stereo
camera over MQTT. VR viewing is covered in `docs/streaming.md`. `carma_voice`
turns operator speech into commands, answers and corrections with offline
speech-to-text (`docs/speech.md`).

## Contributing

Read `AGENTS.md` first. It is the development contract and it is enforced in CI.

## Licence

Apache-2.0. See `LICENSE`.
