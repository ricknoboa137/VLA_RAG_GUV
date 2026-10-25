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

Interfaces, experiment harness, and core adapters are in place. The library can
run end-to-end against:
- **Synthetic simulator** (no dependencies, CI-ready)
- **Gazebo Harmonic** via `carma_bridge` (ROS 2, requires Docker or ROS 2 install)
- **Remote VLA backbone** via HTTP policy server
- **Quest 2 voice operator** via MQTT

## Install

```bash
# Prerequisites: Python 3.11+ (Anaconda recommended on Windows)
git clone https://github.com/your-org/VLA_RAG_GUV.git
cd VLA_RAG_GUV
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate                              # Windows CMD
# .venv\Scripts\Activate.ps1                          # Windows PowerShell

pip install -e ".[dev,operator,stream,voice]"  # all optional deps
# or selectively:
# pip install -e ".[dev,operator]"   # for Gazebo + MQTT operator
# pip install -e ".[dev,stream]"     # for webcam streaming
# pip install -e ".[dev,voice]"      # for speech-to-text

make check
```

### Windows (Anaconda) Quick Start

```powershell
# Use the Anaconda Python explicitly (not the Microsoft Store stub)
C:\Users\<you>\anaconda3\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,operator,stream,voice]"
make check
```

## Run an Episode (Synthetic)

```bash
carma run --config configs/experiment/smoke.yaml --seed 0
carma report runs/
```

Every run writes `runs/<run_id>/manifest.yaml` recording config hash, git
commit, seed, platform and dependency versions. `carma report` refuses to
aggregate runs whose manifests disagree.

## Run an Episode (Gazebo + MQTT Voice + Remote VLA) — End-to-End

### 1. Start Gazebo + ROS 2 Bridge (Terminal 1)
```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
# OR on a machine with ROS 2 Humble/Harmonic installed:
ros2 launch carma_sim row_crop.launch.py stage:=VEGETATIVE
```

### 2. Start carma_bridge Node (Terminal 2)
```bash
# Synchronises stereo, depth, odom → /carma/observation (TCP 5555)
ros2 run carma_bridge bridge_node
```

### 3. Start Policy Server (Terminal 3)
```bash
# Echo policy (no GPU, validates plumbing)
python scripts/policy_server.py --policy echo --self-check

# Real VLA (needs carma-train env + checkpoint):
# & $env:USERPROFILE/anaconda3/envs/carma-train/python.exe scripts/policy_server.py --policy smolvla --self-check
```

### 4. Start MQTT Broker (if not already running)
```bash
# Linux/macOS
mosquitto -c /etc/mosquitto/mosquitto.conf

# Windows
mosquitto.exe -c mosquitto.conf
```

### 5. Run Experiment (Terminal 4)
```bash
carma run --config configs/experiment/e2e_gazebo_mqtt_remote.yaml
```

## Run the Simulation (Docker)

A row-crop field with the tractor UGV, in Docker, no hardware required:

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

It comes up headless and publishes the robot side of the topic contract —
stereo pair, depth, odometry — and drives on `/carma/cmd_vel`. Step-by-step
instructions, including how to drive it, view the cameras and open the Gazebo
window, are in section 8 of `HOW_TO_USE.md`. The world and robot geometry are
documented in `ros2_ws/src/carma_sim/README.md`.

## Conditions

| Key | What it isolates |
|---|---|
| `no_memory` | Backbone alone — the floor |
| `unconditional_retrieval` | Retrieval on every step |
| `ask_only` | Uncertainty-gated asking, no memory |
| `carma` | Full arbitration |
| `carma_no_hygiene` | Arbitration without staleness handling |

## Repository Layout

```
src/carma/        library — see AGENTS.md for module boundaries
ros2_ws/          ROS 2 packages for the physical and simulated robot
docker/           container image and compose file for the ROS 2 side
vr/unity/         Meta Quest 2 client (Unity 6, Meta XR SDK) — see vr/unity/README.md
configs/          every experiment is a config file, not a CLI flag
config/           stereo camera calibration
experiments/      recorded measurements behind published figures
docs/             architecture, interfaces, cost model, data schema
tests/            unit suite, plus architecture and determinism gates
scripts/          host-side tools: camera, microphone, policy server, stereo, gates
```

## On the Robot

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

## Quest 2 Voice Integration

The Meta Quest 2 app (`vr/unity/CarmaQuest/`) provides push-to-talk speech:
1. Hold **B button**, speak, release
2. Audio → MQTT `carma/operator/audio`
3. `carma_voice` container (faster-whisper) transcribes → `carma/operator/utterance`
4. CARMA `MqttOperator` consumes utterances with `CORRECTION` intent as memory entries

See `vr/unity/README.md` for build/run instructions.

## Experiment Configs

| Config | Purpose |
|---|---|
| `configs/experiment/smoke.yaml` | Fast CI check (synthetic, heuristic backbone) |
| `configs/experiment/remote_smoke.yaml` | Policy server plumbing (synthetic, remote backbone) |
| `configs/experiment/sim_study.yaml` | Paper simulation study (100 episodes) |
| `configs/experiment/e2e_gazebo_mqtt_remote.yaml` | **Full stack: Gazebo + MQTT voice + remote VLA** |

## Contributing

Read `AGENTS.md` first. It is the development contract and it is enforced in CI.

## Licence

Apache-2.0. See `LICENSE`.

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
docs/             architecture, interfaces, cost model, data schema
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
