# Container environment

One image runs the ROS 2 side of CARMA on any machine with Docker: the robot's
computer, a laptop, or a CI runner. The `carma` library and the ROS workspace
are baked in, so a new machine needs only Docker and this repository.

| Image | Target | Contains |
|---|---|---|
| `carma-lib:0.1.0` | `lib` | **The Python library and CLI only.** No ROS. Start here. |
| `carma-dev:0.1.0` | `dev` | `lib` plus ruff, mypy and pytest, to run the gates |
| `carma-ros:jazzy` | `base` | ROS 2 Jazzy, carma library, `carma_bridge`, `carma_vision`, `carma_sim` |
| `carma-ros-sim:jazzy` | `sim` | `base` plus Gazebo Harmonic (`ros_gz`) |

Plus `eclipse-mosquitto:2` as the MQTT broker.

## Sharing the library with a colleague

They need Docker and this repository. Nothing else: no Python, no conda, no
ROS.

```bash
git clone <this repo> && cd VLA_RAG_GUV
make docker-build     # builds carma-lib and carma-dev
make docker-check     # lint, types, 145 tests, pin gate — inside the image
make docker-smoke     # a real experiment; writes to ./runs on the host
```

`make docker-check` runs the same `Makefile` the maintainers run locally, so
there is one definition of what "passing" means rather than two that can
drift.

### What makes it reproducible

Pinning the direct dependencies is not enough. Four things were missing and
are now in place:

- **The transitive dependencies are locked.** `pyproject.toml` pins what this
  project chooses; `requirements-lib.lock` and `requirements-dev.lock` pin
  everything those pull in, resolved on linux/amd64 for CPython 3.11. The
  install uses `--no-deps`, so the resolver cannot add anything the lock does
  not name. Without this, `packaging`, `rich` and `Pygments` floated to
  whatever was newest on build day.
- **The base image is pinned by digest**, not by the `python:3.11-slim-bookworm`
  tag, which upstream rebuilds. A tag means the same Dockerfile yields a
  different image next month. Regenerate both with `make lock` after changing
  `pyproject.toml`.
- **The build fails rather than ships a broken library.** The image runs
  `carma components` and imports the CLI at build time. The ROS image installs
  the library with `--no-deps` on purpose, so its CLI dependencies are absent
  there; that is fine for running nodes and useless for running experiments,
  which is why `lib` exists separately.
- **The build context is 2.9 MB.** It was 9.9 GB, because `.dockerignore` did
  not exclude the Unity project's 57,000 files.

### Results match across platforms

The same experiment, same seed, run in the Linux container and on a Windows
host, produced the same config hash and byte-identical decision logs. A
result from a colleague's container is therefore comparable with one from
this machine — and each manifest records its platform, so if that ever stops
being true it is visible rather than silent.

Re-check it after any dependency change:

```bash
make docker-smoke
carma run --config configs/experiment/smoke.yaml --seed 0   # on the host
```

Then compare `decisions.jsonl` between the two runs, ignoring `run_id` and
`timestamp`.

## Prerequisites

- **Linux (robot, recommended):** Docker Engine with the compose plugin.
- **Windows:** Docker Desktop on the WSL 2 backend. USB cameras are not passed
  into containers on Windows without extra tooling (`usbipd-win`), so treat
  Windows as a development and viewing host, not as the robot.

The ROS base images are published for amd64 and arm64, so the same files build
on an x86 laptop and an ARM robot computer (Jetson, Raspberry Pi 5).

## Use

All commands run from the repository root.

```bash
docker compose -f docker/compose.yaml build
```

```bash
docker compose -f docker/compose.yaml up broker vision
```

```bash
docker compose -f docker/compose.yaml run --rm shell
```

Inside `shell`, the usual tools work: `ros2 topic list`, `colcon build`,
`colcon test --packages-select carma_vision`. `ros2_ws/src` and `configs/` are
mounted, so host edits are visible; run `colcon build` in the shell to pick up
node changes. The `vision` service mounts `configs/` and `assets/`, so a new
analyzer config or ONNX model needs a restart, not a rebuild.

**`shell` is the exception, not the rule.** Every other service — `sim`,
`bridge`, `vision` — runs the workspace that was compiled into the image, not
`ros2_ws/src` on the host. That is deliberate: a run should execute the code of
a known commit rather than whatever happens to be on disk. The cost is that
editing anything under `ros2_ws/src` has no effect on those services until you
rebuild:

```bash
docker compose -f docker/compose.yaml build sim
```

The failure mode is quiet. The service starts, publishes, and looks healthy
while running the previous version of the code, so verify a change against the
service you actually ship — not only in `shell`, where the mount hides it.

Watch the MQTT output from any machine on the network (host `broker` inside
compose, the Docker host's IP from outside):

```bash
docker compose -f docker/compose.yaml exec broker mosquitto_sub -t 'carma/#' -v
```

## Testing on a PC with a webcam

Docker Desktop cannot pass a USB camera into a container, so the webcam is read
on the host and sent through the broker instead:

```bash
MQTT_CAMERA=true docker compose -f docker/compose.yaml up --build vision viewer
```

```bash
python scripts/webcam_mqtt.py --camera 0
```

(On Windows PowerShell, set the variable first: `$env:MQTT_CAMERA="true"`.)
Open http://localhost:8088 for the stream and live analysis results (set
`VIEWER_PORT` to use another port). One webcam
is duplicated into both eyes; `--right-camera 1` sends a real pair.

### Ports

By default the stack uses the MQTT broker already running on the host, reached
from containers as `host.docker.internal:1883` (override with `MQTT_HOST`). The
browser viewer needs that broker to have a WebSocket listener; the viewer takes
`?port=` for its port.

Machines without a broker can start the bundled one instead. It is published on
host ports **1884** (MQTT) and **9002** (WebSocket) so it never collides with a
host broker:

```bash
MQTT_HOST=broker docker compose -f docker/compose.yaml --profile local-broker up
```

## Speech commands on a PC

The `voice` service transcribes operator speech offline. Fetch the model once,
start the service, then run the microphone script on the host:

```bash
python scripts/fetch_models.py faster-whisper-base
docker compose -f docker/compose.yaml up -d --build voice
python scripts/mic_mqtt.py --list-devices
python scripts/mic_mqtt.py --device <index>
```

Set `STT_LANGUAGE=en` or `es` to stop auto-detection. See `docs/speech.md`.

## Large images and DDS

Raw camera frames are big (640×480 BGR is 921,600 bytes; a side-by-side pair
twice that). Fast DDS's default transports drop such messages when delivery is
best-effort: they overflow the shared-memory segment and fall back to UDP
fragments into a 208 KB receive buffer. Measured in this container at 15 Hz:

| Image | Default transports | `LARGE_DATA` | `fastdds_shm.xml` |
|---|---|---|---|
| 320×240 (230 KB) | 15.0 Hz | 15.1 Hz | 15.0 Hz |
| 640×480 (922 KB) | 2.1 Hz | 8.6 Hz | 15.0 Hz |
| 1280×480 (1.8 MB) | — | — | 15.0 Hz |

The image therefore ships `docker/fastdds_shm.xml` (16 MB shared-memory
segment, UDP buffers at the 4 MB kernel cap) and loads it through
`FASTRTPS_DEFAULT_PROFILES_FILE`. Symptoms if it is missing: a camera node that
publishes at full rate while subscribers and the viewer see a few fps. Keep
`ipc: host` on every ROS service, or containers cannot share memory.

## Networking

Containers share one compose network, and `ROS_DOMAIN_ID` (default 42) must
match on every ROS participant. To reach ROS nodes running outside Docker on a
Linux host, add `network_mode: host` to the service instead.

## What is not here yet

- The stereo camera driver, pending the camera choice. See the commented
  `camera` service in `compose.yaml`.
- The link from the simulator to experiments. The `sim` profile runs the
  row-crop world and publishes the topic contract, but `carma_bridge` does not
  yet turn those topics into observations for `carma run`.
