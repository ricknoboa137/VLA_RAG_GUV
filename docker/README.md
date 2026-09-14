# Container environment

One image runs the ROS 2 side of CARMA on any machine with Docker: the robot's
computer, a laptop, or a CI runner. The `carma` library and the ROS workspace
are baked in, so a new machine needs only Docker and this repository.

| Image | Target | Contains |
|---|---|---|
| `carma-ros:jazzy` | `base` | ROS 2 Jazzy, carma library, `carma_bridge`, `carma_vision`, `carma_sim` |
| `carma-ros-sim:jazzy` | `sim` | `base` plus Gazebo Harmonic (`ros_gz`) |

Plus `eclipse-mosquitto:2` as the MQTT broker.

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
- Gazebo worlds (WP1). The `sim` profile builds the image but runs nothing.
