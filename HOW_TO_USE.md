# How to use CARMA

A step-by-step guide to running the system on a development PC: the ROS 2
containers, the camera stream with live analysis, operator speech, the Meta
Quest 2 client, and the Gazebo simulation. For what the project is and its
design rules, see `README.md` and `AGENTS.md`.

If you have no hardware to hand, start at section 7: the simulation needs only
Docker and shows the robot side of the system working end to end.

```
 webcam / stereo camera ──► ROS 2 vision container ──► analysis results + stereo stream ──┐
                                                                                         MQTT broker ◄──► viewer / Quest 2
 microphone / Quest mic ──► speech clips ──► voice container (offline speech-to-text) ────┘
```

---

## 1. Requirements

| Needed for | Install |
|---|---|
| Everything | Git, and this repository cloned |
| ROS 2 side | Docker Desktop (Windows, WSL 2 backend) or Docker Engine (Linux) |
| Host scripts | Python 3.11 (e.g. a conda env), then `pip install -e ".[dev,stream,voice]"` |
| MQTT | A Mosquitto broker on port 1883 (the stack uses the one on the host) |
| Quest client | Unity 6000.3.24f1 with Android Build Support, a Meta Quest 2 in developer mode |

Model weights are downloaded once into `assets/` (never committed):

```bash
python scripts/fetch_models.py faster-whisper-base
```

## 2. Check the library

```bash
make check
```

or, without `make`: `ruff check src tests scripts`, `mypy --strict src`,
`pytest -q -m "not sim"`, `python scripts/check_pins.py`.

A smoke experiment on the synthetic simulator:

```bash
carma run --config configs/experiment/smoke.yaml --seed 0
```

Results go to `runs/<run_id>/` with a manifest, a decision log and metrics.

## 3. Start the ROS 2 stack

From the repository root:

```bash
docker compose -f docker/compose.yaml up -d --build vision voice viewer
```

| Service | What it does |
|---|---|
| `vision` | Scene analysis (plant health) on the camera, stereo MQTT stream |
| `voice` | Offline speech-to-text for operator commands |
| `viewer` | Browser page at http://localhost:8088 |

The containers reach the host's broker as `host.docker.internal:1883`. On a
machine without a broker, add the bundled one:

```bash
MQTT_HOST=broker docker compose -f docker/compose.yaml --profile local-broker up -d
```

Stop everything:

```bash
docker compose -f docker/compose.yaml down
```

## 4. Camera: stream and live analysis

Docker Desktop cannot pass a USB camera into a container, so on a PC the webcam
is sent through the broker. Start `vision` with the MQTT camera source, then the
webcam script:

```bash
MQTT_CAMERA=true docker compose -f docker/compose.yaml up -d vision
python scripts/webcam_mqtt.py --camera 0
```

(PowerShell: `$env:MQTT_CAMERA="true"` first.) One webcam is copied into both
eyes; `--right-camera 1` sends a real stereo pair.

| Topic | Content |
|---|---|
| `carma/camera/frame` | Side-by-side JPEG, left eye on the left |
| `carma/camera/meta` | Layout, size, rate (retained) |
| `carma/analysis/<analyzer>` | JSON results, e.g. `plant_health` |

Watch results:

```bash
docker run --rm eclipse-mosquitto:2 mosquitto_sub -h host.docker.internal -t "carma/analysis/#" -v
```

**Browser viewer:** http://localhost:8088 needs a broker with a WebSocket
listener. Add to `mosquitto.conf` and restart the service:

```
listener 9001
protocol websockets
```

then open `http://localhost:8088/?port=9001`.

**Your own models:** export to ONNX and add an entry to
`configs/analyzers/field.yaml`, or write an analyzer class. See
`docs/analyzers.md`.

## 5. Operator speech on the PC

With `voice` running, in a terminal **in the repository folder**:

```bash
python scripts/mic_mqtt.py --list-devices
python scripts/mic_mqtt.py --device <index> --mode ptt
```

Press **Enter**, speak, press **Enter** again. The script prints what was heard:

```
  heard [es] "gira a la derecha" -> turn_right (1.4 s audio, 0.9 s to text)
```

| Intent | Say |
|---|---|
| `stop` | stop, alto, detente |
| `turn_left` / `turn_right` | turn left, gira a la derecha |
| `go` | go forward, sigue adelante |
| `yes` / `no` | yes, sí, no |
| `correction` | any longer sentence of advice, kept for memory |

`--mode vad` detects speech automatically but also picks up other voices in the
room. Languages are English and Spanish (`STT_LANGUAGES`); rejected clips show
`not understood`. Details: `docs/speech.md`.

Speech is **not** an emergency stop. The robot needs a physical e-stop.

## 6. Meta Quest 2 client

Full guide: `vr/unity/README.md`. In short:

1. Unity Hub → *Add project from disk* → `vr/unity/CarmaQuest` (Unity 6000.3.24f1).
   The first open downloads the Meta XR SDK and takes a few minutes.
2. In `SampleScene`, add an empty GameObject with the **Carma Voice** component;
   set **Broker Address** to the PC running Mosquitto (e.g. `192.168.0.153`).
3. *File → Build Profiles → Android → Build and Run* with the headset connected.
4. In the headset: hold **B**, speak, release. The status label shows the
   transcription and the intent.

The Quest and the broker must be on the same network, and Windows Firewall must
allow `mosquitto.exe` inbound.

## 7. Gazebo simulation

A row-crop field with the tractor UGV driving in it. Nothing here needs a
camera, a microphone or a headset — it is the cheapest way to see the robot
side of the topic contract working.

### Start it

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

Keep `--build`. The `sim` service runs the ROS workspace compiled into the
image, not `ros2_ws/src` on your disk, so edits to the world or the robot have
no effect until the image is rebuilt — and the service starts happily on the
old code, which makes that mistake easy to miss.

This is headless: there is no window, and the first start takes a few minutes
because it installs Gazebo Harmonic.

### Check it is alive

In a second terminal:

```bash
docker compose -f docker/compose.yaml run --rm shell
```

Inside that shell:

```bash
ros2 topic hz /carma/stereo/left/image_raw /carma/camera/depth /carma/odom
```

Expect roughly 10 Hz on the cameras and 20 Hz on odometry on a PC without a
GPU; nominal is 15 and 30, and the shortfall is software rendering, not a
fault.

### Drive it

```bash
ros2 topic pub -r 20 /carma/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.6}, angular: {z: 0.0}}'
```

| `linear.x` | `angular.z` | Result |
|---|---|---|
| `0.6` | `0.0` | Straight down the alley |
| `0.6` | `0.4` | Forward, curving left |
| `0.6` | `-0.4` | Forward, curving right |
| `-0.4` | `0.0` | Reverse |
| `0.0` | `0.5` | **Nothing.** No steering without forward motion |

The vehicle is a tractor: rear-wheel drive, Ackermann front steering. It cannot
turn on the spot, and its minimum turning radius of 0.83 m is wider than the
alley, so it turns at the headland. Watch the pose respond:

```bash
ros2 topic echo /carma/odom --field pose.pose.position
```

There is no episode reset yet, so if you drive into a crop row the only way
back to the start line is to restart the service.

### Change the crop stage

The drift study runs the same routes at two phenological stages. The stage is
one recorded world argument, not a second copy of the world file:

```bash
docker compose -f docker/compose.yaml --profile sim run --rm sim \
  ros2 launch carma_sim row_crop.launch.py stage:=CANOPY_CLOSURE
```

Valid names are those of `carma.types.PhenologyStage`. A typo fails the launch
rather than quietly rendering the default stage.

### See it, with a window

The GUI needs a display, which the container does not have by default. On
Windows 11 the WSLg X server can be passed through — note this is a plain
`docker run`, not compose, and it renders in software:

```bash
docker run -d --name carma_gui --ipc host --network carma_default \
  -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 -e ROS_DOMAIN_ID=42 \
  -v /run/desktop/mnt/host/wslg:/mnt/wslg \
  -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix \
  carma-ros-sim:jazzy ros2 launch carma_sim row_crop.launch.py headless:=false
```

On Linux, `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` is enough.

In the Gazebo window, the **⋮** menu at the top right adds panels:

- **Image Display** → topic `/carma/stereo/left/image_rgb` shows the camera.
  Gazebo lists its own topic names, so it is `image_rgb` here; `image_raw` is
  the ROS-side name the `bgr_relay` node publishes after converting to `bgr8`.
- **Teleop** → topic `/carma/cmd_vel` drives it with the arrow keys. Hold
  forward together with a turn; turning alone does nothing.

Close it with `docker rm -f carma_gui`.

### Stop it

```bash
docker compose -f docker/compose.yaml --profile sim down
```

## 8. Quick troubleshooting

| Symptom | Fix |
|---|---|
| Viewer page shows nothing | Broker needs the WebSocket listener (section 4); camera script must be running |
| `vision` logs `0 frames` | Webcam script not running, or sending to another broker/port |
| Speech script prints nothing | Run it in your own terminal, not in the background; check `docker compose -f docker/compose.yaml logs voice` |
| Wrong language or nonsense text | Use `--mode ptt`; set `STT_LANGUAGE=en` or `es` |
| Low camera frame rate in ROS | The image ships `docker/fastdds_shm.xml`; keep `ipc: host` on ROS services |
| Quest: `broker connection failed` | Same Wi-Fi as the broker, correct address, firewall rule |
| Sim: edits to the world do nothing | Only `shell` mounts `ros2_ws/src`; rebuild with `docker compose -f docker/compose.yaml build sim` |
| Sim: `angular.z` does not turn the robot | It is Ackermann-steered; `linear.x` must be non-zero |
| Sim: robot stuck in the crop | No episode reset yet; restart the `sim` service |
| Sim: no Gazebo window | Headless by default; see section 7 for the WSLg/X11 route |

## Where to read more

| Topic | File |
|---|---|
| Architecture and module rules | `docs/architecture.md`, `AGENTS.md` |
| Simulation design and adapters | `docs/simulation.md` |
| World, robot geometry, sensor intrinsics | `ros2_ws/src/carma_sim/README.md` |
| Cost model (act / retrieve / ask) | `docs/cost-model.md` |
| Running experiments | `docs/experiments.md` |
| Data formats | `docs/data-schema.md` |
| Containers and DDS | `docker/README.md` |
| Streaming and VR | `docs/streaming.md` |
| Speech | `docs/speech.md` |
| Custom perception models | `docs/analyzers.md` |
| Quest client | `vr/unity/README.md` |
