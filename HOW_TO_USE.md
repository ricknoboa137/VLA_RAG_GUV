# How to use CARMA

This guide covers how to run every part of the system, alone and together, and
what every file in the repository does and why it is needed. For what the
project is, see `README.md`. For the rules code must follow, see `AGENTS.md`.

If you have no hardware to hand, start with the simulation (section 8). It
needs only Docker.

---

## 1. The system at a glance
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/8ff7ea09-110d-4c05-ad46-bc4cc4ee5709" />



```
                ┌──────────────────────── MQTT broker (Mosquitto, port 1883) ─────────────────────────┐
                │                                                                                     │
 webcam / 3D ──► scripts/webcam_mqtt.py ──► carma/source/frame ──► vision container ──► carma/camera/frame ──► browser viewer
 USB camera                                                        (carma_vision)      carma/analysis/*         (port 8088)
                │
                └──► scripts/webcam_mqtt.py --topic MqttVidFeed ───────────────────────────────────► Quest 2 headset
                                                                                                     (CarmaQuest app)
 PC microphone ──► scripts/mic_mqtt.py ──┐                                                                 │
                                         ├──► carma/operator/audio ──► voice container ──► carma/operator/utterance
 Quest mic (hold B) ─────────────────────┘                             (carma_voice)          (text + intent)

 Gazebo row-crop world (sim container) ──► ROS 2 /carma/stereo/*, /carma/camera/depth, /carma/odom ◄── /carma/cmd_vel

 carma run (Python library) ──► episodes on the synthetic simulator ──► runs/<run_id>/  (manifest, decisions, metrics)
        └── optional: backbone served by scripts/policy_server.py over HTTP
```

### What is connected today, and what is not yet

Each piece below runs and was verified on its own. Some links between pieces
are still to be built, and you need to know which before you try to run
everything at once.

| Piece | Status |
|---|---|
| Camera → vision container → analysis results and browser viewer | Works |
| PC or Quest microphone → voice container → text and intent back to the sender | Works |
| Camera → Quest headset (mono or stereo) | Works, directly from `webcam_mqtt.py`, or through the vision container with `VR_TOPIC=MqttVidFeed` so the same camera also feeds the viewer and the VLA |
| World and arm stereo cameras + speech → VLA policy server (`carma_vla`) | Works with the stand-in policy (Option D, section 4) |
| Quest controller and head pose → `test1` topic | Published. Nothing in this repository reads it yet; the robot side that does is not here |
| Gazebo simulation: cameras, depth, odometry, driving with `/carma/cmd_vel` | Works |
| Experiments (`carma run`) on the synthetic simulator | Works, and writes reproducible manifests |
| Experiments on Gazebo | **Not yet.** `carma_bridge` and `src/carma/sim/gazebo.py` are skeletons; use `sim.kind: synthetic` |
| A real VLA model behind the policy server, driving the arm | Works in Gazebo: SmolVLA or X-VLA moves the SO-101 from the arm camera and a spoken instruction (Option E, section 4). Community checkpoints: no task skill is claimed |
| A VLA model driving the wheeled base | Not yet: no released checkpoint predicts navigation actions (`docs/vla-integration.md`) |

**One camera, one reader.** A USB camera can be opened by only one program at
a time. With a single camera you therefore choose either the **analysis +
browser viewer** path or the **headset** path (section 4). With two cameras you
can run both, one `webcam_mqtt.py` per camera.

---

## 2. Requirements

| Needed for | Install |
|---|---|
| Everything | Git, and this repository cloned |
| Containers (vision, voice, viewer, simulation) | Docker Desktop on Windows (WSL 2 backend), or Docker Engine with the compose plugin on Linux |
| Host scripts and experiments | Python 3.11 or newer, for example a conda environment |
| MQTT | A Mosquitto broker on port 1883 on the PC. If there is none, use the bundled one (section 3) |
| Quest client | Unity 6000.3.24f1 with Android Build Support, and a Meta Quest 2 in developer mode |

---

## 3. One-time setup

From the repository root:

```bash
pip install -e ".[dev,stream,voice]"
```

The extras: `dev` adds the check tools (ruff, mypy, pytest); `stream` adds
`paho-mqtt` for the camera and microphone scripts; `voice` adds faster-whisper
and `sounddevice`.

Download the speech model once. It goes into `assets/`, which is never
committed:

```bash
python scripts/fetch_models.py faster-whisper-base
```

Build the container images. The first build takes several minutes:

```bash
docker compose -f docker/compose.yaml build vision voice
```

Check that the library is healthy:

```bash
make check
```

Without `make`, run the four gates by hand: `ruff check src tests scripts`,
`mypy --strict src`, `pytest -q -m "not sim"` and `python scripts/check_pins.py`.

**No broker on this PC?** Start the bundled one. It listens on host ports
**1884** (MQTT) and **9002** (WebSocket), so it never collides with a host
broker:

```bash
MQTT_HOST=broker docker compose -f docker/compose.yaml --profile local-broker up -d broker
```

When you use it, put `MQTT_HOST=broker` in front of every later
`docker compose ... up` as well, so the containers connect to it, and pass
`--port 1884` to the host scripts.

**Browser viewer.** The page on port 8088 talks to the broker over WebSocket.
The bundled broker already has that listener. A host Mosquitto needs these two
lines in its `mosquitto.conf`, then a restart of the service:

```
listener 9001
protocol websockets
```

On Windows PowerShell, set an environment variable with `$env:NAME="value"` on
its own line before the command, instead of writing `NAME=value` in front of it.

---

## 4. Run the whole live system together

Open one terminal per step, all in the repository folder. Stop host scripts
with **Ctrl+C**, and the containers with the command in section 10.

### Option A: camera analysis + browser viewer + speech

**Terminal 1. Start the containers.** `MQTT_CAMERA=true` makes the vision
container take its camera from the broker, because Docker Desktop cannot pass a
USB camera into a container.

```bash
MQTT_CAMERA=true docker compose -f docker/compose.yaml up -d vision voice viewer
```

**Terminal 2. Send the camera.** `--camera 0` is the first webcam. For the 3D
USB stereo camera use its index and its side-by-side mode, for example
`--camera 1 --width 1280 --height 480`.

```bash
python scripts/webcam_mqtt.py --camera 0
```

**Terminal 3. Talk to it.** List the microphones, then start push-to-talk with
the one you want:

```bash
python scripts/mic_mqtt.py --list-devices
python scripts/mic_mqtt.py --device <index> --mode ptt
```

Press **Enter**, speak, press **Enter** again. The script prints what was
heard and the intent it maps to:

```
  heard [es] "gira a la derecha" -> turn_right (1.4 s audio, 0.9 s to text)
```

**Look at it.** Open http://localhost:8088/?port=9001 (or `?port=9002` with the
bundled broker) for the live picture and the analysis results. To see the raw
messages instead:

```bash
docker run --rm eclipse-mosquitto:2 mosquitto_sub -h host.docker.internal -t "carma/#" -v
```

### Option B: headset + speech

**Terminal 1. Start the speech container:**

```bash
docker compose -f docker/compose.yaml up -d voice
```

**Terminal 2. Send the camera to the headset.** Replace `<pc-ip>` with this
PC's address on the Wi-Fi the headset uses. The headset reads `MqttVidFeed` as
base64 text, which is why both overrides are needed:

```bash
python scripts/webcam_mqtt.py --camera 1 --width 1280 --height 480 --host <pc-ip> --topic MqttVidFeed --encoding base64
```

**In the headset.** Start the CarmaQuest app (section 7). The camera appears in
front of you, and a side-by-side frame is shown in 3D automatically. Hold
**B**, speak, release; the status panel shows the text and the intent.

The headset and the broker must be on the same network, and Windows Firewall
must allow `mosquitto.exe` inbound.

### Option C: everything, with two cameras

Run Option A's terminals 1 and 3. Then run two camera senders, one per camera:
`python scripts/webcam_mqtt.py --camera 0` for the vision container, and the
Option B command with `--camera 1` for the headset.

### Option D: two stereo cameras and speech into the VLA

The robot carries two stereo cameras. The **world camera** gives the platform's
view and goes to the headset; the **arm camera** — the 3D USB camera, mounted
on the SO-101 arm — finds and approaches targets. Each camera has exactly one
sender, and the vision container fans the world camera out to the headset, the
viewer and the VLA.

**Terminal 1. The policy server**, reachable from the containers:

```bash
python scripts/policy_server.py --policy echo --host 0.0.0.0
```

**Terminal 2. The containers.** `ARM_CAMERA=true` adds the arm camera,
`VR_TOPIC=MqttVidFeed` sends the world camera to the headset:

```bash
MQTT_CAMERA=true ARM_CAMERA=true VR_TOPIC=MqttVidFeed docker compose -f docker/compose.yaml up -d vision voice viewer
docker compose -f docker/compose.yaml --profile vla up -d vla
```

**Terminals 3 and 4. One sender per camera.** The arm camera is the 3D USB
camera in side-by-side mode; the world camera keeps the default source:

```bash
python scripts/webcam_mqtt.py --camera 1 --width 1280 --height 480 --source arm_source
python scripts/webcam_mqtt.py --camera 0 --width 1280 --height 480
```

With only the arm camera connected, skip the world sender: the pipeline runs on
the arm camera alone.

**Terminal 5. Tell it what to do**, with `mic_mqtt.py` as in Option A or the
headset's **B** button: say a sentence such as "pick up the red bottle by the
second row". "stop" pauses it, "go" resumes. Watch the answers:

```bash
docker compose -f docker/compose.yaml run --rm shell ros2 topic echo /carma/vla/action
```

It does not drive unless started with `VLA_DRIVE=true`. Details, and what every
spoken command does: `docs/vla-integration.md`.

### Option E: drive the SO-101 arm by voice with SmolVLA or X-VLA

A real VLA behind the policy server moves the arm from the arm camera, the
world camera and the operator's instruction. Checked in Gazebo with both
models. Needs the `carma-train` environment (`training/README.md`) and the
checkpoints, fetched once (3 GB):

```powershell
$py = "$env:USERPROFILE\anaconda3\envs\carma-train\python.exe"
& $py scripts/fetch_training_assets.py svla_so101_red_cube
& $py scripts/fetch_training_assets.py xvla_so101_red_cube
& $py scripts/fetch_training_assets.py smolvlm2-500m-video-instruct
& $py scripts/fetch_training_assets.py bart-large-tokenizer
```

**Terminal 1. The model**, on the host, from the repository root. `--policy xvla`
serves the other one; nothing else changes.

```powershell
$env:PYTHONPATH = "src"
& $py scripts/policy_server.py --policy smolvla --self-check --host 0.0.0.0
```

**Terminal 2. The simulated robot with its arm**, as in section 8:

```bash
docker compose -f docker/compose.yaml --profile sim run --rm sim ros2 launch carma_sim row_crop.launch.py arm:=true
```

**Terminal 3. The pipeline, with the arm switched on**, plus the speech service
from Option A if it is not running:

```bash
VLA_ARM=true docker compose -f docker/compose.yaml --profile vla up vla voice
```

**Then say** "pick up the red cube" (`mic_mqtt.py` or the headset's **B**
button). The arm moves; "stop" holds it where it is, "go" resumes. Each answer
is on `/carma/vla/action`, naming the model, its revision and its uncertainty.

The two checkpoints were fine-tuned by a third party on a red-cube task, and
the simulated world has no red cube: this shows the whole path working, not
the models' skill. How the two are compared, and what a fair benchmark still
needs: `docs/arm-policies.md`.

---

## 5. Experiments (the research part)

These run in plain Python, with no containers and no hardware. Each run writes
`runs/<run_id>/` holding `manifest.yaml` (what was run, from which commit, on
which machine), `decisions.jsonl` (one line per decision, with the three prices
that produced it) and `metrics.yaml`.

A quick check that the install works. It takes seconds:

```bash
carma run --config configs/experiment/smoke.yaml --seed 0
```

A success rate of zero here is expected: the placeholder backbone is never told
where the goal is (see `README.md`, *Status*).

The simulation study over several seeds, then the summary:

```bash
carma run --config configs/experiment/sim_study.yaml --seed 0
carma run --config configs/experiment/sim_study.yaml --seed 1
carma report runs/
```

`carma report` refuses to combine runs whose configs differ, so mixing two
conditions in one folder fails loudly rather than averaging them. Give each
condition its own folder with `--out`. To compare the five conditions, see
`docs/experiments.md`. `carma components` lists every name a config file may
use.

**With the backbone in another process.** This is the path a real VLA model
will take. Terminal 1:

```bash
python scripts/policy_server.py --policy echo --self-check
```

Terminal 2:

```bash
carma run --config configs/experiment/remote_smoke.yaml --seed 0
```

The same checks run inside a container for a colleague who has only Docker:
`make docker-build`, `make docker-check` and `make docker-smoke`. Details are
in `docker/README.md`.

---

## 6. Operator speech in more detail

| Intent | Say |
|---|---|
| `stop` | stop, alto, detente |
| `turn_left` / `turn_right` | turn left, gira a la derecha |
| `go` | go forward, sigue adelante |
| `yes` / `no` | yes, sí, no |
| `correction` | any longer sentence of advice, kept for memory |

- `--mode ptt` (press Enter) is the reliable mode. `--mode vad` starts and
  stops on its own, but it also picks up other voices in the room.
- Only English and Spanish are accepted. Set `STT_LANGUAGE=en` or `es` before
  starting `voice` to force one. A rejected clip prints `not understood`.
- Speech is **not** an emergency stop. The robot needs a physical e-stop.

Full detail: `docs/speech.md`.

---

## 7. Meta Quest 2 client

Full guide: `vr/unity/README.md`. In short:

1. Unity Hub → *Add project from disk* → `vr/unity/CarmaQuest` (Unity
   6000.3.24f1). The first open downloads the Meta XR SDK and takes several
   minutes.
2. The broker address is set in `SampleScene` on two objects: `mqttReceiver`
   (video) and `CarmaVoice` (speech). Both are currently `192.168.0.153`.
   Change both to the address of the PC running the broker.
3. *File → Build Profiles → Android → Build and Run*, with the headset
   connected by USB.
4. In the headset, hold **B**, speak, release.

For a black screen, run `adb logcat -s Unity | grep mqttReceiver`. Frames
counting up means the video arrives and the problem is rendering.

---

## 8. Gazebo simulation

A row-crop field with the tractor-style UGV driving in it. It needs no camera,
microphone or headset.

**Start it.** Keep `--build`: the service runs the code compiled into the
image, not the files on your disk, so without a rebuild an edit to the world or
the robot does nothing, and nothing warns you.

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

It runs headless, with no window. The first start installs Gazebo Harmonic and
takes a few minutes.

**Check that it is alive**, from a second terminal:

```bash
docker compose -f docker/compose.yaml run --rm shell
```

then, inside that shell:

```bash
ros2 topic hz /carma/stereo/left/image_raw /carma/camera/depth /carma/odom
```

Without a GPU, expect about 10 Hz on the cameras and 20 Hz on odometry, against
nominal rates of 15 and 30. The shortfall comes from software rendering and is
not a fault.

**Drive it:**

```bash
ros2 topic pub -r 20 /carma/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.6}, angular: {z: 0.0}}'
```

| `linear.x` | `angular.z` | Result |
|---|---|---|
| `0.6` | `0.0` | Straight down the alley |
| `0.6` | `0.4` | Forward, curving left |
| `0.6` | `-0.4` | Forward, curving right |
| `-0.4` | `0.0` | Reverse |
| `0.0` | `0.5` | **Nothing.** It cannot steer without moving forward |

The vehicle has rear-wheel drive and steers its front wheels like a car
(Ackermann steering), so it cannot turn on the spot. Its minimum turning
radius, 0.83 m, is wider than the alley, so it turns at the headland. There is
no episode reset yet: if it drives into a crop row, restart the service.

**Change the crop stage.** The drift study runs the same routes at two growth
stages:

```bash
docker compose -f docker/compose.yaml --profile sim run --rm sim ros2 launch carma_sim row_crop.launch.py stage:=CANOPY_CLOSURE
```

Valid names are those of `carma.types.PhenologyStage`. A typo stops the launch
instead of silently rendering the default stage.

**Add the SO-101 arm.** Off by default, so recorded worlds are unchanged:

```bash
docker compose -f docker/compose.yaml --profile sim run --rm sim ros2 launch carma_sim row_crop.launch.py arm:=true
```

Then move a joint from the shell, angles in radians:

```bash
ros2 topic pub --once /carma/arm/shoulder_lift/cmd_pos std_msgs/msg/Float64 '{data: -0.8}'
```

Joints, topics, the arm's stereo camera and why the arm is on the front:
`docs/arm.md`.

**See it in a window.** On Windows 11 this goes through the WSLg X server. It
is a plain `docker run`, not compose, and it renders in software:

```bash
docker run -d --name carma_gui --ipc host --network carma_default -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 -e ROS_DOMAIN_ID=42 -v /run/desktop/mnt/host/wslg:/mnt/wslg -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix carma-ros-sim:jazzy ros2 launch carma_sim row_crop.launch.py headless:=false
```

On Linux, `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` is enough. In
the Gazebo window, the **⋮** menu adds an **Image Display** panel (topic
`/carma/stereo/left/image_rgb`) and a **Teleop** panel (topic `/carma/cmd_vel`;
hold forward together with a turn). Close it with `docker rm -f carma_gui`.

World, robot geometry and sensor details: `ros2_ws/src/carma_sim/README.md`.

---

## 9. Stereo camera measurements

These tools measured the 3D USB camera. The numbers they produced are recorded
in `experiments/stereo_depth_validation/` and reported in `docs/stereo_depth.md`.

| Command | What it does |
|---|---|
| `python scripts/stereo_depth.py --camera 1 --width 1280 --height 480` | Live depth preview and a `.ply` point cloud from the camera |
| `python scripts/calibrate_stereo.py sample --distance 0.60`, repeated at several distances, then `python scripts/calibrate_stereo.py fit --out config/stereo_3dusb.json` | Calibrates the depth scale from a box at tape-measured distances |
| `python scripts/test_rolling_shutter.py static`, then `moving`, then `report` | Tests whether the camera has a rolling or a global shutter |
| `python scripts/report_stereo_validation.py` | Recomputes every published depth figure from the recorded data, without a camera |

---

## 10. Stop everything

```bash
docker compose -f docker/compose.yaml --profile sim --profile local-broker down
```

This stops and removes every CARMA container. Host scripts stop with Ctrl+C.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Viewer page shows nothing | The broker needs its WebSocket listener (section 3), `?port=` must match it, and `webcam_mqtt.py` must be running |
| `vision` logs `0 frames` | `webcam_mqtt.py` is not running, or it sends to another broker or port |
| The camera does not open | Another program (Teams, a browser, a second `webcam_mqtt.py`) is holding it |
| Speech script prints nothing | Run it in your own terminal, not in the background; check `docker compose -f docker/compose.yaml logs voice` |
| Wrong language or nonsense text | Use `--mode ptt`; set `STT_LANGUAGE=en` or `es` |
| Low camera frame rate in ROS | Keep `ipc: host` on the ROS services; the image ships `docker/fastdds_shm.xml` for large frames |
| Quest: `broker connection failed` | Same Wi-Fi as the broker, correct address on both scene objects, firewall rule for `mosquitto.exe` |
| Quest: black screen | `adb logcat -s Unity`, look for `mqttReceiver`; frames counting up means rendering, not the feed |
| Sim: edits to the world do nothing | Rebuild with `docker compose -f docker/compose.yaml build sim` |
| Sim: `angular.z` does not turn the robot | It steers like a car; `linear.x` must be non-zero |
| `carma report` refuses to aggregate | The runs in that folder used different configs; give each condition its own `--out` folder |
| `warning: working tree is dirty` | Commit first; a run from uncommitted code cannot be reproduced |

---

## 12. What every file does, and why it is needed

Generated folders are never committed: `assets/` (downloaded models), `runs/`
(experiment output), `build/`, `install/` and `log/` (ROS builds), and Unity's
`Library/`, `Logs/` and `UserSettings/`.

### Repository root

| File | What it does | Why it is needed |
|---|---|---|
| `README.md` | What CARMA is, its status, the five conditions, the layout | The entry point; also the package description in `pyproject.toml` |
| `HOW_TO_USE.md` | This guide | How to run the system and find your way around it |
| `AGENTS.md` | The development contract: module boundaries, reproducibility gates, conventions, definition of done | Every change is checked against it; CI enforces most of it |
| `CONTRIBUTING.md` | Branching, review order, how to report a result bug | A short workflow for collaborators, pointing at `AGENTS.md` |
| `CITATION.cff` | How to cite the software | Lets GitHub and reference managers produce a correct citation for the paper |
| `LICENSE` | Apache-2.0 licence text | Required for the code to be reused and redistributed (`AGENTS.md`, rule 1) |
| `pyproject.toml` | Package definition: pinned dependencies, optional extras, the `carma` command, ruff/mypy/pytest settings | Installs the library, and is the one place the lint, type and test settings live |
| `Makefile` | `make check` (all gates), `smoke`, `format`, `lock`, and the Docker targets | One definition of "passing", shared by people, CI and the container |
| `.gitignore` | Keeps models, runs, build output, caches and keystores out of git | Stops large or secret files being committed |
| `.gitattributes` | Forces LF line endings on shell scripts, Python, YAML and the Dockerfile | A Windows CRLF line ending breaks scripts inside Linux containers |
| `.dockerignore` | Excludes the Unity project, caches and runs from the Docker build | Cuts the build context from 9.9 GB to about 3 MB |
| `.pre-commit-config.yaml` | Optional hooks: ruff, whitespace, YAML check, large-file and private-key detection | Catches problems before they are committed |
| `.github/workflows/ci.yml` | GitHub Actions: lint, types, tests, pin gate, and the container build | Enforces the gates on every push and pull request |

### `src/carma/`: the Python library

Plain Python with no ROS dependency, so its tests run anywhere. Imports may
only travel in the directions shown in `AGENTS.md` section 3;
`tests/test_architecture.py` fails the build if one does not.

| File | What it does | Why it is needed |
|---|---|---|
| `__init__.py` | Package marker and version | Makes `carma` importable and gives the manifest its version |
| `registry.py` | A generic name → class registry | Lets config files name components (`heuristic`, `cost_aware`) instead of code choosing them |
| `config.py` | Loads and validates an experiment YAML, computes its hash, and builds every component from it (`build`, `build_analyzers`) | The only place objects are constructed, so a condition is a config change, not a code change |
| `types/core.py` | Shared value types: `Observation`, `Pose`, `Action`, `Decision`, `MemoryEntry`, `StepRecord`, `EpisodeSpec`, `PhenologyStage`, operator intents and more | Every package exchanges these, in SI units |
| `types/protocols.py` | The interfaces every swappable part must satisfy: `NavigationBackbone`, `Embedder`, `Store`, `Retriever`, `HygienePolicy`, `CostModel`, `Arbiter`, `OperatorInterface`, `SceneAnalyzer`, `SimAdapter` | New implementations plug in without touching the rest |
| `types/errors.py` | `CarmaError` and its subclasses | Failures are raised as typed errors, never returned as `None` |
| `types/__init__.py` | Re-exports the above | Short import paths |
| `arbiter/cost_model.py` | Prices ACT, RETRIEVE and ASK in one currency: expected extra traverse time, in seconds | **The scientific core.** Makes operator attention comparable with retrieval time and with the risk of acting wrongly |
| `arbiter/policies.py` | The four arbiters behind the five conditions: `always_act`, `always_retrieve`, `uncertainty_ask`, `cost_aware` | What makes one experimental condition differ from another |
| `arbiter/__init__.py` | Package docstring | The only package allowed to see both memory and backbone |
| `backbone/heuristic.py` | A model-free stand-in policy built from OpenCV cues | Lets the loop, the metrics and the manifests run before a real VLA model exists |
| `backbone/remote.py` | Calls a policy served by another process over HTTP | A large model runs in its own process, or on a GPU machine, while a condition stays one line of config |
| `backbone/wire.py` | The request and reply format shared by `remote.py` and `scripts/policy_server.py` | Both ends import it, so the contract cannot drift |
| `backbone/registry.py` | Backbone registry | Makes `heuristic` and `remote` nameable in config |
| `backbone/__init__.py` | Package docstring | States the rule that backbone never imports memory |
| `memory/retriever.py` | Turns an observation into retrieved operator corrections, and corrections into memory entries | The RETRIEVE option |
| `memory/hygiene.py` | Scores how stale a memory entry is (crop stage, age, distance) and picks entries for the operator to review | Field memory decays as the crop grows; the `carma_no_hygiene` condition isolates this |
| `memory/embedders/hashing.py` | A dependency-free text embedder | Tests and CI run with no model download |
| `memory/embedders/sentence.py` | Adapter for sentence-transformer models stored in `assets/` | The real embedder for experiments (optional `retrieval` extra) |
| `memory/embedders/registry.py` | Embedder registry | Makes embedders nameable in config |
| `memory/stores/in_memory.py` | Exact cosine-similarity store that keeps insertion order for ties | Deterministic retrieval, which reproducibility requires |
| `memory/stores/registry.py` | Store registry | Makes stores nameable in config |
| `memory/__init__.py`, `memory/embedders/__init__.py`, `memory/stores/__init__.py` | Package markers | State the rule that memory never imports backbone |
| `operator/scripted.py` | A simulated operator that answers from a fixed script with sampled response times | Makes the simulation study reproducible |
| `operator/speech.py` | Maps transcribed words to an intent (stop, go, turn, yes, no, correction) with a small English and Spanish keyword grammar | Used by the voice container; predictable and testable by hand |
| `operator/registry.py` | Operator registry | Makes operators nameable in config |
| `operator/__init__.py` | Package docstring | States that operator latencies are measurements, not telemetry |
| `perception/features.py` | Cheap deterministic image cues: green fraction, texture energy, traversability, a scene descriptor | Used by the heuristic backbone and by hygiene, and as the visual descriptor of memory entries |
| `perception/stereo_depth.py` | Splits a side-by-side frame and computes disparity, metric depth and point clouds | Depth from the 3D USB camera; used by the stereo scripts |
| `perception/analyzers/plant_health.py` | A colour-based plant-health cue | Gives the live pipeline a real result to publish, and is the smallest example of an analyzer |
| `perception/analyzers/onnx_classifier.py` | Runs any image classifier exported to ONNX | How your own trained models join without new code |
| `perception/registry.py`, `perception/analyzers/__init__.py`, `perception/__init__.py` | Analyzer registry and package markers | Importing the package registers the shipped analyzers |
| `runtime/episode.py` | The control loop: observe, propose, price, choose, record, act | One loop, identical for all five conditions |
| `runtime/episodes.py` | Generates the synthetic episodes (start, goal, crop stage) | Gives the harness something to run in CI |
| `runtime/experiment.py` | Runs every episode of an experiment and collects the metrics | Called by `carma run` |
| `runtime/__init__.py` | Package marker | |
| `sim/synthetic.py` | A tiny dependency-free simulated field | The whole loop runs in under a second with no simulator installed |
| `sim/gazebo.py` | Adapter to Gazebo through `carma_bridge` (skeleton) | Where simulated experiments in Gazebo will connect |
| `sim/registry.py`, `sim/__init__.py` | Simulator registry and package marker | Makes `synthetic` and `gazebo` nameable in config |
| `metrics/task.py` | Success rate, navigation error, SPL | Defined like the published agricultural navigation baselines, so the numbers sit beside them |
| `metrics/interaction.py` | Queries per 100 m, interventions per hectare, neglect time, total operator latency | **The project's distinctive measurement:** what each choice costs the operator |
| `metrics/retrieval.py` | Whether retrieval found the right thing | Separates "retrieved well" from "acted well anyway" |
| `metrics/systems.py` | Decision-latency summary | Shows the loop is fast enough to run on the robot |
| `metrics/arm.py` | Open-loop joint error, uncertainty-error rank correlation, latency, for arm policies | Compares SmolVLA with X-VLA; outside `MetricBundle`, because the arm is outside the experiments |
| `metrics/bundle.py` | `MetricBundle`, one struct holding every reported number | Every paper number comes from one place (`AGENTS.md`, rule 5) |
| `metrics/__init__.py` | Package docstring | |
| `cli/main.py` | The `carma` command: `run`, `report`, `components` | The entry point for every experiment |
| `cli/manifest.py` | Writes and reads `manifest.yaml`: config hash, commit, dirty flag, seed, platform, dependency versions | A run without a manifest is not a result |
| `cli/__init__.py` | Package marker | |

### `tests/`: the unit suite

All tests run offline, with no GPU and no simulator, in well under a minute.

| File | What it checks |
|---|---|
| `conftest.py` | Shared fixtures (small configs, observations) used by the other tests |
| `test_architecture.py` | No import crosses a forbidden module boundary |
| `test_determinism.py` | The same config and seed give byte-identical step records |
| `test_manifest.py` | Every manifest has every required field |
| `test_config.py` | Configs load, validate and build the right components |
| `test_arbiter.py` | The cost model and the arbiters choose as designed |
| `test_memory.py` | Store, embedders, retriever and hygiene |
| `test_metrics.py` | Every metric against a value computed by hand |
| `test_runtime.py` | A full episode loop on the synthetic simulator |
| `test_remote_backbone.py` | The remote backbone and its wire format, arm fields included, without opening a socket |
| `test_arm_policy.py` | The arm policy adapter's units, spread, uncertainty and camera mapping, and that the two compared specs differ only by the model |
| `test_metrics_arm.py` | The arm-policy metrics against hand-computed values |
| `test_perception.py` | Image cues are bounded and deterministic |
| `test_analyzers.py` | Analyzers are bounded, deterministic and buildable from config |
| `test_speech.py` | Every speech intent, stated by hand |
| `test_stereo_depth.py` | Splitting, disparity, depth and point clouds |
| `test_stereo_validation.py` | The published depth figures still follow from the recorded measurements |
| `test_types.py` | Value-type invariants |

### `configs/`: experiments as files

| File | What it does | Why it is needed |
|---|---|---|
| `README.md` | Explains the folder | |
| `condition/no_memory.yaml` | Backbone alone (`always_act`) | The floor of the comparison |
| `condition/unconditional_retrieval.yaml` | Retrieve on every step (`always_retrieve`) | The published RAG-VLA pattern |
| `condition/ask_only.yaml` | Ask when uncertain, no memory (`uncertainty_ask`) | The ask-for-help pattern |
| `condition/carma.yaml` | Full cost-aware arbitration | The proposed method |
| `condition/carma_no_hygiene.yaml` | Cost-aware, without staleness handling | Isolates what hygiene contributes |
| `experiment/smoke.yaml` | Three short episodes, no downloads | Install check, used by CI and the container |
| `experiment/sim_study.yaml` | The 100-episode simulation study | The main experiment; copy once per condition |
| `experiment/remote_smoke.yaml` | The smoke run with the backbone served over HTTP | Proves the client–server path before a real model is used |
| `analyzers/field.yaml` | Which scene analyzers the vision container runs | Kept apart from experiment configs, so editing it never changes a result's hash |
| `arm_policy/svla_so101_red_cube.yaml`, `xvla_so101_red_cube.yaml` | The two arm policies: checkpoint, revision, camera slots, joint units, uncertainty map | A model is swapped by naming another spec; not experiment configs, never in a hash |

### `config/` and `experiments/`: camera calibration and evidence

| File | What it does | Why it is needed |
|---|---|---|
| `config/stereo_3dusb.json` | Depth calibration of the 3D USB camera (focal length, baseline, disparity offset) | Read by `scripts/stereo_depth.py` for metric distances |
| `config/stereo_3dusb_samples.json` | The tape-measured samples the calibration was fitted from | Lets the calibration be re-fitted and checked |
| `experiments/stereo_depth_validation/README.md` | How the depth measurements were taken | Documents the evidence |
| `experiments/stereo_depth_validation/measurements.json` | The recorded depth measurements | Source of every number in `docs/stereo_depth.md`; pinned by a test |
| `experiments/stereo_depth_validation/shutter.json` | The recorded rolling-shutter test | Evidence for the "effectively global shutter" finding |

### `scripts/`: host-side tools

| File | What it does | Why it is needed |
|---|---|---|
| `fetch_models.py` | Downloads open model weights into `assets/` | The only step that touches the network, so downloads are explicit and auditable |
| `fetch_training_assets.py` | Downloads the SmolVLA and X-VLA checkpoints, their VLM config and tokenizer, and the SO-101 datasets into `assets/`, each pinned to a commit, and points them at local files | The same explicit fetch for the separate training environment in `training/` |
| `check_pins.py` | Fails if a dependency is not pinned to an exact version | Reproducibility gate, run by `make check` and CI |
| `webcam_mqtt.py` | Sends a webcam (mono, or a side-by-side stereo pair) to the broker | Feeds the vision container and the headset from a PC camera |
| `mic_mqtt.py` | Records speech from a PC microphone and sends each clip to the broker | Operator speech without the headset |
| `policy_server.py` | Serves a policy over HTTP: `echo` for navigation, `smolvla` or `xvla` for the arm | The other end of `backbone/remote.py` and of `carma_vla` |
| `arm_policy.py` | Loads SmolVLA or X-VLA through LeRobot from a spec in `configs/arm_policy/` and answers with arm chunks | The model behind `--policy smolvla` / `xvla`; torch-free parts are unit-tested |
| `benchmark_arm_policies.py` | Feeds recorded SO-101 episodes to every arm policy server and scores their chunks | The SmolVLA-against-X-VLA comparison |
| `stereo_depth.py` | Live depth preview and point cloud from the stereo camera | Shows the depth the robot will see |
| `calibrate_stereo.py` | Fits the depth calibration from tape-measured distances | Produces `config/stereo_3dusb.json` |
| `test_rolling_shutter.py` | Measures whether the camera's shutter is rolling or global | Produced `shutter.json`; matters for SLAM on a moving robot |
| `report_stereo_validation.py` | Recomputes the published depth figures from the recorded data | Anyone can verify the paper's numbers without the camera |

### `docker/`: containers

| File | What it does | Why it is needed |
|---|---|---|
| `Dockerfile` | Four images: `base` (ROS 2 Jazzy, the library, the ROS packages), `sim` (`base` plus Gazebo), `lib` (the library only) and `dev` (`lib` plus the check tools) | The same environment on any PC or robot |
| `compose.yaml` | The services: `vision`, `voice`, `viewer`, `broker`, `sim`, `shell`, `bridge`, `lib`, `check` | One command starts any part of the stack |
| `entrypoint.sh` | Loads the ROS environment before running a command | Every ROS container needs it |
| `fastdds_shm.xml` | ROS 2 transport settings for large images | Without it, 640×480 camera frames drop to about 2 fps |
| `mosquitto.conf` | Settings of the bundled broker: MQTT on 1883, WebSocket on 9001, anonymous | For machines without their own broker |
| `requirements-lib.lock`, `requirements-dev.lock` | Exact versions of every package, including indirect ones | Two builds of the same commit install identical packages |
| `viewer/index.html` | The browser page on port 8088: live picture and analysis results over WebSocket MQTT | Watch the robot's view from any browser |
| `README.md` | Images, reproducibility, ports, DDS details | Reference for the container side |

### `training/`: VLA fine-tuning, in its own environment

Not part of the `carma` package: torch and LeRobot live only in the
`carma-train` conda environment, so the library's pinned dependencies are
untouched.

| File | What it does | Why it is needed |
|---|---|---|
| `README.md` | Installing the environment, fetching assets, both smoke tests, what 8 GB allows | How to reproduce the fine-tuning setup on a clean machine |
| `requirements.in` | The top-level pins: torch, torchvision, torchcodec, LeRobot | What the lock was resolved from |
| `requirements-win64.lock`, `conda-win64.lock` | Every pip and conda package of the environment, exact versions | Two installs give identical environments |
| `smoke_inference.py` | Loads SmolVLA on the GPU and times action-chunk prediction | Measures latency and memory for any state/action size |
| `measure_train.py` | Runs `lerobot-train` and records peak GPU memory | LeRobot does not report peak memory itself |
| `windows_compat.py` | Makes LeRobot's `checkpoints/last` link a directory junction when Windows refuses the symlink | Without it a run on Windows stops at its first checkpoint |
| `sweep_batch.py` | Short training runs over batch sizes and precision settings | Finds the largest batch that fits on the card |
| `results/` | Measured numbers as JSON, and `RESULTS.md`, the dated log of every run | The record that reports and papers cite |

### `ros2_ws/`: ROS 2 packages, run inside the containers

| File | What it does | Why it is needed |
|---|---|---|
| `README.md` | The four packages and how to build them | |
| `carma_vision/carma_vision/mqtt_camera_node.py` | Turns JPEG frames from the broker into ROS stereo image topics; one instance per camera (world, arm) | Lets a PC camera stand in for the robot's cameras |
| `carma_vision/carma_vision/analysis_node.py` | Runs the scene analyzers from `configs/analyzers/` on the camera and publishes the results | Live plant-health results, or your own model's |
| `carma_vision/carma_vision/stream_node.py` | Publishes the world camera to the broker as side-by-side JPEG, and optionally to the headset | Feeds the browser viewer and the Quest |
| `carma_vision/carma_vision/mqtt_link.py` | A small auto-reconnecting MQTT client | Shared by the three nodes |
| `carma_vision/carma_vision/stereo.py` | Joins and encodes left and right frames | Kept free of ROS so it can be unit-tested |
| `carma_vision/config/vision.yaml` | Topics, rates and broker settings of the three nodes | Change camera topics without touching code |
| `carma_vision/launch/vision.launch.py` | Starts the three nodes; `mqtt_camera` switches the camera source | Used by the `vision` service |
| `carma_vision/test/test_stereo.py` | Tests `stereo.py` | Run with `colcon test` |
| `carma_voice/carma_voice/stt_node.py` | Transcribes each speech clip offline with faster-whisper, rejects noise, returns text and intent | The speech channel to the robot |
| `carma_voice/carma_voice/audio.py` | Decodes the WAV clips | Kept free of ROS so it can be unit-tested |
| `carma_voice/config/voice.yaml` | Model path, languages, CPU settings | Change speech settings without touching code |
| `carma_voice/launch/voice.launch.py` | Starts the speech node | Used by the `voice` service |
| `carma_voice/test/test_audio.py` | Tests `audio.py` | Run with `colcon test` |
| `carma_sim/worlds/row_crop.sdf` | The row-crop field, as a template expanded per crop stage | The simulated experimental world |
| `carma_sim/models/carma_ugv/model.sdf`, `model.config` | The tractor-style UGV: body, wheels, stereo camera, depth camera, odometry | The simulated robot, shaped like the real one |
| `carma_sim/models/so101_follower/model.sdf`, `model.config` | The SO-101 arm: six position-controlled joints from the official URDF, and the stereo camera it carries | The arm, testable before the real one is built |
| `carma_sim/models/carma_ugv_arm/model.sdf`, `model.config` | `carma_ugv` unchanged, plus the arm on a plate over the front guard | Spawned instead of `carma_ugv` with `arm:=true` |
| `carma_sim/launch/row_crop.launch.py` | Expands the world for the chosen stage, then starts Gazebo, the ROS bridge and `bgr_relay` | Used by the `sim` service |
| `carma_sim/config/bridge.yaml` | Maps Gazebo topics onto the ROS topic names | The simulator meets the same topic contract as the real robot |
| `carma_sim/scripts/bgr_relay.py` | Converts the simulator's RGB images to BGR | Gazebo cannot output BGR, and the project uses BGR throughout |
| `carma_sim/README.md` | World, stage argument, robot geometry, sensor details | Reference for the simulation |
| `carma_vla/carma_vla/vla_node.py` | Takes both stereo cameras, the arm's joint angles and the operator's speech, asks the policy server, publishes the action, and with `arm:=true` plays arm chunks out | The live link from sensors and voice to the VLA |
| `carma_vla/carma_vla/pipeline.py` | What speech does to the instruction, how frames become one request, how a chunk is played and clamped | Kept free of ROS so it can be unit-tested |
| `carma_vla/config/vla.yaml`, `launch/vla.launch.py` | Policy address, rate, primary camera, image size, drive and arm switches, joint limits | Used by the `vla` service |
| `carma_vla/test/test_pipeline.py` | Tests `pipeline.py` | Run with `colcon test` |
| `carma_bridge/carma_bridge/bridge_node.py` | Skeleton of the node that will join ROS and the library | Marks where the Gazebo experiment link will be built |
| `carma_bridge/README.md` | The ROS topic contract between robot and library | Both the simulator and the real robot must publish these topics |
| `*/package.xml`, `setup.py`, `setup.cfg`, `CMakeLists.txt`, `resource/*`, `__init__.py` | ROS package metadata and build files | Required by `colcon` to build and install each package |

### `REPORTS/`: written reports

| File | What it does | Why it is needed |
|---|---|---|
| `README.md` | Index of the reports | Says what a report holds and what it must not |
| `arm-benchmark/` | The SmolVLA-against-X-VLA report as a Word file, its figures, the script that draws them and the data they come from | A report that can be rebuilt and checked on its own; `arm-benchmark/README.md` has the steps |

### `docs/`: design and reference

| File | Covers |
|---|---|
| `architecture.md` | The control loop, and why memory and backbone never meet |
| `cost-model.md` | How the three options are priced, and in what units |
| `data-schema.md` | The decision-log format, versioned because it is a research artefact |
| `experiments.md` | Running seeds and conditions so the comparison is controlled |
| `simulation.md` | Why Gazebo Harmonic, and how the simulator connects |
| `vla-integration.md` | Serving a VLA model behind `policy_server.py` |
| `arm-policies.md` | SmolVLA and X-VLA on the SO-101: checkpoints, specs, units, uncertainty, the benchmark |
| `analyzers.md` | Adding your own perception model |
| `speech.md` | The speech pipeline and its rejection rules |
| `streaming.md` | Camera streaming now (MQTT) and the planned VR transport (WebRTC) |
| `stereo_depth.md` | Measured depth accuracy, range and shutter of the 3D USB camera |
| `arm.md` | The SO-101 arm: where it is mounted and why, its topics in simulation, driving the real one |

### `vr/unity/`: the Meta Quest 2 app

| File or folder | What it does | Why it is needed |
|---|---|---|
| `README.md` | Opening, building and running the app | The headset guide |
| `.gitignore`, `.gitattributes` | Keep Unity's generated folders and keystores out of git; store binary assets correctly | The project stays a few MB in git |
| `CarmaQuest/Assets/Scenes/SampleScene.unity` | The only scene in the build | The app itself |
| `CarmaQuest/Assets/Scripts/mqttReceiver.cs` | Receives camera frames on `MqttVidFeed` and draws them | The video in the headset |
| `CarmaQuest/Assets/Scripts/CarmaStereoDisplay.cs` | Detects mono or side-by-side frames and sets up the 3D display | Stereo without configuration |
| `CarmaQuest/Assets/Shaders/StereoSideBySide.shader`, `.mat` | Shows the left half of the frame to the left eye and the right half to the right eye | The original per-eye-layer method renders nothing under OpenXR |
| `CarmaQuest/Assets/Scripts/mqttController.cs` | Publishes head rotation, thumbstick and the A button on `test1` | Controller input for the robot side |
| `CarmaQuest/Assets/Scripts/CarmaVoice.cs` | Push-to-talk: records while B is held, sends the clip, shows the reply | Operator speech from the headset |
| `CarmaQuest/Assets/Materials/*.mat` | Materials of the scene's panels | Referenced by the scene |
| `CarmaQuest/Assets/M2Mqtt/` | M2Mqtt, the MQTT client library (third party) | All of the headset's networking uses it |
| `CarmaQuest/Assets/M2MqttUnity/Scripts/M2MqttUnityClient.cs` | Unity wrapper around M2Mqtt | Base class of `mqttReceiver` and `CarmaVoice` |
| `CarmaQuest/Assets/LegacyOVR/` | Two prefabs and a script from the old Oculus Integration (README inside) | The scene still uses them, and Meta XR SDK 205 no longer ships them |
| `CarmaQuest/Assets/Plugins/Android/AndroidManifest.xml` | Android permissions, including the microphone | Without it the headset cannot record speech |
| `CarmaQuest/Assets/Resources/*.asset` | Meta XR SDK runtime settings: build config, audio, Immersive Debugger, input actions | Read by the Meta SDK at run time |
| `CarmaQuest/Assets/Oculus/OculusProjectConfig.asset` | Meta project settings: target devices, hand-tracking support | Read by the Meta SDK when building |
| `CarmaQuest/Assets/XR/` | XR plug-in management: the OpenXR loader and its settings, plus mock-HMD and XR Simulation settings | Makes OpenXR the headset runtime |
| `CarmaQuest/Assets/TextMesh Pro/` | TextMesh Pro essential resources: fonts and shaders | Kept for the Meta SDK's interface components, which can draw text with it |
| `CarmaQuest/Assets/CompositionLayers/`, `StreamingAssets.meta` | Folder markers created by packages | Without them Unity recreates the folders and reports changes |
| `CarmaQuest/Packages/manifest.json`, `packages-lock.json` | The Unity packages and their exact versions (Meta XR SDK 205, OpenXR, MCP for Unity) | Unity downloads them on first open, so they need not be committed |
| `CarmaQuest/ProjectSettings/*` | Unity project settings: player, Android build, graphics, input, XR, quality | Without them the project would not build the same way |
| `CarmaQuest/.vsconfig` | Visual Studio workload hint for C# editing | Visual Studio offers the right components |
