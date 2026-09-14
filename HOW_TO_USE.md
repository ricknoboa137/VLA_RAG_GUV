# How to use CARMA

A step-by-step guide to running the system on a development PC: the ROS 2
containers, the camera stream with live analysis, operator speech, and the Meta
Quest 2 client. For what the project is and its design rules, see `README.md`
and `AGENTS.md`.

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

## 7. Quick troubleshooting

| Symptom | Fix |
|---|---|
| Viewer page shows nothing | Broker needs the WebSocket listener (section 4); camera script must be running |
| `vision` logs `0 frames` | Webcam script not running, or sending to another broker/port |
| Speech script prints nothing | Run it in your own terminal, not in the background; check `docker compose -f docker/compose.yaml logs voice` |
| Wrong language or nonsense text | Use `--mode ptt`; set `STT_LANGUAGE=en` or `es` |
| Low camera frame rate in ROS | The image ships `docker/fastdds_shm.xml`; keep `ipc: host` on ROS services |
| Quest: `broker connection failed` | Same Wi-Fi as the broker, correct address, firewall rule |

## Where to read more

| Topic | File |
|---|---|
| Architecture and module rules | `docs/architecture.md`, `AGENTS.md` |
| Cost model (act / retrieve / ask) | `docs/cost-model.md` |
| Running experiments | `docs/experiments.md` |
| Data formats | `docs/data-schema.md` |
| Containers and DDS | `docker/README.md` |
| Streaming and VR | `docs/streaming.md` |
| Speech | `docs/speech.md` |
| Custom perception models | `docs/analyzers.md` |
| Quest client | `vr/unity/README.md` |
