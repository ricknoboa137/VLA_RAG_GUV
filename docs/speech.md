# Operator speech

The operator talks to the robot: commands ("stop", "gira a la izquierda"),
answers to the robot's questions ("yes"), and advice worth remembering ("keep to
the left where the canopy narrows"). Speech is the planned operator channel in
the VR headset; until the headset app exists, a PC microphone stands in for it
over the same pipeline.

## Pipeline

```
microphone ─► utterance clip (WAV) ─► carma/operator/audio ─► carma_voice/stt_node
                                                              faster-whisper (CPU, offline)
                                                              parse_utterance()
                                        carma/operator/utterance ◄─┘  (+ /carma/operator/utterance)
```

| Stage | Now (PC test) | Later (VR) |
|---|---|---|
| Capture | `scripts/mic_mqtt.py` on the PC | Headset microphone in the WebXR page |
| Transport | One WAV clip per utterance over MQTT | Audio track on the same WebRTC session as the video |
| Recognition | `stt_node` in the container | Unchanged |
| Meaning | `carma.operator.speech.parse_utterance` | Unchanged |

Only capture and transport change when the headset arrives.

## Speech recognition

**faster-whisper** (MIT) running Whisper weights converted by Systran (MIT),
fetched once into `assets/` and loaded with `local_files_only`, so nothing
reaches the network at run time and no account is needed. It handles English
and Spanish; the language is auto-detected per clip unless `STT_LANGUAGE` is
set, which helps when very short commands are misdetected.

`faster-whisper-base` is the default for CPU. `faster-whisper-small` is more
accurate and slower; on an NVIDIA board either can run on the GPU.

## From words to intent

`parse_utterance` is a small keyword grammar, not a language model, so its
behaviour is predictable and every case is tested by hand
(`tests/test_speech.py`):

| Intent | Examples |
|---|---|
| `stop` | stop, halt, alto, detente, para (short utterances only) |
| `turn_left` / `turn_right` | turn left, left, gira a la derecha |
| `go` | go forward, continue, sigue adelante |
| `yes` / `no` | yes, sí, that's right, no, negativo |
| `correction` | any longer utterance: advice destined for memory |
| `none` | nothing recognisable was said |

Commands must be at most four words, so advice that mentions a direction is
never taken as a steering command. Stop always wins.

## Safety

Speech recognition takes a noticeable fraction of a second and can mishear. It
is **not** the emergency stop. The robot needs a physical or radio e-stop that
does not depend on this pipeline.

## Why latency is recorded

Operator attention is the resource CARMA measures. Each result carries
`audio_s` (how long the operator spoke) and `stt_latency_s` (receipt to text),
so time spent talking to the robot is accounted for, never hidden. The schema
is in `docs/data-schema.md`.

## Try it on a PC

```bash
python scripts/fetch_models.py faster-whisper-base
docker compose -f docker/compose.yaml up -d --build voice
python scripts/mic_mqtt.py --list-devices
python scripts/mic_mqtt.py --device <index>
```

`--mode ptt` switches to press-Enter-to-talk. A Meta headset connected through
Oculus Link appears as a microphone too, so the real headset microphone can be
tested this way before any VR app exists.
