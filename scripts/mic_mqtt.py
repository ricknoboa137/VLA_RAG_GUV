"""Capture operator speech from a local microphone and send it to the broker.

The stand-in for the VR headset microphone while there is no headset app. Each
utterance is cut out by a simple energy detector (or by pressing Enter) and
published as one WAV clip to ``<prefix>/operator/audio``; the ``carma_voice``
container transcribes it, and this script prints what was understood.

    pip install -e ".[voice,stream]"
    python scripts/mic_mqtt.py --list-devices
    python scripts/mic_mqtt.py --device 14
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import threading
import wave

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd

BLOCK_S = 0.03


def _wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buf.getvalue()


def _level_dbfs(block: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2))) / 32768.0
    return 20.0 * np.log10(max(rms, 1e-6))


def _on_message(_client: mqtt.Client, _userdata: object, message: mqtt.MQTTMessage) -> None:
    try:
        r = json.loads(message.payload)
    except ValueError:
        return
    print(
        f'  heard [{r.get("language")}] "{r.get("text")}" -> {r.get("intent")} '
        f"({r.get('audio_s')} s audio, {r.get('stt_latency_s')} s to text)",
        flush=True,
    )


def _vad_loop(stream: sd.InputStream, rate: int, args: argparse.Namespace, send: object) -> None:
    block = int(rate * BLOCK_S)
    print("calibrating: stay quiet for one second…", flush=True)
    noise = [_level_dbfs(stream.read(block)[0]) for _ in range(int(1.0 / BLOCK_S))]
    threshold = args.threshold if args.threshold is not None else max(np.median(noise) + 12, -50)
    print(
        f"noise floor {np.median(noise):.0f} dBFS, speech threshold {threshold:.0f} dBFS",
        flush=True,
    )
    print("listening — speak a command (Ctrl+C to quit)", flush=True)

    pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=int(0.3 / BLOCK_S))
    clip: list[np.ndarray] = []
    loud_blocks = quiet_blocks = 0
    while True:
        data = stream.read(block)[0][:, 0].copy()
        loud = _level_dbfs(data) > threshold
        if not clip:
            pre_roll.append(data)
            loud_blocks = loud_blocks + 1 if loud else 0
            if loud_blocks >= 2:
                clip = list(pre_roll)
                quiet_blocks = 0
            continue
        clip.append(data)
        quiet_blocks = 0 if loud else quiet_blocks + 1
        length_s = len(clip) * BLOCK_S
        if quiet_blocks * BLOCK_S >= args.silence_s or length_s >= args.max_clip_s:
            speech_s = length_s - quiet_blocks * BLOCK_S
            if speech_s >= args.min_speech_s:
                send(np.concatenate(clip), rate)  # type: ignore[operator]
            clip, loud_blocks = [], 0
            pre_roll.clear()


def _ptt_loop(stream: sd.InputStream, rate: int, args: argparse.Namespace, send: object) -> None:
    block = int(rate * BLOCK_S)
    while True:
        input("press Enter to talk…")
        stop = threading.Event()
        chunks: list[np.ndarray] = []

        def record(stop: threading.Event, chunks: list[np.ndarray]) -> None:
            while not stop.is_set() and len(chunks) * BLOCK_S < args.max_clip_s:
                chunks.append(stream.read(block)[0][:, 0].copy())

        worker = threading.Thread(target=record, args=(stop, chunks))
        worker.start()
        input("  recording — press Enter to send")
        stop.set()
        worker.join()
        if chunks:
            send(np.concatenate(chunks), rate)  # type: ignore[operator]


def main() -> int:
    """Capture utterances and publish them until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--mode", choices=("vad", "ptt"), default="vad")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--prefix", default="carma")
    parser.add_argument("--threshold", type=float, default=None, help="speech level in dBFS")
    parser.add_argument("--silence-s", type=float, default=0.8, help="silence that ends a clip")
    parser.add_argument("--min-speech-s", type=float, default=0.3)
    parser.add_argument("--max-clip-s", type=float, default=10.0)
    args = parser.parse_args()

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                api = sd.query_hostapis(d["hostapi"])["name"]
                print(f"[{i:2}] {d['name']}  ({api}, {int(d['default_samplerate'])} Hz)")
        return 0

    info = sd.query_devices(args.device, "input")
    rate = int(info["default_samplerate"])

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="carma-mic")
    client.on_message = _on_message
    client.connect(args.host, args.port, keepalive=30)
    client.subscribe(f"{args.prefix}/operator/utterance", qos=1)
    client.loop_start()
    audio_topic = f"{args.prefix}/operator/audio"

    def send(samples: np.ndarray, sample_rate: int) -> None:
        client.publish(audio_topic, _wav(samples, sample_rate), qos=1)
        print(f"  sent {samples.size / sample_rate:.1f} s clip", flush=True)

    print(f"microphone: {info['name']} at {rate} Hz -> {args.host}:{args.port}/{audio_topic}")
    try:
        with sd.InputStream(
            device=args.device, samplerate=rate, channels=1, dtype="int16"
        ) as stream:
            loop = _vad_loop if args.mode == "vad" else _ptt_loop
            loop(stream, rate, args, send)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
