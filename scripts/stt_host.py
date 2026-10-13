"""Transcribe operator utterances on the host, without ROS or Docker.

    pip install -e ".[voice,stream]"
    python scripts/fetch_models.py faster-whisper-base
    python scripts/stt_host.py --host 192.168.0.153

Subscribes to ``<prefix>/operator/audio``, transcribes each WAV clip with
faster-whisper and publishes the same JSON result to
``<prefix>/operator/utterance`` that ``carma_voice``'s ROS node publishes, so
the Quest client cannot tell the difference.

It exists because the container needs Docker running and a ROS install, which
is a lot of moving parts for a demonstration or for debugging a microphone.
The decision logic is deliberately identical to
``ros2_ws/src/carma_voice/carma_voice/stt_node.py`` -- language restricted to
the operator's languages, and the same three rejection rules -- so results
from either path are comparable. Keep them in step: a change to one belongs in
both, or the comparison quietly stops holding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
from faster_whisper import WhisperModel

from carma.operator.speech import parse_utterance


def _decode_wav(payload: bytes) -> tuple[np.ndarray, float]:
    """16-bit PCM WAV bytes to mono float32 at 16 kHz, with its duration."""
    with wave.open(BytesIO(payload), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        msg = f"expected 16-bit PCM, got {width * 8}-bit"
        raise ValueError(msg)

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    duration_s = samples.size / float(rate)
    if rate != 16000:
        # Whisper wants 16 kHz; linear resampling is enough for speech.
        target = int(round(duration_s * 16000))
        samples = np.interp(
            np.linspace(0.0, samples.size - 1, target, dtype=np.float32),
            np.arange(samples.size, dtype=np.float32),
            samples,
        ).astype(np.float32)
    return samples, duration_s


def main() -> int:
    """Listen for clips and publish transcriptions until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--prefix", default="carma")
    parser.add_argument("--model", type=Path, default=Path("assets/faster-whisper-base"))
    parser.add_argument("--languages", default="en,es", help="operator languages to choose among")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0)
    parser.add_argument("--max-no-speech-prob", type=float, default=0.6)
    parser.add_argument("--operator-id", default="operator-0")
    args = parser.parse_args()

    languages = [code.strip() for code in args.languages.split(",") if code.strip()]
    in_topic = f"{args.prefix}/operator/audio"
    out_topic = f"{args.prefix}/operator/utterance"

    print(f"loading {args.model} ({args.compute_type})...", flush=True)
    model = WhisperModel(
        str(args.model),
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        local_files_only=True,
    )

    def on_clip(_client: mqtt.Client, _userdata: object, message: mqtt.MQTTMessage) -> None:
        received = time.monotonic()
        stamp = time.time()
        try:
            samples, audio_s = _decode_wav(message.payload)
        except (ValueError, wave.Error) as exc:
            print(f"  bad clip: {exc}", file=sys.stderr, flush=True)
            return

        _, _, all_probs = model.detect_language(audio=samples, vad_filter=True)
        allowed = [(code, prob) for code, prob in all_probs if code in languages]
        language, language_prob = (
            max(allowed, key=lambda item: item[1]) if allowed else (languages[0], 0.0)
        )

        segments = list(
            model.transcribe(
                samples,
                language=language,
                beam_size=args.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )[0],
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else None
        no_speech_prob = max(s.no_speech_prob for s in segments) if segments else None

        if not segments or not text:
            rejected = "no_speech"
        elif avg_logprob is not None and avg_logprob < args.min_avg_logprob:
            rejected = "low_confidence"
        elif no_speech_prob is not None and no_speech_prob > args.max_no_speech_prob:
            rejected = "no_speech"
        else:
            rejected = None

        intent = parse_utterance("" if rejected else text, language=language)
        latency_s = time.monotonic() - received
        result = {
            "stamp": stamp,
            "operator_id": args.operator_id,
            "text": text,
            "intent": intent.kind.value,
            "rejected": rejected,
            "language": language,
            "language_probability": round(float(language_prob), 3),
            "audio_s": round(audio_s, 3),
            "stt_latency_s": round(latency_s, 3),
            "avg_logprob": None if avg_logprob is None else round(avg_logprob, 3),
            "no_speech_prob": None if no_speech_prob is None else round(no_speech_prob, 3),
        }
        _client.publish(out_topic, json.dumps(result, ensure_ascii=False, sort_keys=True), qos=1)

        verdict = f"rejected ({rejected})" if rejected else intent.kind.value
        print(
            f"[{language} {language_prob:.2f}] {verdict}: {text!r} "
            f"({audio_s:.1f} s audio, {latency_s:.2f} s)",
            flush=True,
        )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="carma-stt-host")
    client.on_message = on_clip
    client.connect(args.host, args.port, keepalive=30)
    client.subscribe(in_topic, qos=1)
    print(f"listening on {in_topic} at {args.host}:{args.port} — Ctrl+C to stop", flush=True)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
