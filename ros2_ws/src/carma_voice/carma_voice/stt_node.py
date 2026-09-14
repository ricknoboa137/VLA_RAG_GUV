"""Offline speech-to-text for operator utterances.

Receives one WAV clip per utterance on ``<p>/operator/audio`` (MQTT), transcribes
it with faster-whisper on the CPU, classifies it with
``carma.operator.speech.parse_utterance``, and publishes the result to
``<p>/operator/utterance`` (MQTT, JSON) and ``/carma/operator/utterance`` (ROS).

Two safeguards against the failure modes seen on a live microphone:

* **Language is chosen among the operator's languages only** (``languages``,
  default ``en,es``). Free detection over ~100 languages misreads short clips
  as Russian, Japanese or Italian and then transcribes nonsense in that language.
* **Unreliable transcriptions are rejected**: no speech left after voice
  activity filtering, a low mean log-probability, or a high no-speech
  probability. A rejected clip keeps its raw text for analysis but yields intent
  ``none``, so background talk and hallucinations never become commands.

The model is loaded from disk only (``local_files_only``): fetch it first with
``python scripts/fetch_models.py faster-whisper-base``. No network at inference.

Transcription runs on a worker thread so a slow clip never stalls the MQTT
network thread. ``stt_latency_s`` in every result is measured, not estimated:
operator response time is a research measurement in this project.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from faster_whisper import WhisperModel
from rclpy.node import Node
from std_msgs.msg import String

from carma.operator.speech import parse_utterance
from carma_vision.mqtt_link import MqttLink
from carma_voice.audio import decode_wav


class SttNode(Node):
    """Transcribes queued utterance clips one at a time."""

    def __init__(self) -> None:
        super().__init__("carma_stt")
        self.declare_parameter("model_path", "/ros2_ws/assets/faster-whisper-base")
        self.declare_parameter("languages", "en,es")  # operator languages to choose among
        self.declare_parameter("language", "auto")  # "auto" picks from languages; else forced
        self.declare_parameter("compute_type", "int8")
        self.declare_parameter("cpu_threads", 4)
        self.declare_parameter("beam_size", 1)
        self.declare_parameter("max_clip_s", 15.0)
        self.declare_parameter("min_avg_logprob", -1.0)
        self.declare_parameter("max_no_speech_prob", 0.6)
        self.declare_parameter("operator_id", "operator-0")
        self.declare_parameter("mqtt_host", "host.docker.internal")
        self.declare_parameter("mqtt_port", 1883)
        self.declare_parameter("mqtt_prefix", "carma")

        param = self.get_parameter
        model_path = Path(param("model_path").value)
        if not (model_path / "model.bin").is_file():
            msg = (
                f"speech model not found at {model_path}. Fetch it on the host with "
                "'python scripts/fetch_models.py faster-whisper-base' (writes assets/)."
            )
            raise SystemExit(msg)

        started = time.monotonic()
        self._model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type=param("compute_type").value,
            cpu_threads=int(param("cpu_threads").value),
            local_files_only=True,
        )

        self._languages = [
            code.strip().lower() for code in str(param("languages").value).split(",") if code.strip()
        ] or ["en"]
        forced = str(param("language").value).strip().lower()
        self._forced = None if forced in ("", "auto") else forced
        self._beam_size = int(param("beam_size").value)
        self._max_clip_s = float(param("max_clip_s").value)
        self._min_avg_logprob = float(param("min_avg_logprob").value)
        self._max_no_speech_prob = float(param("max_no_speech_prob").value)
        self._operator_id = param("operator_id").value
        prefix = param("mqtt_prefix").value
        self._out_topic = f"{prefix}/operator/utterance"
        self.get_logger().info(
            f"loaded {model_path.name} in {time.monotonic() - started:.1f} s; "
            f"language {self._forced or 'auto among ' + ','.join(self._languages)}"
        )

        self._pub = self.create_publisher(String, "/carma/operator/utterance", 10)
        self._clips: queue.Queue[tuple[bytes, float, float]] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._work, name="stt-worker", daemon=True)
        self._worker.start()

        self._mqtt = MqttLink(param("mqtt_host").value, param("mqtt_port").value, "carma-stt")
        self._mqtt.subscribe(f"{prefix}/operator/audio", self._on_clip, qos=1)

    def _on_clip(self, _topic: str, payload: bytes) -> None:
        item = (payload, time.time(), time.monotonic())
        try:
            self._clips.put_nowait(item)
        except queue.Full:
            self.get_logger().warning("speech queue full; dropping the oldest clip")
            self._clips.get_nowait()
            self._clips.put_nowait(item)

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                payload, stamp, received = self._clips.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._transcribe(payload, stamp, received)
            except ValueError as exc:
                self.get_logger().warning(f"ignoring clip: {exc}")

    def _pick_language(self, samples: np.ndarray) -> tuple[str, float]:
        """Most probable language among the operator's languages, with its probability."""
        if self._forced is not None:
            return self._forced, 1.0
        _, _, all_probs = self._model.detect_language(audio=samples, vad_filter=True)
        allowed = [(code, prob) for code, prob in all_probs if code in self._languages]
        if not allowed:
            return self._languages[0], 0.0
        code, prob = max(allowed, key=lambda item: item[1])
        return code, float(prob)

    def _transcribe(self, payload: bytes, stamp: float, received: float) -> None:
        samples, audio_s = decode_wav(payload)
        if audio_s > self._max_clip_s:
            msg = f"clip is {audio_s:.1f} s, longer than max_clip_s={self._max_clip_s}"
            raise ValueError(msg)

        language, language_prob = self._pick_language(samples)
        segments, _info = self._model.transcribe(
            samples,
            language=language,
            beam_size=self._beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else None
        no_speech_prob = max(s.no_speech_prob for s in segments) if segments else None

        if not segments or not text:
            rejected = "no_speech"
        elif avg_logprob is not None and avg_logprob < self._min_avg_logprob:
            rejected = "low_confidence"
        elif no_speech_prob is not None and no_speech_prob > self._max_no_speech_prob:
            rejected = "no_speech"
        else:
            rejected = None

        intent = parse_utterance("" if rejected else text, language=language)
        latency_s = time.monotonic() - received

        result = {
            "stamp": stamp,
            "operator_id": self._operator_id,
            "text": text,
            "intent": intent.kind.value,
            "rejected": rejected,
            "language": language,
            "language_probability": round(language_prob, 3),
            "audio_s": round(audio_s, 3),
            "stt_latency_s": round(latency_s, 3),
            "avg_logprob": None if avg_logprob is None else round(avg_logprob, 3),
            "no_speech_prob": None if no_speech_prob is None else round(no_speech_prob, 3),
        }
        payload_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
        self._pub.publish(String(data=payload_json))
        self._mqtt.publish(self._out_topic, payload_json, qos=1)
        verdict = f"rejected ({rejected})" if rejected else intent.kind.value
        self.get_logger().info(
            f"[{language} {language_prob:.2f}] {verdict}: {text!r} "
            f"({audio_s:.1f} s audio, {latency_s:.2f} s)"
        )

    def destroy_node(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)
        self._mqtt.close()
        super().destroy_node()


def main() -> None:
    """Entry point for ``ros2 run carma_voice stt_node``."""
    rclpy.init()
    node = SttNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
