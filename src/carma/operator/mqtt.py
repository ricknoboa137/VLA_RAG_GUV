"""MQTT-backed operator for the Quest push-to-talk voice channel.

The Quest app (CarmaVoice) publishes 16-bit PCM WAV to ``carma/operator/audio``.
The ``carma_voice`` container transcribes it, classifies the intent and
publishes JSON to ``carma/operator/utterance``::

    {"text": "...", "intent": "correction", "language": "en",
     "rejected": null, "audio_s": 2.1, "stt_latency_s": 0.4}

Only ``intent == "correction"`` becomes a :class:`Correction`; other intents are
dropped here. Voice is push-to-talk, so this channel cannot put a question to
the operator: :meth:`MqttOperator.ask` raises :class:`OperatorUnavailableError`
rather than inventing an answer, which would hide operator cost.

``paho-mqtt`` is an optional dependency (the ``stream`` extra). It is imported
when an operator is constructed, so registering this class costs nothing on a
machine without it.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Any

from carma.operator.registry import OPERATORS
from carma.types import OperatorUnavailableError
from carma.types.core import (
    Correction,
    IntentKind,
    Observation,
    OperatorQuery,
    OperatorResponse,
)


@dataclass(frozen=True)
class _PendingCorrection:
    """A correction waiting for the observation it will be stamped with."""

    text: str
    language: str


def parse_utterance_payload(raw: bytes) -> _PendingCorrection | None:
    """Turn one ``carma/operator/utterance`` message into a pending correction.

    Returns None for anything that is not a usable correction: malformed JSON,
    a rejected clip, empty text, or an intent other than ``correction``.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("rejected"):
        return None
    text = str(payload.get("text", "")).strip()
    if not text:
        return None
    try:
        intent = IntentKind(str(payload.get("intent", "correction")).lower())
    except ValueError:
        intent = IntentKind.CORRECTION
    if intent is not IntentKind.CORRECTION:
        return None
    return _PendingCorrection(text=text, language=str(payload.get("language", "")))


@OPERATORS.register("mqtt")
class MqttOperator:
    """Operator whose corrections arrive as transcribed speech over MQTT.

    Args:
        broker: MQTT broker host.
        port: MQTT broker port.
        operator_id: Provenance tag recorded on every correction.
        username: Optional MQTT username.
        password: Optional MQTT password.
        client_id: MQTT client id; defaults to ``carma-operator-<operator_id>``.
        utterance_topic: Topic the ``carma_voice`` container publishes to.
        keepalive_s: MQTT keepalive interval in seconds.
        connect_timeout_s: How long to wait for the first connection, seconds.
    """

    def __init__(
        self,
        broker: str = "127.0.0.1",
        port: int = 1883,
        operator_id: str = "quest",
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        utterance_topic: str = "carma/operator/utterance",
        keepalive_s: int = 60,
        connect_timeout_s: float = 10.0,
    ) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            msg = "the mqtt operator needs paho-mqtt: pip install -e '.[stream]'"
            raise OperatorUnavailableError(msg) from exc

        self._operator_id = operator_id
        self._utterance_topic = utterance_topic
        self._queue: queue.Queue[_PendingCorrection] = queue.Queue()
        self._connected = threading.Event()
        self._client: Any = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or f"carma-operator-{operator_id}",
            clean_session=True,
        )
        if username is not None:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._client.connect_async(broker, port, keepalive_s)
        self._client.loop_start()
        if not self._connected.wait(timeout=connect_timeout_s):
            self._client.loop_stop()
            msg = f"mqtt operator did not connect to {broker}:{port} in {connect_timeout_s} s"
            raise OperatorUnavailableError(msg)

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        if not reason_code.is_failure:
            client.subscribe(self._utterance_topic, qos=1)
            self._connected.set()

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        if msg.topic != self._utterance_topic:
            return
        pending = parse_utterance_payload(msg.payload)
        if pending is not None:
            self._queue.put(pending)

    def _on_disconnect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any
    ) -> None:
        self._connected.clear()

    def ask(self, query: OperatorQuery) -> OperatorResponse:
        """Not supported: push-to-talk voice cannot answer a question on demand."""
        msg = "the mqtt voice operator is push-to-talk only and cannot answer questions"
        raise OperatorUnavailableError(msg)

    def poll_corrections(self, obs: Observation) -> tuple[Correction, ...]:
        """Every correction received since the last call, stamped with ``obs``."""
        results: list[Correction] = []
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break
            results.append(
                Correction(text=pending.text, observation=obs, operator_id=self._operator_id)
            )
        return tuple(results)

    def close(self) -> None:
        """Disconnect and stop the network thread."""
        self._client.loop_stop()
        self._client.disconnect()
