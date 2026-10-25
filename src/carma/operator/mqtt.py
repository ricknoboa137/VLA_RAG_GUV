"""MQTT-backed operator interface for the Quest voice client.

Subscribes to ``carma/operator/utterance`` where the carma_voice container
publishes transcribed utterances with intent classification, and converts
them into the :class:`carma.types.core.Correction` objects the runtime expects.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import replace
from typing import Any

import paho.mqtt.client as mqtt

from carma.operator.registry import OPERATORS
from carma.operator.speech import IntentKind, parse_utterance
from carma.types.core import Correction, Observation, OperatorQuery, OperatorResponse


@OPERATORS.register("mqtt")
class MqttOperator:
    """Operator backed by the Quest push-to-talk voice channel.

    The Quest app (CarmaVoice) publishes 16-bit PCM WAV to ``carma/operator/audio``.
    The carma_voice container (faster-whisper) transcribes and classifies intent,
    then publishes JSON to ``carma/operator/utterance`` with fields:
        - text: transcribed text
        - intent: one of "stop", "go", "turn_left", "turn_right", "yes", "no", "correction", "none"
        - language: ISO 639-1 code
        - rejected: optional rejection reason (e.g. "no_speech")
        - audio_s: clip duration
        - stt_latency_s: transcription latency

    Only utterances with ``intent == "correction"`` become :class:`Correction`
    objects. Other intents are logged but not turned into memory entries.

    Args:
        broker: MQTT broker hostname or IP.
        port: MQTT broker port (default 1883).
        operator_id: Provenance tag recorded on every correction.
        username: Optional MQTT username.
        password: Optional MQTT password.
        client_id: MQTT client ID (default: auto-generated).
        utterance_topic: Topic the carma_voice container publishes to.
        keepalive_s: MQTT keepalive interval.
        connect_timeout_s: How long to wait for initial connection.
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
        self._broker = broker
        self._port = port
        self._operator_id = operator_id
        self._utterance_topic = utterance_topic
        self._connect_timeout_s = connect_timeout_s

        self._queue: "queue.Queue[Correction]" = queue.Queue()
        self._connected = threading.Event()
        self._client = mqtt.Client(
            client_id=client_id or f"carma-operator-{operator_id}",
            clean_session=True,
        )
        if username is not None:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        # Start the network loop in a background thread
        self._client.loop_start()

        # Block until connected or timeout
        if not self._connected.wait(timeout=connect_timeout_s):
            self._client.loop_stop()
            raise ConnectionError(
                f"MQTT operator did not connect to {broker}:{port} within {connect_timeout_s}s"
            )

    # --------------------------------------------------------------------- MQTT callbacks

    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: dict, rc: int
    ) -> None:
        if rc == 0:
            client.subscribe(self._utterance_topic, qos=1)
            self._connected.set()
        else:
            self._connected.set()  # unblock the waiter; error will surface on use

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        if msg.topic != self._utterance_topic:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            return

        # The carma_voice container may send a rejection (silence, noise)
        if payload.get("rejected"):
            return

        text = payload.get("text", "").strip()
        if not text:
            return

        intent_str = payload.get("intent", "correction").lower()
        try:
            intent = IntentKind(intent_str)
        except ValueError:
            intent = IntentKind.CORRECTION

        # Only CORRECTION intents become memory entries
        if intent is not IntentKind.CORRECTION:
            return

        language = payload.get("language", "")

        # We don't have the observation yet — it will be attached in poll_corrections
        # Store a partial correction with a placeholder observation
        # The actual observation is filled when poll_corrections is called
        self._queue.put(
            _PartialCorrection(
                text=text,
                operator_id=self._operator_id,
                language=language,
                took_over=False,
            )
        )

    def _on_disconnect(
        self, client: mqtt.Client, userdata: Any, rc: int
    ) -> None:
        self._connected.clear()

    # --------------------------------------------------------------------- OperatorInterface

    def ask(self, query: OperatorQuery) -> OperatorResponse:
        """MQTT voice is push-to-talk only; we cannot ask questions back."""
        raise NotImplementedError(
            "MqttOperator does not support ask(); voice is push-to-talk only. "
            "Use a different OperatorInterface for interactive queries."
        )

    def poll_corrections(self, obs: Observation) -> tuple[Correction, ...]:
        """Return all corrections received since the last call, stamped with ``obs``."""
        results: list[Correction] = []
        while True:
            try:
                partial = self._queue.get_nowait()
            except queue.Empty:
                break
            results.append(
                Correction(
                    text=partial.text,
                    observation=obs,
                    operator_id=partial.operator_id,
                    took_over=partial.took_over,
                )
            )
        return tuple(results)

    # --------------------------------------------------------------------- Lifecycle

    def close(self) -> None:
        """Disconnect and stop the network thread."""
        try:
            self._client.unsubscribe(self._utterance_topic)
        except Exception:
            pass
        self._client.loop_stop()
        self._client.disconnect()


class _PartialCorrection:
    """A correction waiting for its observation to be attached."""

    __slots__ = ("text", "operator_id", "language", "took_over")

    def __init__(
        self,
        text: str,
        operator_id: str,
        language: str,
        took_over: bool,
    ) -> None:
        self.text = text
        self.operator_id = operator_id
        self.language = language
        self.took_over = took_over