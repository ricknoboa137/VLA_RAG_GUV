"""Thin, reconnecting MQTT client (paho-mqtt 2.x)."""

from __future__ import annotations

from collections.abc import Callable

import paho.mqtt.client as mqtt

MessageHandler = Callable[[str, bytes], None]


class MqttLink:
    """Publishes and subscribes from a background network thread.

    Connection is asynchronous and retried with backoff, so a node starts even
    when the broker is not up yet; messages published while disconnected with
    QoS 0 are dropped, which is the right behaviour for live video.
    Subscriptions are renewed on every reconnect.
    """

    def __init__(self, host: str, port: int, client_id: str, keepalive_s: int = 30) -> None:
        self._handlers: dict[str, MessageHandler] = {}
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.connect_async(host, port, keepalive_s)
        self._client.loop_start()

    @property
    def connected(self) -> bool:
        """True while the broker connection is up."""
        return self._client.is_connected()

    def publish(self, topic: str, payload: bytes | str, qos: int = 0, retain: bool = False) -> None:
        """Queue one message."""
        self._client.publish(topic, payload, qos=qos, retain=retain)

    def subscribe(self, topic: str, handler: MessageHandler, qos: int = 0) -> None:
        """Call ``handler(topic, payload)`` on the network thread for each message.

        ``topic`` must be an exact topic, not a wildcard filter.
        """
        self._handlers[topic] = handler
        if self.connected:
            self._client.subscribe(topic, qos)

    def close(self) -> None:
        """Stop the network thread and disconnect."""
        self._client.disconnect()
        self._client.loop_stop()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:  # noqa: ANN001
        for topic in self._handlers:
            client.subscribe(topic)

    def _on_message(self, client, userdata, message) -> None:  # noqa: ANN001
        handler = self._handlers.get(message.topic)
        if handler is not None:
            handler(message.topic, message.payload)
