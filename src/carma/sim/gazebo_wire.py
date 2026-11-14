"""Length-prefixed JSON framing for the Gazebo bridge.

This is the library's copy of ``ros2_ws/src/carma_bridge/carma_bridge/protocol.py``,
which owns the format and its documentation. The two are duplicated, not shared,
because the bridge is a ROS package that must not import ``carma`` and the
library must not import the workspace; ``tests/test_gazebo_adapter.py`` loads
both and fails if their behaviour drifts.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_FRAME_BYTES = 64 * 1024 * 1024


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the connection mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send one length-prefixed JSON message."""
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    """Receive one length-prefixed JSON message."""
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {length} bytes exceeds the {MAX_FRAME_BYTES} byte limit")
    decoded = json.loads(_recv_exact(sock, length).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("a frame must hold a JSON object")
    return decoded
