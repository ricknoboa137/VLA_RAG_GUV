"""The wire format between ``GazeboAdapter`` and the bridge node.

One request, one reply, strictly alternating: a 4-byte big-endian length, then
that many bytes of UTF-8 JSON. No ROS import, so a test can load it with the
library's copy (``src/carma/sim/gazebo_wire.py``) and check the two agree.

Requests (library to bridge)::

    {"op": "reset", "seed": int}
    {"op": "step", "linear": float, "angular": float, "stop": bool, "dt_s": float}
    {"op": "close"}

Replies (bridge to library)::

    {"ok": true, "t": float, "pose": {"x", "y", "yaw"},
     "rgb_png": str, "depth_f32": str, "height": int, "width": int,
     "image_t": float, "sync_slop_s": float}
    {"ok": false, "error": str}

``rgb_png`` is base64 of a PNG (lossless, BGR order as OpenCV writes it).
``depth_f32`` is base64 of little-endian float32 metres, row-major,
``height`` by ``width``. ``t`` is simulated seconds since the last reset.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

# A frame bigger than this is a framing error, not a large image: a 640x480
# float32 depth map base64-encoded is about 1.6 MB.
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
