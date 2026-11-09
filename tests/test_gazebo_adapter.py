"""GazeboAdapter against a fake bridge: protocol, decoding and failure modes.

No ROS and no Gazebo: a thread plays the bridge on a local socket. The real
simulator's determinism is checked separately, by the replay procedure in
``docs/simulation.md``, because it needs the sim container.
"""

from __future__ import annotations

import base64
import importlib.util
import socket
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from carma.sim.gazebo import GazeboAdapter
from carma.sim.gazebo_wire import recv_frame, send_frame
from carma.types import SimulatorError
from carma.types.core import Action, PhenologyStage, Pose

_HEIGHT, _WIDTH = 6, 8
_START = Pose(0.0, 0.0, 0.0)


def _reply(t: float, x: float = 0.0) -> dict[str, Any]:
    image = np.full((_HEIGHT, _WIDTH, 3), (10, 20, 30), dtype=np.uint8)
    ok, png = cv2.imencode(".png", image)
    assert ok
    depth = np.full((_HEIGHT, _WIDTH), 2.5, dtype="<f4")
    return {
        "ok": True,
        "t": t,
        "pose": {"x": x, "y": 0.5, "yaw": 0.25},
        "rgb_png": base64.b64encode(png.tobytes()).decode("ascii"),
        "depth_f32": base64.b64encode(depth.tobytes()).decode("ascii"),
        "height": _HEIGHT,
        "width": _WIDTH,
    }


class FakeBridge:
    """A one-client bridge that records requests and answers from a script."""

    def __init__(self, script: Any = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._script = script or self._default
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _default(request: dict[str, Any], count: int) -> dict[str, Any]:
        return _reply(t=0.1 * count, x=0.05 * count)

    def _run(self) -> None:
        conn, _ = self._server.accept()
        with conn:
            while True:
                try:
                    request = recv_frame(conn)
                except (ConnectionError, OSError):
                    return
                self.requests.append(request)
                if request["op"] == "close":
                    return
                try:
                    send_frame(conn, self._script(request, len(self.requests) - 1))
                except ConnectionError:
                    return  # the script asked the bridge to hang up

    def close(self) -> None:
        self._server.close()


@pytest.fixture
def bridge() -> Any:
    fake = FakeBridge()
    yield fake
    fake.close()


def _adapter(port: int, **kwargs: Any) -> GazeboAdapter:
    return GazeboAdapter(bridge_port=port, timeout_s=5.0, **kwargs)


def test_reset_decodes_an_observation(bridge: FakeBridge) -> None:
    sim = _adapter(bridge.port, instruction="go", stage="CANOPY_CLOSURE")
    obs = sim.reset(seed=7, start=_START)
    sim.close()
    assert bridge.requests[0] == {
        "op": "reset",
        "seed": 7,
        "start": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    }
    assert obs.rgb.shape == (_HEIGHT, _WIDTH, 3)
    assert obs.rgb.dtype == np.uint8
    assert tuple(obs.rgb[0, 0]) == (10, 20, 30)  # BGR order survives the PNG round trip
    assert obs.depth is not None
    assert obs.depth.dtype == np.float32
    assert float(obs.depth[0, 0]) == 2.5
    assert (obs.pose.y, obs.pose.yaw) == (0.5, 0.25)
    assert obs.instruction == "go"
    assert obs.stage is PhenologyStage.CANOPY_CLOSURE


def test_step_sends_the_action_and_one_control_period(bridge: FakeBridge) -> None:
    sim = _adapter(bridge.port, control_hz=20.0)
    sim.reset(seed=0, start=_START)
    obs = sim.step(Action(linear=0.4, angular=-0.2))
    sim.close()
    assert bridge.requests[1] == {
        "op": "step",
        "linear": 0.4,
        "angular": -0.2,
        "stop": False,
        "dt_s": 0.05,
    }
    assert obs.t == pytest.approx(0.1)


def test_stop_is_forwarded(bridge: FakeBridge) -> None:
    sim = _adapter(bridge.port)
    sim.reset(seed=0, start=_START)
    sim.step(Action(linear=0.0, angular=0.0, stop=True))
    sim.close()
    assert bridge.requests[1]["stop"] is True


def test_close_tells_the_bridge(bridge: FakeBridge) -> None:
    sim = _adapter(bridge.port)
    sim.reset(seed=0, start=_START)
    sim.close()
    sim.close()  # idempotent
    bridge._thread.join(timeout=2.0)
    assert bridge.requests[-1] == {"op": "close"}


def test_step_before_reset_is_an_error() -> None:
    with pytest.raises(SimulatorError, match="reset"):
        GazeboAdapter().step(Action(0.1, 0.0))


def test_unreachable_bridge_names_the_address() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    with pytest.raises(SimulatorError, match=f"127.0.0.1:{port}"):
        _adapter(port).reset(seed=0, start=_START)


def test_a_refusal_carries_the_bridge_message() -> None:
    def refuse(request: dict[str, Any], count: int) -> dict[str, Any]:
        return {"ok": False, "error": "no odometry after reset"}

    fake = FakeBridge(refuse)
    sim = _adapter(fake.port)
    try:
        with pytest.raises(SimulatorError, match="no odometry after reset"):
            sim.reset(seed=0, start=_START)
    finally:
        sim.close()
        fake.close()


def test_a_dropped_connection_is_a_simulator_error() -> None:
    def hang_up(request: dict[str, Any], count: int) -> dict[str, Any]:
        raise ConnectionError

    fake = FakeBridge(hang_up)
    sim = _adapter(fake.port)
    try:
        with pytest.raises(SimulatorError, match="failed on 'reset'"):
            sim.reset(seed=0, start=_START)
        # The adapter forgot the dead socket: a step now says "not connected".
        with pytest.raises(SimulatorError, match="not connected"):
            sim.step(Action(0.1, 0.0))
    finally:
        fake.close()


def test_a_mismatched_image_is_rejected() -> None:
    def wrong_size(request: dict[str, Any], count: int) -> dict[str, Any]:
        reply = _reply(0.0)
        reply["width"] = _WIDTH + 1
        return reply

    fake = FakeBridge(wrong_size)
    sim = _adapter(fake.port)
    try:
        with pytest.raises(SimulatorError, match="stated size"):
            sim.reset(seed=0, start=_START)
    finally:
        sim.close()
        fake.close()


def test_a_non_positive_rate_is_rejected() -> None:
    with pytest.raises(SimulatorError, match="control_hz"):
        GazeboAdapter(control_hz=0.0)


def test_the_two_copies_of_the_wire_format_agree() -> None:
    """The bridge owns the format; the library's copy must behave identically."""
    path = Path(__file__).resolve().parents[1] / "ros2_ws/src/carma_bridge/carma_bridge/protocol.py"
    spec = importlib.util.spec_from_file_location("bridge_protocol", path)
    assert spec is not None and spec.loader is not None
    bridge_protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge_protocol)

    import carma.sim.gazebo_wire as library_wire

    assert bridge_protocol.MAX_FRAME_BYTES == library_wire.MAX_FRAME_BYTES
    message = {"op": "step", "linear": 0.5, "angular": -0.1, "stop": False, "dt_s": 0.1}
    for sender, receiver in (
        (bridge_protocol.send_frame, library_wire.recv_frame),
        (library_wire.send_frame, bridge_protocol.recv_frame),
    ):
        left, right = socket.socketpair()
        with left, right:
            sender(left, message)
            assert receiver(right) == message
