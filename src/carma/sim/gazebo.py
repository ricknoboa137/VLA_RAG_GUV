"""Gazebo Harmonic adapter, advanced in lockstep through the ROS 2 bridge.

The Python process here holds no ROS dependency: the ROS 2 node lives in
``ros2_ws/src/carma_bridge`` and the two talk over a local TCP socket (format
in ``carma_bridge/protocol.py``). Keeping ``rclpy`` out of the library is what
lets the unit suite run in a plain virtualenv with no ROS installation.

Lockstep is what makes a run reproducible. The world is started paused; each
``step`` sends one velocity command, the bridge advances exactly
``round(dt_s / physics step)`` physics iterations, and only then returns the
observation. Nothing depends on wall-clock time, so the same ``(config, seed,
commit)`` replays the same simulated trajectory, which AGENTS.md section 2.2
requires. Bring-up, topic names and the world files are in
``docs/simulation.md``.
"""

from __future__ import annotations

import base64
import socket
from typing import Any

import cv2
import numpy as np

from carma.sim.gazebo_wire import recv_frame, send_frame
from carma.sim.registry import SIMS
from carma.types import SimulatorError
from carma.types.core import Action, Observation, PhenologyStage, Pose


@SIMS.register("gazebo")
class GazeboAdapter:
    """Drives a paused Gazebo world one control period at a time.

    Args:
        control_hz: Control rate in hertz; each ``step`` advances ``1/control_hz``
            simulated seconds.
        instruction: Instruction handed to the backbone each step.
        stage: Name of the ``PhenologyStage`` the world was launched at. The
            world's stage is fixed at launch (``stage:=`` on the launch file);
            this only labels the observations, so it must match.
        timeout_s: Wall-clock seconds to wait for the bridge on one request.
            Software rendering needs generous values; see docs/simulation.md.
        bridge_url: Host the bridge listens on.
        bridge_port: TCP port the bridge listens on.
    """

    def __init__(
        self,
        control_hz: float = 10.0,
        instruction: str = "follow the row to the end and stop at the headland",
        stage: str = "VEGETATIVE",
        timeout_s: float = 60.0,
        bridge_url: str = "127.0.0.1",
        bridge_port: int = 5555,
    ) -> None:
        if control_hz <= 0.0:
            raise SimulatorError(f"control_hz must be positive, got {control_hz}")
        self._dt_s = 1.0 / control_hz
        self._instruction = instruction
        self._stage = PhenologyStage[stage]
        self._timeout_s = timeout_s
        self._bridge_url = bridge_url
        self._bridge_port = bridge_port
        self._socket: socket.socket | None = None

    def reset(self, *, seed: int, start: Pose) -> Observation:
        """Reset the world, place the robot at ``start``, and return the first observation."""
        request = {
            "op": "reset",
            "seed": int(seed),
            "start": {"x": float(start.x), "y": float(start.y), "yaw": float(start.yaw)},
        }
        return self._observation(self._request(request))

    def step(self, action: Action) -> Observation:
        """Hold ``action`` for one control period and return the observation after it."""
        if self._socket is None:
            raise SimulatorError("the simulator is not connected: call reset() first")
        reply = self._request(
            {
                "op": "step",
                "linear": float(action.linear),
                "angular": float(action.angular),
                "stop": bool(action.stop),
                "dt_s": self._dt_s,
            }
        )
        return self._observation(reply)

    def close(self) -> None:
        """Say goodbye to the bridge and release the socket."""
        if self._socket is None:
            return
        try:
            send_frame(self._socket, {"op": "close"})
        except OSError:
            pass  # the bridge is already gone; nothing is left to tell
        finally:
            self._socket.close()
            self._socket = None

    def _connect(self) -> socket.socket:
        try:
            sock = socket.create_connection(
                (self._bridge_url, self._bridge_port), timeout=self._timeout_s
            )
        except OSError as error:
            raise SimulatorError(
                f"cannot reach the Gazebo bridge at {self._bridge_url}:{self._bridge_port}: "
                f"{error}. Start it as described in docs/simulation.md."
            ) from error
        sock.settimeout(self._timeout_s)
        return sock

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            self._socket = self._connect()
        try:
            send_frame(self._socket, message)
            reply = recv_frame(self._socket)
        except (OSError, ValueError) as error:
            self.close()
            raise SimulatorError(
                f"the Gazebo bridge failed on {message['op']!r}: {error}"
            ) from error
        if not reply.get("ok", False):
            raise SimulatorError(
                f"the Gazebo bridge refused {message['op']!r}: {reply.get('error')}"
            )
        return reply

    def _observation(self, reply: dict[str, Any]) -> Observation:
        height, width = int(reply["height"]), int(reply["width"])
        png = np.frombuffer(base64.b64decode(reply["rgb_png"]), dtype=np.uint8)
        decoded = cv2.imdecode(png, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (height, width, 3):
            raise SimulatorError("the bridge sent an image that does not match its stated size")
        rgb = decoded.astype(np.uint8, copy=False)
        depth_bytes = base64.b64decode(reply["depth_f32"])
        if len(depth_bytes) != height * width * 4:
            raise SimulatorError("the bridge sent a depth map that does not match its stated size")
        depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(height, width).copy()
        pose = reply["pose"]
        return Observation(
            rgb=rgb,
            depth=depth,
            pose=Pose(x=float(pose["x"]), y=float(pose["y"]), yaw=float(pose["yaw"])),
            t=float(reply["t"]),
            instruction=self._instruction,
            stage=self._stage,
        )
