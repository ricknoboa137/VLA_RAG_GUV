"""Gazebo Harmonic adapter, driven over ROS 2.

The Python process here holds no ROS dependency of its own: the ROS 2 node
lives in ``ros2_ws/src/carma_bridge`` and exchanges messages with this adapter
over a local transport. Keeping ``rclpy`` out of the library is what lets the
unit suite run in a plain virtualenv with no ROS installation.

Bring-up, topic names and the world files are documented in
``docs/simulation.md``.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any

from carma.sim.registry import SIMS
from carma.types import SimulatorError
from carma.types.core import Action, Observation, PhenologyStage, Pose

# The contract requires BGR, uint8, HWC
import numpy as np


@SIMS.register("gazebo")
class GazeboAdapter:
    """Talks to a running Gazebo Harmonic instance through the ROS 2 bridge.

    Args:
        world: World file name under ``ros2_ws/src/carma_sim/worlds``.
        robot: Robot description name.
        control_hz: Rate at which actions are published.
        timeout_s: How long to wait for the bridge before failing.
        bridge_url: Hostname or IP of the bridge node.
        bridge_port: TCP port of the bridge node.
    """

    def __init__(
        self,
        world: str = "row_crop.sdf",
        robot: str = "carma_ugv",
        control_hz: float = 10.0,
        timeout_s: float = 30.0,
        bridge_url: str = "127.0.0.1",
        bridge_port: int = 5555,
    ) -> None:
        self._world = world
        self._robot = robot
        self._control_hz = control_hz
        self._timeout_s = timeout_s
        self._bridge_url = bridge_url
        self._bridge_port = bridge_port
        self._socket: socket.socket | None = None

    def _connect(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout_s)
        self._socket.connect((self._bridge_url, self._bridge_port))

    def reset(self, *, seed: int) -> Observation:
        """Reset the world and return the first observation."""
        if self._socket is None:
            self._connect()
        # Trigger reset service via bridge
        self._socket.sendall(json.dumps({"op": "reset", "seed": seed}).encode())
        return self._receive_observation()

    def step(self, action: Action) -> Observation:
        """Publish an action and return the next synchronised observation."""
        if self._socket is None:
            raise SimulatorError("Sim not connected. Call reset() first.")

        # Publish action
        payload = {
            "op": "step",
            "linear": action.linear,
            "angular": action.angular,
            "stop": action.stop,
        }
        self._socket.sendall(json.dumps(payload).encode())
        return self._receive_observation()

    def _receive_observation(self) -> Observation:
        """Receive serialized Observation from the bridge."""
        # Simple length-prefixed protocol: 4-byte length + JSON body
        header = self._socket.recv(4)
        if not header:
            raise SimulatorError("Bridge disconnected")
        length = struct.unpack(">I", header)[0]
        data = b""
        while len(data) < length:
            chunk = self._socket.recv(length - len(data))
            if not chunk:
                raise SimulatorError("Bridge disconnected during observation")
            data += chunk

        payload = json.loads(data.decode())

        # Decode image (base64 encoded in payload)
        rgb_data = base64.b64decode(payload["rgb"])
        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(
            (payload["height"], payload["width"], 3)
        )

        depth = None
        if payload.get("depth"):
            depth_data = base64.b64decode(payload["depth"])
            depth = np.frombuffer(depth_data, dtype=np.float32).reshape(
                (payload["height"], payload["width"])
            )

        pose = Pose(
            x=payload["pose"]["x"],
            y=payload["pose"]["y"],
            yaw=payload["pose"]["yaw"],
        )

        return Observation(
            rgb=rgb,
            depth=depth,
            pose=pose,
            t=payload["t"],
            instruction=payload["instruction"],
            stage=PhenologyStage[payload["stage"]],
        )

    def close(self) -> None:
        """Shut the bridge down."""
        if self._socket:
            self._socket.close()
            self._socket = None