#!/usr/bin/env python3
"""Drive the lockstep Gazebo platform by keyboard and by voice, with the camera view.

    python scripts/drive_sim.py                    # keyboard + speech from the MQTT broker
    python scripts/drive_sim.py --no-voice         # keyboard only

Keys (click the window first): W/S forward/back, A/D turn, Q/E forward-and-turn,
Space or X stop, R reset to the start pose, Esc quit. Hold a key to keep going.

Speech arrives as ``carma_voice`` utterances on ``carma/operator/utterance``
(start ``scripts/mic_mqtt.py`` to feed it): "go" latches forward, "turn left /
right" is a timed turn, "stop" clears everything. Anything longer (a correction
such as "keep to the left of the row") does not drive; it is shown on screen as
a suggestion and appended to ``--log``.

The bridge serves one client: stop other recordings or runs first. Needs the
world launched with ``lockstep:=true`` and ``carma_bridge`` running
(``docs/simulation.md``). Time advances only while this script steps it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from carma.sim.gazebo import GazeboAdapter
from carma.sim.teleop import DriveController
from carma.types.core import IntentKind, Observation, Pose

WINDOW = "CARMA platform view"


class _Shared:
    """State shared between the stepping thread and the window loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.obs: Observation | None = None
        self.heard = ""
        self.error: str | None = None
        self.reset = False
        self.stop = False


def _voice(args: argparse.Namespace, ctrl: DriveController, shared: _Shared) -> object | None:
    import paho.mqtt.client as mqtt

    def on_message(_c: object, _u: object, message: mqtt.MQTTMessage) -> None:
        try:
            data = json.loads(message.payload)
            kind = IntentKind(data.get("intent", "none"))
        except ValueError:
            return
        text = str(data.get("text", "")).strip()
        if data.get("rejected") or not text:
            return
        drove = ctrl.intent(kind)
        with shared.lock:
            shared.heard = f'"{text}" -> {kind.value}' + ("" if drove else " (suggestion)")
        if not drove and kind is IntentKind.CORRECTION and args.log:
            with args.log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"text": text, "wall_time": time.time()}) + "\n")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="carma-drive-sim")
    client.on_message = on_message
    client.on_connect = lambda c, *_a: c.subscribe(args.topic, qos=1)
    try:
        client.connect(args.broker, args.port, keepalive=60)
    except OSError as error:
        print(f"no speech channel ({args.broker}:{args.port}: {error}); keyboard only")
        return None
    client.loop_start()
    return client


def _sim_loop(sim: GazeboAdapter, ctrl: DriveController, shared: _Shared, start: Pose) -> None:
    try:
        obs = sim.reset(seed=0, start=start)
        while not shared.stop:
            with shared.lock:
                shared.obs = obs
                do_reset, shared.reset = shared.reset, False
            if do_reset:
                ctrl.key("x")
                obs = sim.reset(seed=0, start=start)
                continue
            obs = sim.step(ctrl.next_action())
    except Exception as error:
        with shared.lock:
            shared.error = f"{type(error).__name__}: {error}"


def _draw(obs: Observation, ctrl: DriveController, heard: str) -> np.ndarray:
    frame = cv2.resize(obs.rgb, (960, int(obs.rgb.shape[0] * 960 / obs.rgb.shape[1])))
    status = ctrl.status()
    lines = [
        f"x {obs.pose.x:6.2f}  y {obs.pose.y:5.2f}  yaw {obs.pose.yaw:5.2f}  t {obs.t:6.1f} s",
        f"command: {status.source}{'  [go latched]' if status.latched_go else ''}",
        f"heard: {heard}" if heard else "heard: -",
    ]
    for row, text in enumerate(lines):
        y = 24 + 24 * row
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def main() -> int:
    """Run the drive session; the return value is the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--stage", default="VEGETATIVE")
    parser.add_argument("--bridge-url", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5555)
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="carma/operator/utterance")
    parser.add_argument("--log", type=Path, default=Path("assets/drive_suggestions.jsonl"))
    args = parser.parse_args()

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    ctrl = DriveController()
    shared = _Shared()
    sim = GazeboAdapter(
        control_hz=args.control_hz,
        stage=args.stage,
        timeout_s=300.0,
        bridge_url=args.bridge_url,
        bridge_port=args.bridge_port,
    )
    client = None if args.no_voice else _voice(args, ctrl, shared)
    worker = threading.Thread(
        target=_sim_loop, args=(sim, ctrl, shared, Pose(0.0, 0.0, 0.0)), daemon=True
    )
    worker.start()
    code = 0
    try:
        while True:
            with shared.lock:
                obs, heard, error = shared.obs, shared.heard, shared.error
            if error:
                print(error, file=sys.stderr)
                code = 1
                break
            if obs is not None:
                cv2.imshow(WINDOW, _draw(obs, ctrl, heard))
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                with shared.lock:
                    shared.reset = True
            elif key != 255:
                ctrl.key(chr(key))
    finally:
        shared.stop = True
        worker.join(timeout=30.0)
        sim.close()
        if client is not None:
            client.loop_stop()  # type: ignore[attr-defined]
        cv2.destroyAllWindows()
    return code


if __name__ == "__main__":
    sys.exit(main())
