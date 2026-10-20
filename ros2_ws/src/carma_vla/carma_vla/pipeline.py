"""The ROS-free half of the live VLA pipeline: voice to instruction, frames to request.

Kept apart from the node so it can be tested without a ROS installation.

Voice. The speech node publishes one JSON result per utterance with an
``intent`` from :func:`carma.operator.speech.parse_utterance`. Here that intent
decides what the VLA is told:

========================  ==================================================
``stop``                  pause: no requests, and the node commands a halt
``go``                    resume with the instruction already in force
``turn_left/turn_right``  the words become the instruction, and resume
``correction``            the words become the instruction, and resume. Any
                          utterance longer than four words is a correction,
                          so "pick up the red bottle by the second row" lands
                          here
``yes``, ``no``, ``none``  ignored: answers, not instructions
rejected clips            ignored, whatever their text
========================  ==================================================

Cameras. The robot has two stereo cameras: ``world`` (the platform's view, the
one the headset shows) and ``arm`` (on the SO-101, for finding and approaching
targets). One of them is the *primary* and fills ``image`` / ``image_right``;
the other travels in ``views`` as ``<camera>_left`` and ``<camera>_right``.

Arm. Each request carries the arm's measured joint angles (``arm_state``), and
an arm policy answers with a chunk of joint targets. The loop is synchronous,
as LeRobot runs it on the real arm: the node observes only while the arm is
still, so the observation is still true when the answer arrives, and
:class:`ChunkPlayer` plays the whole chunk from that moment. (Timing rows from
the observation instead skips what the model's latency made stale; with a
1.8 s round trip and a 0.83 s chunk that left nothing to play.)
:func:`step_toward` then clamps each target to the joint limits and caps how
far one tick may move it, so a chunk that disagrees with where the arm is
cannot slam it across its range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from carma.backbone.wire import ActRequest, ActResponse, ArmChunk, encode_image
from carma.types.core import BgrImage

# The SO-101's joints, in the URDF's order.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

PAUSING = frozenset({"stop"})
RESUMING = frozenset({"go"})
INSTRUCTING = frozenset({"turn_left", "turn_right", "correction"})


@dataclass
class InstructionState:
    """What the operator has asked for, as of the latest utterance.

    Attributes:
        instruction: The instruction in force, or ``""`` before the first one.
        paused: True after a spoken stop, until go or a new instruction.
    """

    instruction: str = ""
    paused: bool = False

    def apply(self, utterance: dict[str, Any]) -> str | None:
        """Update from one speech result; return a one-line note when it changed anything."""
        if utterance.get("rejected"):
            return None
        intent = str(utterance.get("intent", "none"))
        text = str(utterance.get("text", "")).strip()
        if intent in PAUSING:
            self.paused = True
            return f"paused by {text!r}"
        if intent in RESUMING:
            if not self.instruction:
                return None
            self.paused = False
            return f"resumed: {self.instruction!r}"
        if intent in INSTRUCTING and text:
            self.instruction = text
            self.paused = False
            return f"instruction: {text!r}"
        return None

    @property
    def ready(self) -> bool:
        """True when there is an instruction and the operator has not paused."""
        return bool(self.instruction) and not self.paused


def parse_utterance_message(data: str) -> dict[str, Any] | None:
    """Decode the speech node's JSON, or None when it is not a JSON object."""
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def fit_width(image: BgrImage, width: int) -> BgrImage:
    """Scale ``image`` down to ``width`` pixels wide, keeping its aspect; never up."""
    if width <= 0 or image.shape[1] <= width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))


StereoPair = tuple[BgrImage, BgrImage | None]


def build_request(
    frames: dict[str, StereoPair],
    primary: str,
    instruction: str,
    *,
    pose: tuple[float, float, float],
    t: float,
    stage: str,
    seed: int,
    samples: int,
    width: int,
    arm_state: tuple[tuple[str, float], ...] = (),
) -> ActRequest:
    """One policy request from the newest frame of each camera.

    Args:
        frames: Newest ``(left, right)`` pair per camera name; right is None
            for a mono camera.
        primary: Camera whose pair fills ``image`` and ``image_right``.
        instruction: The operator's instruction in force.
        pose: ``(x_m, y_m, heading_rad)`` in the map frame.
        t: Seconds since the pipeline started.
        stage: Crop stage name.
        seed: Seed for this request; the server samples from it alone.
        samples: Action chunks the server should draw.
        width: Each frame is scaled down to this many pixels wide before
            encoding; 0 sends full resolution.
        arm_state: Measured arm joint angles, ``(joint, radians)``.

    Raises:
        KeyError: When the primary camera has no frame yet.
    """
    left, right = frames[primary]
    views: list[tuple[str, str]] = []
    for name in sorted(frames):
        if name == primary:
            continue
        other_left, other_right = frames[name]
        views.append((f"{name}_left", encode_image(fit_width(other_left, width))))
        if other_right is not None:
            views.append((f"{name}_right", encode_image(fit_width(other_right, width))))
    return ActRequest(
        image=encode_image(fit_width(left, width)),
        image_right=None if right is None else encode_image(fit_width(right, width)),
        instruction=instruction,
        pose=pose,
        t=t,
        stage=stage,
        seed=seed,
        samples=samples,
        views=tuple(views),
        arm_state=arm_state,
    )


def action_record(
    response: ActResponse, request: ActRequest, primary: str, latency_s: float
) -> dict[str, Any]:
    """What the node publishes for each answered request, as plain JSON."""
    return {
        "t": request.t,
        "seed": request.seed,
        "instruction": request.instruction,
        "primary": primary,
        "stereo": request.image_right is not None,
        "views": [name for name, _ in request.views],
        "linear": response.linear,
        "angular": response.angular,
        "stop": response.stop,
        "uncertainty": response.uncertainty,
        "uncertainty_method": response.uncertainty_method,
        "model_id": response.model_id,
        "model_revision": response.model_revision,
        "inference_s": response.inference_s,
        "round_trip_s": round(latency_s, 4),
        "info": response.info,
        "arm": None
        if response.arm is None
        else {
            "steps": len(response.arm.positions_rad),
            "dt_s": response.arm.dt_s,
            "first": dict(zip(response.arm.joints, response.arm.positions_rad[0], strict=True)),
        },
    }


def arm_state_from_joint_state(
    names: list[str], positions: list[float], joints: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    """The named joints' angles from a ``JointState``, in ``joints`` order.

    Returns an empty tuple when any of them is missing, because a partial
    state would have to be completed with invented angles.
    """
    measured = dict(zip(names, positions, strict=False))
    if any(name not in measured for name in joints):
        return ()
    return tuple((name, float(measured[name])) for name in joints)


@dataclass
class ChunkPlayer:
    """Plays an arm chunk out against the clock.

    Attributes:
        chunk: The chunk in force, or None.
        t0: Monotonic time, in seconds, at which row 0 is due; row ``i`` is
            due at ``t0 + i * dt_s``.
    """

    chunk: ArmChunk | None = None
    t0: float = 0.0

    def load(self, chunk: ArmChunk, t0: float) -> None:
        """Replace the chunk in force; a later start always wins."""
        if self.chunk is None or t0 >= self.t0:
            self.chunk, self.t0 = chunk, t0

    def clear(self) -> None:
        """Drop the chunk, so nothing more is commanded."""
        self.chunk = None

    def target(self, now: float) -> dict[str, float] | None:
        """The row due at ``now``, or None before a chunk or after its last row.

        After the last row nothing is commanded and the arm holds its last
        target: a chunk is as far ahead as the model was trained to act
        without looking again.
        """
        if self.chunk is None:
            return None
        index = max(0, int((now - self.t0) / self.chunk.dt_s))
        if index >= len(self.chunk.positions_rad):
            return None
        return dict(zip(self.chunk.joints, self.chunk.positions_rad[index], strict=True))


def step_toward(
    target: dict[str, float],
    previous: dict[str, float],
    max_step_rad: float,
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """One tick's command: ``target``, clamped to the limits and to ``max_step_rad`` per tick.

    Joints in ``target`` without a limit or a previous value are left out:
    commanding a joint whose range or position is unknown is not safe.
    """
    command: dict[str, float] = {}
    for name, goal in target.items():
        if name not in limits or name not in previous:
            continue
        low, high = limits[name]
        goal = min(max(goal, low), high)
        start = previous[name]
        command[name] = start + min(max(goal - start, -max_step_rad), max_step_rad)
    return command
