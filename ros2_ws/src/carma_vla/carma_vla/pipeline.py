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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from carma.backbone.wire import ActRequest, ActResponse, encode_image
from carma.types.core import BgrImage

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
    }
