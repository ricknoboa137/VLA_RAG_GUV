"""The request and reply exchanged with a policy server.

Both ends import these dataclasses, so the contract cannot drift between the
client in :mod:`carma.backbone.remote` and whatever serves it. Changing a field
here is a breaking change and must land with both ends updated, exactly as the
ROS topic contract in ``ros2_ws/src/carma_bridge`` works.

Images cross as base64 PNG rather than raw arrays: PNG is lossless, so the
server sees the pixels the policy saw, and base64 survives JSON without a
second serialisation format on either side. A 256x256 frame is about 90 kB
encoded, which is nothing beside the inference it triggers.

The seed travels with every request. A run is ``(config, seed, commit)`` and a
flow-matching policy samples, so a server that draws its own noise would make
the same run give different numbers and break the determinism gate. The server
is required to seed from this field and to report, in
:class:`ActResponse.model_id` and :attr:`ActResponse.model_revision`, exactly
what answered -- those go into the run manifest, where a reader can see which
weights produced a result.

A manipulator policy also reads the arm's joint angles (``ActRequest.arm_state``)
and answers with a chunk of joint targets (``ActResponse.arm``). Both are
optional and default to empty, so a navigation client and server never see
them. Angles cross in radians, in the arm's URDF convention; converting from
a model's own units is the server's job, so the robot side never needs to know
which model answered.
"""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from carma.types.core import BgrImage

_PNG = ".png"


def encode_image(image: BgrImage) -> str:
    """BGR uint8 array to a base64 PNG string."""
    ok, buffer = cv2.imencode(_PNG, image)
    if not ok:
        msg = "PNG encoding failed"
        raise ValueError(msg)
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image(encoded: str) -> BgrImage:
    """Base64 PNG string back to a BGR uint8 array."""
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        msg = "PNG decoding failed"  # type: ignore[unreachable]
        raise ValueError(msg)
    return cast("BgrImage", image)


def encode_png16(image: npt.NDArray[np.uint16]) -> str:
    """Single-channel uint16 array to a base64 PNG string.

    PNG carries 16-bit greyscale natively, so depth in millimetres crosses
    without being squeezed into 8 bits or sent as raw floats.
    """
    if image.dtype != np.uint16:
        msg = f"expected uint16, got {image.dtype}"
        raise ValueError(msg)
    ok, buffer = cv2.imencode(_PNG, image)
    if not ok:
        msg = "16-bit PNG encoding failed"
        raise ValueError(msg)
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_png16(encoded: str) -> npt.NDArray[np.uint16]:
    """Base64 PNG string back to a single-channel uint16 array."""
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = "16-bit PNG decoding failed"  # type: ignore[unreachable]
        raise ValueError(msg)
    return cast("npt.NDArray[np.uint16]", image.astype(np.uint16))


@dataclass(frozen=True)
class ActRequest:
    """One observation, as the policy server receives it.

    Attributes:
        image: Left camera frame, base64 PNG, BGR.
        image_right: Right frame of a stereo pair, or None when monocular.
        depth_mm: Depth as base64 PNG, 16-bit millimetres, or None. The
            RGB-D convention used elsewhere in this project: 0 means no
            reading, which is distinct from zero distance.
        instruction: The navigation instruction in force.
        context: Retrieved operator corrections, most relevant first. A policy
            that ignores them must still accept them, so that conditions
            differ only by config.
        pose: Robot pose in the map frame, ``(x_m, y_m, heading_rad)``.
        t: Seconds since episode start.
        stage: Crop stage name at capture time.
        seed: Seed for this step. The server samples from it and nothing else.
        samples: How many action chunks to draw. Their spread becomes the
            uncertainty, so one sample means the server must say so rather
            than invent a confident number.
        views: Further camera frames, as ``(name, base64 PNG BGR)`` pairs in a
            fixed order, for a robot with more than one camera. ``image`` and
            ``image_right`` are the primary camera; a second stereo camera
            arrives here as ``<camera>_left`` and ``<camera>_right``, for
            example ``world_left``. A policy that uses one view must still
            accept the others. On the wire this is a JSON object, whose key
            order is kept.
        arm_state: Measured arm joint angles as ``(joint, radians)`` pairs, in
            the arm's URDF convention, or empty for a robot without an arm. A
            manipulator policy conditions on these; a navigation policy
            ignores them. On the wire, a JSON object.
    """

    image: str
    instruction: str
    pose: tuple[float, float, float]
    t: float
    stage: str
    seed: int
    samples: int = 4
    context: tuple[str, ...] = ()
    image_right: str | None = None
    depth_mm: str | None = None
    views: tuple[tuple[str, str], ...] = ()
    arm_state: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.views]
        if len(set(names)) != len(names):
            msg = f"view names must be unique, got {names}"
            raise ValueError(msg)
        joints = [name for name, _ in self.arm_state]
        if len(set(joints)) != len(joints):
            msg = f"arm joint names must be unique, got {joints}"
            raise ValueError(msg)
        if not all(math.isfinite(angle) for _, angle in self.arm_state):
            msg = f"arm joint angles must be finite, got {self.arm_state}"
            raise ValueError(msg)

    def to_json(self) -> dict[str, Any]:
        """A plain dict ready for ``json.dumps``."""
        payload = asdict(self)
        payload["context"] = list(self.context)
        payload["pose"] = list(self.pose)
        payload["views"] = dict(self.views)
        payload["arm_state"] = dict(self.arm_state)
        return payload

    @staticmethod
    def from_json(payload: dict[str, Any]) -> ActRequest:
        """Rebuild a request, rejecting one that is missing a required field."""
        required = ("image", "instruction", "pose", "t", "stage", "seed")
        missing = [key for key in required if key not in payload]
        if missing:
            msg = f"request is missing {', '.join(missing)}"
            raise ValueError(msg)
        pose = tuple(float(v) for v in payload["pose"])
        if len(pose) != 3:
            msg = f"pose must be (x, y, heading), got {len(pose)} values"
            raise ValueError(msg)
        return ActRequest(
            image=str(payload["image"]),
            instruction=str(payload["instruction"]),
            pose=(pose[0], pose[1], pose[2]),
            t=float(payload["t"]),
            stage=str(payload["stage"]),
            seed=int(payload["seed"]),
            samples=int(payload.get("samples", 4)),
            context=tuple(str(c) for c in payload.get("context", ())),
            image_right=payload.get("image_right"),
            depth_mm=payload.get("depth_mm"),
            views=tuple((str(k), str(v)) for k, v in dict(payload.get("views", {})).items()),
            arm_state=tuple(
                (str(k), float(v)) for k, v in dict(payload.get("arm_state", {})).items()
            ),
        )


@dataclass(frozen=True)
class ArmChunk:
    """A chunk of arm joint targets, as a manipulator policy returns it.

    VLAs for arms predict a short trajectory rather than one command: SmolVLA
    50 steps, X-VLA 30, at the recording rate of their training data. The
    robot plays the chunk out at ``dt_s`` until the next one arrives.

    Attributes:
        joints: Joint names, in the order of each row of ``positions_rad``.
        positions_rad: Joint position targets in radians, in the arm's URDF
            convention, one row per step; row 0 is the first to execute.
        dt_s: Seconds between consecutive rows.
    """

    joints: tuple[str, ...]
    positions_rad: tuple[tuple[float, ...], ...]
    dt_s: float

    def __post_init__(self) -> None:
        if not self.joints or len(set(self.joints)) != len(self.joints):
            msg = f"joint names must be non-empty and unique, got {self.joints}"
            raise ValueError(msg)
        if not self.positions_rad:
            msg = "an arm chunk needs at least one step"
            raise ValueError(msg)
        for i, row in enumerate(self.positions_rad):
            if len(row) != len(self.joints):
                msg = f"step {i} has {len(row)} values for {len(self.joints)} joints"
                raise ValueError(msg)
            if not all(math.isfinite(v) for v in row):
                msg = f"step {i} has a non-finite joint target: {row}"
                raise ValueError(msg)
        if not (math.isfinite(self.dt_s) and self.dt_s > 0.0):
            msg = f"dt_s must be positive, got {self.dt_s}"
            raise ValueError(msg)

    def to_json(self) -> dict[str, Any]:
        """A plain dict ready for ``json.dumps``."""
        return {
            "joints": list(self.joints),
            "positions_rad": [list(row) for row in self.positions_rad],
            "dt_s": self.dt_s,
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> ArmChunk:
        """Rebuild a chunk, rejecting one that is missing a field."""
        missing = [key for key in ("joints", "positions_rad", "dt_s") if key not in payload]
        if missing:
            msg = f"arm chunk is missing {', '.join(missing)}"
            raise ValueError(msg)
        return ArmChunk(
            joints=tuple(str(j) for j in payload["joints"]),
            positions_rad=tuple(tuple(float(v) for v in row) for row in payload["positions_rad"]),
            dt_s=float(payload["dt_s"]),
        )


@dataclass(frozen=True)
class ActResponse:
    """One action, as the policy server returns it.

    Attributes:
        linear: Forward velocity in metres per second.
        angular: Yaw rate in radians per second, positive counter-clockwise.
        stop: True when the policy declares the goal reached.
        uncertainty: In ``[0, 1]``. How it was obtained belongs in
            ``uncertainty_method``, because a number with no stated origin
            cannot be calibrated or argued with.
        uncertainty_method: Short key naming the estimator, for example
            ``chunk_spread`` or ``none``.
        model_id: What answered, for example ``2toINF/X-VLA-WidowX``. Recorded
            in the run manifest.
        model_revision: Commit or revision of those weights, or None when the
            server cannot determine it.
        inference_s: Server-side inference time in seconds, measured.
        info: Anything else the server wants to record. Never read by the
            arbiter; it exists so a server can explain itself in the log.
        arm: Joint targets for the arm, from a manipulator policy, or None.
            Such a policy does not drive the base and answers ``linear`` and
            ``angular`` of zero, so a client that only drives never moves.
            Never read by the arbiter: the arm is outside the arbitration.
    """

    linear: float
    angular: float
    stop: bool
    uncertainty: float
    uncertainty_method: str
    model_id: str
    inference_s: float
    model_revision: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    arm: ArmChunk | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            msg = f"uncertainty must be in [0, 1], got {self.uncertainty}"
            raise ValueError(msg)

    def to_json(self) -> dict[str, Any]:
        """A plain dict ready for ``json.dumps``."""
        payload = asdict(self)
        payload["arm"] = None if self.arm is None else self.arm.to_json()
        return payload

    @staticmethod
    def from_json(payload: dict[str, Any]) -> ActResponse:
        """Rebuild a response, rejecting one that is missing a required field."""
        required = (
            "linear",
            "angular",
            "stop",
            "uncertainty",
            "uncertainty_method",
            "model_id",
            "inference_s",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            msg = f"response is missing {', '.join(missing)}"
            raise ValueError(msg)
        return ActResponse(
            linear=float(payload["linear"]),
            angular=float(payload["angular"]),
            stop=bool(payload["stop"]),
            uncertainty=float(payload["uncertainty"]),
            uncertainty_method=str(payload["uncertainty_method"]),
            model_id=str(payload["model_id"]),
            inference_s=float(payload["inference_s"]),
            model_revision=payload.get("model_revision"),
            info=dict(payload.get("info", {})),
            arm=None if payload.get("arm") is None else ArmChunk.from_json(dict(payload["arm"])),
        )
