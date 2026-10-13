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
"""

from __future__ import annotations

import base64
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

    def to_json(self) -> dict[str, Any]:
        """A plain dict ready for ``json.dumps``."""
        payload = asdict(self)
        payload["context"] = list(self.context)
        payload["pose"] = list(self.pose)
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

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            msg = f"uncertainty must be in [0, 1], got {self.uncertainty}"
            raise ValueError(msg)

    def to_json(self) -> dict[str, Any]:
        """A plain dict ready for ``json.dumps``."""
        return asdict(self)

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
        )
