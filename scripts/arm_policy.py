"""SO-101 arm policies -- SmolVLA and X-VLA through LeRobot -- behind the act contract.

Imported by ``scripts/policy_server.py`` for ``--policy smolvla`` and
``--policy xvla``. Everything a run depends on is in one spec file under
``configs/arm_policy/``: which checkpoint, at which revision, which robot camera
feeds which of the model's camera slots, how the model's joint units map to the
arm's URDF radians, and how sample spread becomes an uncertainty. The two
models are compared by swapping the spec, never the code.

torch and LeRobot are imported only when a policy is built, so the pure parts
here -- units, spread, uncertainty, observation assembly -- are tested in the
``carma`` environment without either. The model itself runs in the
``carma-train`` environment (``training/README.md``), offline: the spec names a
local checkpoint fetched by ``scripts/fetch_training_assets.py``, and the Hub
is switched off before LeRobot is imported.

Determinism. The request's seed seeds torch before each call, and all
``samples`` chunks are drawn in one batch, so the same request gives the same
chunk. X-VLA's LeRobot port ignores an explicit noise tensor and draws from
torch's global generator; seeding that generator is the one mechanism that
works for both models.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import yaml

from carma.backbone.wire import ActRequest, ActResponse, ArmChunk, decode_image

KINDS = ("smolvla", "xvla")
# Where a camera slot can be filled from: the request's primary pair, or a named view.
PRIMARY_SOURCES = ("image", "image_right")

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class JointUnits:
    """A linear map between one joint's model units and URDF radians.

    Given as two corresponding points, because that is how the units are
    defined: LeRobot's ``range_m100_100`` puts -100 and +100 at the ends of the
    calibrated range, ``range_0_100`` does the same for the gripper, and
    ``degrees`` is a scale with an offset.

    Attributes:
        name: URDF joint name.
        model: Two values in the model's units.
        rad: The joint angles, in radians, those two values mean.
    """

    name: str
    model: tuple[float, float]
    rad: tuple[float, float]

    def __post_init__(self) -> None:
        if self.model[0] == self.model[1] or self.rad[0] == self.rad[1]:
            msg = f"joint {self.name}: the two points of a unit map must differ"
            raise ValueError(msg)

    @property
    def range_rad(self) -> float:
        """Width of the mapped range in radians, the scale spread is measured against."""
        return abs(self.rad[1] - self.rad[0])

    def to_rad(self, value: FloatArray) -> FloatArray:
        """Model units to radians."""
        m0, m1 = self.model
        r0, r1 = self.rad
        return r0 + (np.asarray(value, dtype=np.float64) - m0) * (r1 - r0) / (m1 - m0)

    def to_model(self, angle_rad: FloatArray) -> FloatArray:
        """Radians to model units."""
        m0, m1 = self.model
        r0, r1 = self.rad
        return m0 + (np.asarray(angle_rad, dtype=np.float64) - r0) * (m1 - m0) / (r1 - r0)


@dataclass(frozen=True)
class UncertaintyMap:
    """Turns the spread of sampled chunks into an uncertainty in [0, 1].

    ``uncalibrated``: ``u = s / (s + scale)``. Monotone and bounded, but its
    numbers are not probabilities, and they are not comparable between models:
    X-VLA's samples spread far less than SmolVLA's for the same observation.
    The method key says so, so nothing downstream mistakes it for calibrated.

    ``logistic``: ``u = sigmoid(a * ln(s) + b)``, fitted on held-out frames by
    ``scripts/fit_arm_uncertainty.py`` (:func:`fit_logistic`) to predict whether
    the chunk's mean joint error exceeds ``tolerance_rad``. Calibrated against
    that event only, and only for the model it was fitted on.

    Attributes:
        method: ``uncalibrated`` or ``logistic``.
        scale: Spread at which the uncalibrated map gives 0.5.
        a: Logistic slope on ln(spread).
        b: Logistic intercept.
        tolerance_rad: The error the logistic map predicts exceeding, in radians.
        source: Where the fit came from, recorded with every reply.
    """

    method: str = "uncalibrated"
    scale: float = 0.02
    a: float = 0.0
    b: float = 0.0
    tolerance_rad: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.method not in ("uncalibrated", "logistic"):
            msg = f"unknown uncertainty method {self.method!r}"
            raise ValueError(msg)
        if self.method == "uncalibrated" and self.scale <= 0.0:
            msg = f"scale must be positive, got {self.scale}"
            raise ValueError(msg)

    @property
    def key(self) -> str:
        """The value of ``ActResponse.uncertainty_method``."""
        return f"chunk_spread/{self.method}"

    def __call__(self, spread: float) -> float:
        """Uncertainty for a spread, in [0, 1]."""
        if not math.isfinite(spread) or spread < 0.0:
            return 1.0
        if self.method == "uncalibrated":
            return spread / (spread + self.scale)
        if spread == 0.0:
            return 0.0
        z = self.a * math.log(spread) + self.b
        return 1.0 / (1.0 + math.exp(-z)) if z > -700.0 else 0.0


@dataclass(frozen=True)
class ArmPolicySpec:
    """Everything that defines an arm policy server, read from one YAML file.

    Attributes:
        kind: ``smolvla`` or ``xvla``.
        checkpoint: Local checkpoint directory, relative to the repository root.
        model_id: The Hub repository the weights came from.
        model_revision: The commit fetched, as pinned in the fetch script.
        dtype: ``bfloat16`` or ``float32``, the precision the model runs in.
        fps: Rate of the training data, so the spacing of chunk rows is
            ``1 / fps`` seconds.
        horizon: How many rows of each chunk to send, at most the model's
            chunk size.
        cameras: Model image feature to source: ``image``, ``image_right``, or
            the name of a request view such as ``world_left``. A source the
            request lacks leaves the slot out, which both models handle by
            masking it.
        joints: Unit maps, in the order of the model's action and state vector.
        uncertainty: The spread-to-uncertainty map.
    """

    kind: str
    checkpoint: Path
    model_id: str
    model_revision: str
    dtype: str
    fps: float
    horizon: int
    cameras: tuple[tuple[str, str], ...]
    joints: tuple[JointUnits, ...]
    uncertainty: UncertaintyMap

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            msg = f"kind must be one of {KINDS}, got {self.kind!r}"
            raise ValueError(msg)
        if self.dtype not in ("bfloat16", "float32"):
            msg = f"dtype must be bfloat16 or float32, got {self.dtype!r}"
            raise ValueError(msg)
        if self.fps <= 0.0 or self.horizon < 1:
            msg = f"fps must be positive and horizon at least 1, got {self.fps}, {self.horizon}"
            raise ValueError(msg)
        if not self.cameras or not self.joints:
            msg = "an arm policy needs at least one camera slot and one joint"
            raise ValueError(msg)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """URDF joint names, in model order."""
        return tuple(j.name for j in self.joints)

    def state_in_model_units(self, request: ActRequest) -> FloatArray:
        """The request's arm state as the model's state vector.

        Raises:
            ValueError: When the request lacks a joint the model reads. A
                missing joint must not be silently filled with zero, which is a
                real pose.
        """
        state = dict(request.arm_state)
        missing = [name for name in self.joint_names if name not in state]
        if missing:
            msg = f"request arm_state lacks joints {missing}"
            raise ValueError(msg)
        return np.array(
            [float(j.to_model(np.float64(state[j.name]))) for j in self.joints], dtype=np.float64
        )

    def chunk_to_rad(self, chunk_model: FloatArray) -> FloatArray:
        """``(..., joints)`` in model units to radians."""
        out = np.empty(chunk_model.shape, dtype=np.float64)
        for i, joint in enumerate(self.joints):
            out[..., i] = joint.to_rad(chunk_model[..., i])
        return out


def load_spec(path: Path) -> ArmPolicySpec:
    """Read an arm policy spec from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    joints = tuple(
        JointUnits(
            name=str(name),
            model=(float(entry["model"][0]), float(entry["model"][1])),
            rad=(float(entry["rad"][0]), float(entry["rad"][1])),
        )
        for name, entry in raw["joints"].items()
    )
    return ArmPolicySpec(
        kind=str(raw["kind"]),
        checkpoint=Path(raw["checkpoint"]),
        model_id=str(raw["model_id"]),
        model_revision=str(raw["model_revision"]),
        dtype=str(raw.get("dtype", "bfloat16")),
        fps=float(raw["fps"]),
        horizon=int(raw["horizon"]),
        cameras=tuple((str(k), str(v)) for k, v in raw["cameras"].items()),
        joints=joints,
        uncertainty=UncertaintyMap(**raw.get("uncertainty", {})),
    )


def fit_logistic(
    spreads: FloatArray, exceeded: npt.NDArray[np.bool_], ridge: float = 1e-3, iterations: int = 50
) -> tuple[float, float]:
    """Fit ``P(exceeded) = sigmoid(a * ln(spread) + b)`` by maximum likelihood.

    Newton's method (iteratively reweighted least squares) with a small ridge
    penalty on ``a``, so a perfectly separable calibration set still gives a
    finite slope. Deterministic: no randomness, fixed iteration count with an
    early stop on convergence.

    Args:
        spreads: ``(n,)`` chunk spreads, each positive.
        exceeded: ``(n,)`` whether that chunk's error exceeded the tolerance.
        ridge: L2 penalty on the slope.
        iterations: Most Newton steps taken.

    Returns:
        ``(a, b)``.
    """
    s = np.asarray(spreads, dtype=np.float64)
    y = np.asarray(exceeded, dtype=np.float64)
    if s.ndim != 1 or s.shape != y.shape or len(s) < 2:
        msg = f"need two equal 1-D samples of length >= 2, got {s.shape} and {y.shape}"
        raise ValueError(msg)
    if not np.all(s > 0.0) or not np.all(np.isfinite(s)):
        msg = "every spread must be positive and finite"
        raise ValueError(msg)
    x = np.stack([np.log(s), np.ones_like(s)], axis=1)
    penalty = np.diag([ridge, 0.0])
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -500.0, 500.0)))
        gradient = x.T @ (p - y) + penalty @ w
        hessian = (x * (p * (1.0 - p))[:, None]).T @ x + penalty + 1e-12 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def chunk_spread(chunks_rad: FloatArray, ranges_rad: FloatArray) -> float:
    """How much sampled chunks disagree, as a fraction of each joint's range.

    The standard deviation across samples at each step and joint, divided by
    that joint's range, averaged over steps and joints. Dimensionless, so a
    gripper and a shoulder count alike.

    Args:
        chunks_rad: ``(samples, steps, joints)`` in radians.
        ranges_rad: ``(joints,)`` range of each joint in radians.

    Returns:
        The mean normalised spread; 0.0 for a single sample, which cannot
        disagree with itself -- the caller must then not claim a sampled
        estimate.
    """
    if chunks_rad.ndim != 3:
        msg = f"chunks must be (samples, steps, joints), got shape {chunks_rad.shape}"
        raise ValueError(msg)
    if chunks_rad.shape[0] < 2:
        return 0.0
    std = chunks_rad.std(axis=0)
    return float((std / ranges_rad).mean())


def camera_frames(
    request: ActRequest, cameras: tuple[tuple[str, str], ...]
) -> tuple[dict[str, npt.NDArray[np.uint8]], list[str]]:
    """Decode the frames each camera slot is fed from, as BGR.

    Returns:
        ``(frames, missing)``: frames by model feature name, and the feature
        names whose source the request did not carry.
    """
    views = dict(request.views)
    frames: dict[str, npt.NDArray[np.uint8]] = {}
    missing: list[str] = []
    for feature, source in cameras:
        if source == "image":
            encoded: str | None = request.image
        elif source == "image_right":
            encoded = request.image_right
        else:
            encoded = views.get(source)
        if encoded is None:
            missing.append(feature)
            continue
        frames[feature] = decode_image(encoded)
    return frames, missing


def to_model_rgb(
    image_bgr: npt.NDArray[np.uint8], height: int, width: int
) -> npt.NDArray[np.uint8]:
    """A BGR frame as the HWC RGB uint8 array LeRobot expects, at the feature's size.

    The conversion to RGB happens here, at the boundary of the module that
    needs it (AGENTS.md section 7).
    """
    if image_bgr.shape[:2] != (height, width):
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


class LeRobotArmPolicy:
    """SmolVLA or X-VLA, loaded through LeRobot, answering with arm chunks.

    Does not drive the base: every reply has ``linear = angular = 0``, and an
    arm chunk in ``arm``.
    """

    def __init__(self, spec: ArmPolicySpec, device: str = "cuda") -> None:
        # Before LeRobot or transformers is imported: no network at inference.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from lerobot.policies.factory import make_pre_post_processors

        if not spec.checkpoint.is_dir():
            msg = (
                f"{spec.checkpoint} not found; fetch it with scripts/fetch_training_assets.py "
                "and run from the repository root"
            )
            raise FileNotFoundError(msg)
        policy_class: Any
        if spec.kind == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            policy_class = SmolVLAPolicy
        else:
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

            policy_class = XVLAPolicy

        self._torch = torch
        self._spec = spec
        self._device = torch.device(device)
        policy = policy_class.from_pretrained(str(spec.checkpoint)).to(self._device).eval()
        if spec.dtype == "bfloat16":
            # SmolVLA has no dtype field; its VLM and action expert are cast, the
            # small projections stay fp32, the layout smolvla_base ships in.
            # X-VLA casts its whole model.
            target = policy.model.vlm_with_expert if spec.kind == "smolvla" else policy
            target.to(torch.bfloat16)
        self._policy = policy
        self._pre, self._post = make_pre_post_processors(
            policy.config,
            str(spec.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self._device)}},
        )
        features = policy.config.input_features
        self._image_shapes = {
            name: (int(f.shape[1]), int(f.shape[2]))
            for name, f in features.items()
            if f.type.name == "VISUAL"
        }
        state = [f for name, f in features.items() if f.type.name == "STATE"]
        self._state_dim = int(state[0].shape[0]) if state else 0
        self._action_dim = int(policy.config.output_features["action"].shape[0])
        if self._action_dim != len(spec.joints):
            msg = f"model predicts {self._action_dim} values, spec maps {len(spec.joints)} joints"
            raise ValueError(msg)
        unknown = [name for name, _ in spec.cameras if name not in self._image_shapes]
        if unknown:
            msg = (
                f"spec maps cameras the model lacks: {unknown}; it has {sorted(self._image_shapes)}"
            )
            raise ValueError(msg)
        self._chunk_size = int(policy.config.chunk_size)
        self._ranges = np.array([j.range_rad for j in spec.joints], dtype=np.float64)

    @property
    def model_id(self) -> str:
        """The Hub repository the weights came from."""
        return self._spec.model_id

    def act(self, request: ActRequest) -> ActResponse:
        """Draw ``request.samples`` chunks, answer with the first and their spread."""
        from lerobot.policies.utils import prepare_observation_for_inference

        torch = self._torch
        started = time.monotonic()
        frames, missing = camera_frames(request, self._spec.cameras)
        if not frames:
            msg = "none of the model's cameras is in the request"
            raise ValueError(msg)
        observation: dict[str, Any] = {
            name: to_model_rgb(frame, *self._image_shapes[name]) for name, frame in frames.items()
        }
        state = np.zeros(self._state_dim, dtype=np.float32)
        state[: len(self._spec.joints)] = self._spec.state_in_model_units(request)
        observation["observation.state"] = state

        samples = max(1, request.samples)
        batch = self._pre(
            prepare_observation_for_inference(observation, self._device, task=request.instruction)
        )
        batch = {
            key: value.expand(samples, *value.shape[1:]).contiguous()
            if isinstance(value, torch.Tensor) and value.shape[:1] == (1,)
            else value
            for key, value in batch.items()
        }
        self._policy.reset()
        torch.manual_seed(request.seed)
        with torch.inference_mode():
            raw = self._policy.predict_action_chunk(batch)[..., : self._action_dim]
            chunks = self._post(raw).float().cpu().numpy().astype(np.float64)

        chunks_rad = self._spec.chunk_to_rad(chunks)
        spread = chunk_spread(chunks_rad, self._ranges)
        horizon = min(self._spec.horizon, chunks_rad.shape[1])
        uncertainty_map = self._spec.uncertainty
        return ActResponse(
            linear=0.0,
            angular=0.0,
            stop=False,
            # One sample cannot disagree with itself: say so rather than claim 0.
            uncertainty=uncertainty_map(spread) if samples > 1 else 1.0,
            uncertainty_method=uncertainty_map.key if samples > 1 else "none",
            model_id=self._spec.model_id,
            model_revision=self._spec.model_revision,
            inference_s=time.monotonic() - started,
            info={
                "kind": self._spec.kind,
                "dtype": self._spec.dtype,
                "samples": samples,
                "spread": spread,
                "chunk_size": self._chunk_size,
                "cameras_used": sorted(frames),
                "cameras_missing": missing,
                "uncertainty_source": uncertainty_map.source,
            },
            arm=ArmChunk(
                joints=self._spec.joint_names,
                positions_rad=tuple(
                    tuple(float(v) for v in row) for row in chunks_rad[0, :horizon]
                ),
                dt_s=1.0 / self._spec.fps,
            ),
        )
