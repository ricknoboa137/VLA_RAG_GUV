"""The remote backbone, its wire contract, and the guarantees it must keep.

No test here opens a socket. The transport is injected, so the suite stays
offline as section 2.3 requires while still exercising every byte of the
encoding and the whole decision path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from carma.backbone.remote import HttpTransport, RemoteBackbone
from carma.backbone.wire import (
    ActRequest,
    ActResponse,
    ArmChunk,
    decode_image,
    decode_png16,
    encode_image,
    encode_png16,
)
from carma.types import BackboneError
from carma.types.core import Observation, PhenologyStage, Pose

_RNG = np.random.default_rng(0)


def _image(height: int = 24, width: int = 32) -> np.ndarray:
    return _RNG.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _observation(**overrides: Any) -> Observation:
    base = {
        "rgb": _image(),
        "depth": None,
        "pose": Pose(x=1.5, y=-2.0, yaw=0.25),
        "t": 3.0,
        "instruction": "follow the row",
        "stage": PhenologyStage.VEGETATIVE,
    }
    base.update(overrides)
    return Observation(**base)  # type: ignore[arg-type]


class RecordingTransport:
    """Answers with a fixed reply and keeps every request it was given."""

    def __init__(self, reply: dict[str, Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = reply if reply is not None else _ok_reply()

    def post(self, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
        self.requests.append(dict(payload))
        self.timeout_s = timeout_s
        return dict(self.reply)


def _ok_reply(**overrides: Any) -> dict[str, Any]:
    reply = {
        "linear": 0.4,
        "angular": -0.1,
        "stop": False,
        "uncertainty": 0.25,
        "uncertainty_method": "chunk_spread",
        "model_id": "test/policy",
        "model_revision": "abc123",
        "inference_s": 0.012,
    }
    reply.update(overrides)
    return reply


# --------------------------------------------------------------- the wire


def test_image_survives_the_round_trip_exactly() -> None:
    """PNG is lossless, so the server must see the pixels the robot saw."""
    original = _image(48, 64)
    assert np.array_equal(decode_image(encode_image(original)), original)


def test_depth_round_trips_as_16_bit_millimetres() -> None:
    depth_mm = np.array([[0, 1, 1500, 65535]], dtype=np.uint16)
    assert np.array_equal(decode_png16(encode_png16(depth_mm)), depth_mm)


def test_depth_encoder_rejects_the_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="uint16"):
        encode_png16(np.zeros((4, 4), dtype=np.uint8))  # type: ignore[arg-type]


def test_request_round_trips_through_json() -> None:
    request = ActRequest(
        image=encode_image(_image(8, 8)),
        instruction="turn at the post",
        pose=(1.0, 2.0, 3.0),
        t=4.5,
        stage="FLOWERING",
        seed=99,
        samples=8,
        context=("the left row is blocked",),
    )
    assert ActRequest.from_json(request.to_json()) == request


def test_request_rejects_a_missing_field() -> None:
    with pytest.raises(ValueError, match="missing"):
        ActRequest.from_json({"image": "x", "instruction": "y"})


def test_request_rejects_a_malformed_pose() -> None:
    with pytest.raises(ValueError, match="pose"):
        ActRequest.from_json(
            {
                "image": "x",
                "instruction": "y",
                "pose": [1.0, 2.0],
                "t": 0.0,
                "stage": "VEGETATIVE",
                "seed": 0,
            },
        )


def test_response_rejects_an_uncertainty_outside_the_unit_interval() -> None:
    """The Protocol promises [0, 1]; a server that breaks it must not pass silently."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ActResponse.from_json(_ok_reply(uncertainty=1.4))


# ----------------------------------------------------------- the backbone


def test_act_returns_the_servers_action_and_uncertainty() -> None:
    backbone = RemoteBackbone(transport=RecordingTransport())
    action, uncertainty = backbone.act(_observation(), rng=np.random.default_rng(1))

    assert action.linear == pytest.approx(0.4)
    assert action.angular == pytest.approx(-0.1)
    assert action.stop is False
    assert uncertainty == pytest.approx(0.25)


def test_the_seed_follows_the_run_seed() -> None:
    """Same run seed, same sequence of requests: determinism depends on it."""
    seeds = []
    for _ in range(2):
        transport = RecordingTransport()
        backbone = RemoteBackbone(transport=transport)
        rng = np.random.default_rng(7)
        for _ in range(3):
            backbone.act(_observation(), rng=rng)
        seeds.append([r["seed"] for r in transport.requests])

    assert seeds[0] == seeds[1]
    assert len(set(seeds[0])) == 3, "each step must get its own seed"


def test_context_is_forwarded() -> None:
    transport = RecordingTransport()
    backbone = RemoteBackbone(transport=transport)
    backbone.act(_observation(), ("go around the left side",), rng=np.random.default_rng(0))
    assert transport.requests[0]["context"] == ["go around the left side"]


def test_the_model_that_answered_is_recorded_for_the_manifest() -> None:
    """A result whose weights are unnamed is not a result."""
    backbone = RemoteBackbone(transport=RecordingTransport())
    assert backbone.last_model_id is None

    backbone.act(_observation(), rng=np.random.default_rng(0))
    assert backbone.last_model_id == "test/policy"
    assert backbone.last_model_revision == "abc123"
    assert backbone.last_uncertainty_method == "chunk_spread"

    backbone.reset()
    assert backbone.last_model_id is None, "reset must not leave a stale attribution"


def test_stereo_and_depth_are_omitted_unless_asked_for() -> None:
    """They are most of the payload and most policies ignore them."""
    transport = RecordingTransport()
    obs = _observation(rgb_right=_image(), depth=np.full((24, 32), 2.0, dtype=np.float32))

    RemoteBackbone(transport=transport).act(obs, rng=np.random.default_rng(0))
    assert transport.requests[0]["image_right"] is None
    assert transport.requests[0]["depth_mm"] is None

    both = RecordingTransport()
    RemoteBackbone(transport=both, send_right=True, send_depth=True).act(
        obs,
        rng=np.random.default_rng(0),
    )
    assert both.requests[0]["image_right"] is not None
    sent = decode_png16(str(both.requests[0]["depth_mm"]))
    assert sent[0, 0] == 2000, "depth must cross as millimetres"


def test_unreadable_depth_becomes_zero_not_a_distance() -> None:
    """0 means no reading in the RGB-D convention; NaN has no integer form."""
    depth = np.full((4, 4), np.nan, dtype=np.float32)
    depth[0, 0] = 1.25
    transport = RecordingTransport()
    RemoteBackbone(transport=transport, send_depth=True).act(
        _observation(rgb=_image(4, 4), depth=depth),
        rng=np.random.default_rng(0),
    )
    sent = decode_png16(str(transport.requests[0]["depth_mm"]))
    assert sent[0, 0] == 1250
    assert sent[1, 1] == 0


def test_a_broken_reply_fails_loudly() -> None:
    """Better a failed episode than an action invented from a bad payload."""
    backbone = RemoteBackbone(transport=RecordingTransport(reply={"linear": 1.0}))
    with pytest.raises(BackboneError, match="unusable reply"):
        backbone.act(_observation(), rng=np.random.default_rng(0))


def test_samples_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="samples"):
        RemoteBackbone(samples=0)


def test_it_is_registered_for_config() -> None:
    """A condition is a config file, so the key must resolve."""
    from carma.backbone.registry import BACKBONES

    # Registry.keys() returns a tuple, not a mapping view.
    registered = BACKBONES.keys()
    assert "remote" in registered


def test_extra_camera_views_round_trip_in_order() -> None:
    request = ActRequest(
        image="L",
        image_right="R",
        instruction="pick up the red bottle",
        pose=(0.0, 0.0, 0.0),
        t=0.0,
        stage="VEGETATIVE",
        seed=0,
        views=(("world_left", "WL"), ("world_right", "WR")),
    )
    payload = request.to_json()
    assert payload["views"] == {"world_left": "WL", "world_right": "WR"}
    assert list(payload["views"]) == ["world_left", "world_right"]
    assert ActRequest.from_json(payload) == request


def test_a_request_without_views_has_none() -> None:
    payload = {"image": "x", "instruction": "y", "pose": [0, 0, 0], "t": 0, "stage": "V", "seed": 1}
    assert ActRequest.from_json(payload).views == ()


def test_duplicate_view_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        ActRequest(
            image="x",
            instruction="y",
            pose=(0.0, 0.0, 0.0),
            t=0.0,
            stage="V",
            seed=0,
            views=(("arm_left", "a"), ("arm_left", "b")),
        )


# ------------------------------------------------------- the arm, on the wire

_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _bare_request(**overrides: Any) -> ActRequest:
    base: dict[str, Any] = {
        "image": "x",
        "instruction": "pick up the red cube",
        "pose": (0.0, 0.0, 0.0),
        "t": 0.0,
        "stage": "VEGETATIVE",
        "seed": 0,
    }
    base.update(overrides)
    return ActRequest(**base)


def test_arm_state_round_trips_in_joint_order() -> None:
    state = tuple((name, 0.1 * i) for i, name in enumerate(_ARM_JOINTS))
    request = _bare_request(arm_state=state)
    payload = request.to_json()
    assert list(payload["arm_state"]) == list(_ARM_JOINTS)
    assert ActRequest.from_json(payload) == request


def test_a_request_without_an_arm_has_no_arm_state() -> None:
    payload = {"image": "x", "instruction": "y", "pose": [0, 0, 0], "t": 0, "stage": "V", "seed": 1}
    assert ActRequest.from_json(payload).arm_state == ()


def test_arm_state_rejects_duplicates_and_non_finite_angles() -> None:
    with pytest.raises(ValueError, match="unique"):
        _bare_request(arm_state=(("gripper", 0.0), ("gripper", 0.1)))
    with pytest.raises(ValueError, match="finite"):
        _bare_request(arm_state=(("gripper", float("nan")),))


def test_arm_chunk_round_trips_through_the_response() -> None:
    chunk = ArmChunk(
        joints=_ARM_JOINTS,
        positions_rad=((0.0, -0.5, 0.5, 0.2, 0.0, 0.3), (0.01, -0.49, 0.5, 0.2, 0.0, 0.35)),
        dt_s=1.0 / 30.0,
    )
    response = ActResponse.from_json(_ok_reply(linear=0.0, angular=0.0, arm=chunk.to_json()))
    assert response.arm == chunk
    assert ActResponse.from_json(response.to_json()) == response


def test_a_navigation_reply_carries_no_arm() -> None:
    response = ActResponse.from_json(_ok_reply())
    assert response.arm is None
    assert response.to_json()["arm"] is None


@pytest.mark.parametrize(
    ("joints", "rows", "dt_s", "match"),
    [
        ((), ((0.0,),), 0.1, "non-empty"),
        (("a", "a"), ((0.0, 0.0),), 0.1, "unique"),
        (("a", "b"), (), 0.1, "at least one step"),
        (("a", "b"), ((0.0,),), 0.1, "1 values for 2 joints"),
        (("a",), ((float("inf"),),), 0.1, "non-finite"),
        (("a",), ((0.0,),), 0.0, "positive"),
    ],
)
def test_arm_chunk_rejects_a_malformed_trajectory(
    joints: tuple[str, ...], rows: tuple[tuple[float, ...], ...], dt_s: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ArmChunk(joints=joints, positions_rad=rows, dt_s=dt_s)


def test_arm_chunk_rejects_a_missing_field() -> None:
    with pytest.raises(ValueError, match="dt_s"):
        ArmChunk.from_json({"joints": ["a"], "positions_rad": [[0.0]]})


def test_the_navigation_backbone_refuses_an_arm_policy() -> None:
    """An arm server's zero velocity must not pass for a navigation action."""
    arm = ArmChunk(joints=("gripper",), positions_rad=((0.2,),), dt_s=0.1).to_json()
    backbone = RemoteBackbone(transport=RecordingTransport(_ok_reply(arm=arm)))
    with pytest.raises(BackboneError, match="arm policy"):
        backbone.act(_observation(), (), rng=np.random.default_rng(0))


def test_the_navigation_backbone_sends_no_arm_state() -> None:
    transport = RecordingTransport()
    RemoteBackbone(transport=transport).act(_observation(), (), rng=np.random.default_rng(0))
    assert transport.requests[0]["arm_state"] == {}


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionResetError("reset")])
def test_a_reply_that_never_arrives_is_a_backbone_error(
    failure: OSError, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read timeout is TimeoutError, not URLError; it once escaped and froze the live loop."""

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(BackboneError, match="did not answer"):
        HttpTransport("http://127.0.0.1:1/act").post({}, timeout_s=0.1)
