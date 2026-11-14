"""Voice-to-instruction rules and request assembly, without ROS."""

from __future__ import annotations

import numpy as np
import pytest
from carma_vla.pipeline import (
    ARM_JOINTS,
    ChunkPlayer,
    InstructionState,
    action_record,
    arm_state_from_joint_state,
    build_request,
    fit_width,
    parse_utterance_message,
    step_toward,
)

from carma.backbone.wire import ActResponse, ArmChunk, decode_image


def _said(intent: str, text: str, rejected: str | None = None) -> dict[str, object]:
    return {"intent": intent, "text": text, "rejected": rejected}


def test_a_sentence_becomes_the_instruction() -> None:
    state = InstructionState()
    assert not state.ready
    state.apply(_said("correction", "pick up the red bottle by the second row"))
    assert state.instruction == "pick up the red bottle by the second row"
    assert state.ready


def test_stop_pauses_and_go_resumes_the_same_instruction() -> None:
    state = InstructionState(instruction="approach the bin")
    state.apply(_said("stop", "stop"))
    assert state.paused
    assert not state.ready
    state.apply(_said("go", "go"))
    assert state.ready
    assert state.instruction == "approach the bin"


def test_go_without_an_instruction_does_nothing() -> None:
    state = InstructionState()
    assert state.apply(_said("go", "go")) is None
    assert not state.ready


def test_a_new_instruction_also_resumes() -> None:
    state = InstructionState(instruction="old", paused=True)
    state.apply(_said("turn_left", "turn left"))
    assert state.ready
    assert state.instruction == "turn left"


@pytest.mark.parametrize("intent", ["yes", "no", "none"])
def test_answers_are_not_instructions(intent: str) -> None:
    state = InstructionState(instruction="keep going")
    assert state.apply(_said(intent, "yes")) is None
    assert state.instruction == "keep going"


def test_rejected_clips_are_ignored_even_when_they_say_stop() -> None:
    state = InstructionState(instruction="keep going")
    assert state.apply(_said("stop", "stop", rejected="low_confidence")) is None
    assert state.ready


def test_utterance_json_is_decoded_and_junk_is_refused() -> None:
    assert parse_utterance_message('{"intent": "go"}') == {"intent": "go"}
    assert parse_utterance_message("not json") is None
    assert parse_utterance_message("[1, 2]") is None


def test_fit_width_scales_down_only() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    assert fit_width(img, 320).shape == (240, 320, 3)
    assert fit_width(img, 1280).shape == (480, 640, 3)
    assert fit_width(img, 0).shape == (480, 640, 3)


def _pair(value: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((480, 640, 3), value, np.uint8), np.full((480, 640, 3), value + 1, np.uint8)


def _request(primary: str, cameras: tuple[str, ...] = ("arm", "world"), width: int = 320):
    values = {"arm": 10, "world": 50}
    return build_request(
        {cam: _pair(values[cam]) for cam in cameras},
        primary,
        "pick up the bottle",
        pose=(1.0, 2.0, 0.5),
        t=3.0,
        stage="VEGETATIVE",
        seed=7,
        samples=4,
        width=width,
    )


def test_primary_camera_fills_image_and_the_other_goes_to_views() -> None:
    request = _request("arm")
    assert decode_image(request.image).shape == (240, 320, 3)
    assert int(decode_image(request.image).mean()) == 10
    assert request.image_right is not None
    assert int(decode_image(request.image_right).mean()) == 11
    assert [name for name, _ in request.views] == ["world_left", "world_right"]
    assert int(decode_image(dict(request.views)["world_right"]).mean()) == 51
    assert request.instruction == "pick up the bottle"
    assert request.seed == 7


def test_world_can_be_primary() -> None:
    request = _request("world", width=0)
    assert int(decode_image(request.image).mean()) == 50
    assert decode_image(request.image).shape == (480, 640, 3)
    assert [name for name, _ in request.views] == ["arm_left", "arm_right"]


def test_one_camera_sends_no_views() -> None:
    assert _request("arm", cameras=("arm",)).views == ()


def test_a_missing_primary_camera_is_an_error() -> None:
    with pytest.raises(KeyError):
        _request("arm", cameras=("world",))


def test_action_record_names_what_was_sent_and_what_answered() -> None:
    request = _request("arm", width=64)
    response = ActResponse(0.3, 0.1, False, 0.2, "heuristic", "carma:heuristic", 0.01)
    record = action_record(response, request, "arm", 0.05)
    assert record["views"] == ["world_left", "world_right"]
    assert record["stereo"] is True
    assert record["model_id"] == "carma:heuristic"
    assert record["linear"] == 0.3
    assert record["seed"] == 7
    assert record["arm"] is None


# ------------------------------------------------------------------- the arm


def _chunk(rows: int = 3, dt_s: float = 0.1) -> ArmChunk:
    return ArmChunk(
        joints=("shoulder_pan", "gripper"),
        positions_rad=tuple((0.1 * i, 0.5) for i in range(rows)),
        dt_s=dt_s,
    )


def test_arm_state_comes_out_in_urdf_order() -> None:
    names = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
    state = arm_state_from_joint_state(names, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], ARM_JOINTS)
    assert state == tuple(zip(ARM_JOINTS, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True))


def test_a_partial_joint_state_is_no_state() -> None:
    assert arm_state_from_joint_state(["gripper"], [0.1], ARM_JOINTS) == ()


def test_the_request_carries_the_arm_state() -> None:
    frames = {"arm": (np.zeros((8, 8, 3), dtype=np.uint8), None)}
    state = tuple((name, 0.0) for name in ARM_JOINTS)
    request = build_request(
        frames,
        "arm",
        "pick",
        pose=(0, 0, 0),
        t=0,
        stage="V",
        seed=0,
        samples=1,
        width=0,
        arm_state=state,
    )
    assert request.arm_state == state


def test_the_chunk_plays_against_the_clock_from_its_start() -> None:
    player = ChunkPlayer()
    assert player.target(10.0) is None
    player.load(_chunk(), t0=10.0)
    assert player.target(10.0) == {"shoulder_pan": 0.0, "gripper": 0.5}
    # 0.25 s after the start: row 2, whatever ticks were missed on the way.
    assert player.target(10.25) == pytest.approx({"shoulder_pan": 0.2, "gripper": 0.5})
    # Past the last row nothing is commanded, and the arm holds.
    assert player.target(10.31) is None


def test_an_older_answer_never_replaces_a_newer_one() -> None:
    player = ChunkPlayer()
    newer = _chunk()
    player.load(newer, t0=5.0)
    player.load(_chunk(rows=5), t0=4.0)
    assert player.chunk is newer


def test_clear_stops_the_chunk() -> None:
    player = ChunkPlayer()
    player.load(_chunk(), t0=0.0)
    player.clear()
    assert player.target(0.0) is None


def test_a_step_is_clamped_to_the_limits_and_the_speed() -> None:
    limits = {"shoulder_pan": (-1.0, 1.0), "gripper": (0.0, 0.4)}
    previous = {"shoulder_pan": 0.0, "gripper": 0.38}
    command = step_toward({"shoulder_pan": 3.0, "gripper": 0.9}, previous, 0.05, limits)
    # The pan wants 3.0, is clamped to 1.0, and moves 0.05 toward it.
    # The gripper wants 0.9, is clamped to 0.4, and only 0.02 remains.
    assert command == pytest.approx({"shoulder_pan": 0.05, "gripper": 0.4})


def test_a_joint_without_a_limit_or_a_position_is_not_commanded() -> None:
    command = step_toward(
        {"shoulder_pan": 0.1, "mystery": 0.2, "gripper": 0.1},
        {"shoulder_pan": 0.0, "mystery": 0.0},
        1.0,
        {"shoulder_pan": (-1.0, 1.0), "gripper": (0.0, 1.0)},
    )
    assert command == {"shoulder_pan": 0.1}


def test_action_record_summarises_an_arm_chunk() -> None:
    request = _request("arm", width=64)
    response = ActResponse(
        0.0, 0.0, False, 0.4, "chunk_spread/uncalibrated", "m", 0.5, arm=_chunk()
    )
    record = action_record(response, request, "arm", 0.6)
    assert record["arm"] == {
        "steps": 3,
        "dt_s": 0.1,
        "first": {"shoulder_pan": 0.0, "gripper": 0.5},
    }
