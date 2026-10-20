"""Voice-to-instruction rules and request assembly, without ROS."""

from __future__ import annotations

import numpy as np
import pytest

from carma.backbone.wire import ActResponse, decode_image
from carma_vla.pipeline import (
    InstructionState,
    action_record,
    build_request,
    fit_width,
    parse_utterance_message,
)


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
