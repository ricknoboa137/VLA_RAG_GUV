"""The MQTT operator's payload parsing, without a broker."""

from __future__ import annotations

import json

from carma.operator.mqtt import parse_utterance_payload


def _msg(**fields: object) -> bytes:
    return json.dumps(fields).encode("utf-8")


def test_a_correction_utterance_becomes_a_pending_correction() -> None:
    pending = parse_utterance_payload(
        _msg(text="  stay left when the canopy narrows ", intent="correction", language="en")
    )
    assert pending is not None
    assert pending.text == "stay left when the canopy narrows"
    assert pending.language == "en"


def test_other_intents_rejections_and_junk_are_dropped() -> None:
    assert parse_utterance_payload(_msg(text="stop", intent="stop")) is None
    assert parse_utterance_payload(_msg(text="hello", rejected="no_speech")) is None
    assert parse_utterance_payload(_msg(text="   ", intent="correction")) is None
    assert parse_utterance_payload(b"not json") is None
    assert parse_utterance_payload(b"[1, 2]") is None


def test_an_unknown_intent_counts_as_a_correction() -> None:
    pending = parse_utterance_payload(_msg(text="mind the gap", intent="mumble"))
    assert pending is not None and pending.text == "mind the gap"
