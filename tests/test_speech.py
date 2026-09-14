"""Operator utterance parsing, with every expected intent stated by hand."""

from __future__ import annotations

import pytest

from carma.operator.speech import normalise, parse_utterance
from carma.types.core import IntentKind


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Stop!", IntentKind.STOP),
        ("alto", IntentKind.STOP),
        ("¡Detente!", IntentKind.STOP),
        ("para", IntentKind.STOP),
        ("no, stop", IntentKind.STOP),
        ("stop, there is a ditch just ahead of the robot", IntentKind.STOP),
        ("turn left", IntentKind.TURN_LEFT),
        ("left", IntentKind.TURN_LEFT),
        ("gira a la derecha", IntentKind.TURN_RIGHT),
        ("go right", IntentKind.TURN_RIGHT),
        ("go forward", IntentKind.GO),
        ("sigue adelante", IntentKind.GO),
        ("yes", IntentKind.YES),
        ("Sí", IntentKind.YES),
        ("that's right", IntentKind.YES),
        ("no", IntentKind.NO),
        ("negativo", IntentKind.NO),
    ],
)
def test_short_commands(text: str, expected: IntentKind) -> None:
    """Each short utterance maps to the intent a person would expect."""
    assert parse_utterance(text).kind is expected


def test_long_advice_is_a_correction_not_a_turn() -> None:
    """Mentioning a direction inside advice must not steer the robot."""
    intent = parse_utterance("keep to the left of the row near the headland")
    assert intent.kind is IntentKind.CORRECTION
    assert intent.text == "keep to the left of the row near the headland"


def test_spanish_para_in_a_sentence_is_not_stop() -> None:
    """'para' means 'for' in running Spanish; only short utterances stop."""
    intent = parse_utterance("para llegar al final gira a la izquierda")
    assert intent.kind is IntentKind.CORRECTION


def test_right_as_agreement_is_not_a_turn() -> None:
    """'that's right' agrees; it does not turn."""
    assert parse_utterance("that's right").kind is IntentKind.YES


def test_empty_and_noise_are_none() -> None:
    """Silence or punctuation only yields nothing to act on."""
    assert parse_utterance("").kind is IntentKind.NONE
    assert parse_utterance("  ...  ").kind is IntentKind.NONE


def test_language_is_carried_through() -> None:
    """The recognised language is recorded, not used to pick a grammar."""
    intent = parse_utterance("gira a la izquierda", language="es")
    assert intent.language == "es"
    assert intent.kind is IntentKind.TURN_LEFT


def test_normalise_strips_accents_and_punctuation() -> None:
    """Accents, case and punctuation never change the parse."""
    assert normalise("¡Sí, gira a la IZQUIERDA!") == ["si", "gira", "a", "la", "izquierda"]
