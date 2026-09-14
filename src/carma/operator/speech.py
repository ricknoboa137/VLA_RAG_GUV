"""Turning an operator's transcribed words into an intent.

A deliberately small keyword grammar in English and Spanish, not a language
model: it must be predictable, testable by hand, and fast. Anything that is not
a short command becomes a ``CORRECTION`` carrying the operator's words, which is
the raw material for memory.

Rules, in priority order:

1. **Stop wins.** An utterance that starts with an unambiguous stop word, or a
   short utterance containing any stop word, is ``STOP``. Speech recognition is
   too slow and fallible to be the emergency stop; this only makes sure a
   spoken "stop" is never misread as something else.
2. **Commands must be short** (at most four words). "Keep to the left of the
   row" is advice for memory, not a turn command.
3. Short utterances: yes/no phrases, then turn left/right, then go.
4. Everything else non-empty is ``CORRECTION``; empty is ``NONE``.

Words that are ambiguous in running Spanish ("para" also means "for") only count
as stop in short utterances.
"""

from __future__ import annotations

import re
import unicodedata

from carma.types.core import IntentKind, OperatorIntent

_MAX_COMMAND_WORDS = 4

_STOP_STRONG = frozenset({"stop", "halt", "freeze", "alto", "detente", "detener", "quieto"})
_STOP_SHORT_ONLY = frozenset({"para", "parar", "parate"})
_GO = frozenset(
    {"go", "forward", "continue", "proceed", "adelante", "avanza", "avanzar", "sigue", "seguir"}
    | {"continua", "continuar"}
)
_TURN = frozenset({"turn", "gira", "girar", "dobla", "doblar", "voltea"})
_LEFT = frozenset({"left", "izquierda"})
_RIGHT = frozenset({"right", "derecha"})
_YES = frozenset({"yes", "yeah", "yep", "correct", "affirmative", "si", "claro", "correcto"})
_NO = frozenset({"no", "nope", "negative", "incorrect", "negativo", "incorrecto"})
_YES_PHRASES = frozenset({"thats right", "that is right", "exactly", "exacto", "eso es"})


def normalise(text: str) -> list[str]:
    """Lower-case, strip accents and punctuation, and split into words.

    ``"¡Sí, gira a la IZQUIERDA!"`` becomes ``["si", "gira", "a", "la", "izquierda"]``.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_only = ascii_only.replace("'", "").replace(chr(0x2019), "")  # straight and curly
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).split()


def parse_utterance(text: str, language: str = "") -> OperatorIntent:
    """Classify one transcribed utterance.

    Args:
        text: The transcription, in English or Spanish.
        language: Language code reported by speech recognition; carried through
            unchanged, never used to pick a grammar, because short commands are
            often misdetected.
    """
    trimmed = text.strip()
    words = normalise(trimmed)
    if not words:
        return OperatorIntent(IntentKind.NONE, trimmed, language)

    short = len(words) <= _MAX_COMMAND_WORDS
    vocabulary = set(words)

    if words[0] in _STOP_STRONG or (short and vocabulary & (_STOP_STRONG | _STOP_SHORT_ONLY)):
        return OperatorIntent(IntentKind.STOP, trimmed, language)

    if not short:
        return OperatorIntent(IntentKind.CORRECTION, trimmed, language)

    if " ".join(words) in _YES_PHRASES or words[0] in _YES:
        return OperatorIntent(IntentKind.YES, trimmed, language)
    if words[0] in _NO:
        return OperatorIntent(IntentKind.NO, trimmed, language)

    lone = len(words) == 1
    steering = lone or bool(vocabulary & (_TURN | _GO))
    if steering and vocabulary & _LEFT:
        return OperatorIntent(IntentKind.TURN_LEFT, trimmed, language)
    if steering and vocabulary & _RIGHT:
        return OperatorIntent(IntentKind.TURN_RIGHT, trimmed, language)
    if vocabulary & _GO:
        return OperatorIntent(IntentKind.GO, trimmed, language)

    return OperatorIntent(IntentKind.CORRECTION, trimmed, language)
