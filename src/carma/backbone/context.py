"""How retrieved or answered operator knowledge reaches a language-conditioned policy.

A backbone receives context as a tuple of strings (``NavigationBackbone.act``).
A vision-language-action model has one text input, the task, so the context
has to be folded into it. This module is the one place that does it, and the
demonstration recorder, the dataset it feeds and the policy server all call
it, so a model is trained on exactly the text it will be served.

Format: the instruction, then ``". operator: "`` and the context items joined
by ``"; "``. With no context the task is the instruction unchanged, which is
what every model trained before context existed saw.
"""

from __future__ import annotations

import re

_STOP_AT = re.compile(r"stop at (\d+(?:\.\d+)?)\s*(?:metres|meters|m)\b", re.IGNORECASE)


def task_text(instruction: str, context: tuple[str, ...]) -> str:
    """The policy's task string for ``instruction`` with operator ``context``."""
    items = tuple(c.strip() for c in context if c.strip())
    if not items:
        return instruction
    return f"{instruction}. operator: {'; '.join(items)}"


def stop_distance(context: tuple[str, ...]) -> float | None:
    """The distance, in metres, the operator said to stop at, if any context says one."""
    for item in context:
        match = _STOP_AT.search(item)
        if match is not None:
            return float(match.group(1))
    return None
