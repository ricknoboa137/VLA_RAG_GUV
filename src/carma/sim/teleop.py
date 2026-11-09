"""Turn keyboard presses and spoken intents into the velocity command for each step.

Pure logic, no window and no network, so it is tested by hand-computed
sequences. ``scripts/drive_sim.py`` wires it to a camera window, the keyboard
and the MQTT speech channel.

Precedence, highest first: a stop (key or spoken) always wins and clears
everything; a held key beats speech; a timed spoken turn beats a latched spoken
"go"; otherwise the platform stands still. Time is counted in control steps,
not seconds, so the behaviour does not depend on how slowly the renderer runs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from carma.types.core import Action, IntentKind


@dataclass(frozen=True)
class DriveStatus:
    """What the controller is doing, for the on-screen overlay."""

    source: str  # "key", "voice", "voice-turn" or "idle"
    latched_go: bool


class DriveController:
    """Maps key presses and speech intents to :class:`Action` each control step.

    Args:
        linear_mps: Forward speed in metres per second.
        angular_radps: Turn rate in radians per second.
        key_hold_steps: Control steps a key press stays active. A terminal or
            window gives press events, not releases, so a key is "held" by
            repeating; the repeat rate must beat this window.
        turn_steps: Control steps a spoken turn lasts (about 0.8 rad at the
            defaults with a 10 Hz step: 10 x 0.1 s x 0.8 rad/s).
        turn_linear_mps: Forward speed while executing a spoken turn.
    """

    def __init__(
        self,
        linear_mps: float = 0.5,
        angular_radps: float = 0.8,
        key_hold_steps: int = 3,
        turn_steps: int = 10,
        turn_linear_mps: float = 0.2,
    ) -> None:
        self._linear = linear_mps
        self._angular = angular_radps
        self._key_hold = key_hold_steps
        self._turn_steps = turn_steps
        self._turn_linear = turn_linear_mps
        self._lock = threading.Lock()
        self._key: tuple[float, float] = (0.0, 0.0)
        self._key_left = 0
        self._turn: float = 0.0
        self._turn_left = 0
        self._go = False

    def key(self, name: str) -> bool:
        """Register a key press; returns False for a key the controller ignores."""
        lowered = name.lower()
        with self._lock:
            if lowered in ("x", " "):
                self._clear()
                return True
            table = {
                "w": (self._linear, 0.0),
                "s": (-self._linear, 0.0),
                "a": (0.0, self._angular),
                "d": (0.0, -self._angular),
                "q": (self._linear, self._angular),
                "e": (self._linear, -self._angular),
            }
            if lowered not in table:
                return False
            self._key = table[lowered]
            self._key_left = self._key_hold
            return True

    def intent(self, kind: IntentKind) -> bool:
        """Apply a spoken intent; returns True if it changed the driving command.

        ``yes``, ``no`` and free-form corrections do not drive the platform, so
        they return False and the caller shows them as suggestions instead.
        """
        with self._lock:
            if kind is IntentKind.STOP:
                self._clear()
            elif kind is IntentKind.GO:
                self._go = True
            elif kind in (IntentKind.TURN_LEFT, IntentKind.TURN_RIGHT):
                self._turn = self._angular if kind is IntentKind.TURN_LEFT else -self._angular
                self._turn_left = self._turn_steps
            else:
                return False
            return True

    def next_action(self) -> Action:
        """The command for the coming step; call exactly once per control step."""
        with self._lock:
            if self._key_left > 0:
                self._key_left -= 1
                return Action(*self._key)
            if self._turn_left > 0:
                self._turn_left -= 1
                return Action(self._turn_linear, self._turn)
            if self._go:
                return Action(self._linear, 0.0)
            return Action(0.0, 0.0)

    def status(self) -> DriveStatus:
        """Current source of the command, without advancing anything."""
        with self._lock:
            if self._key_left > 0:
                return DriveStatus("key", self._go)
            if self._turn_left > 0:
                return DriveStatus("voice-turn", self._go)
            return DriveStatus("voice" if self._go else "idle", self._go)

    def _clear(self) -> None:
        self._key_left = 0
        self._turn_left = 0
        self._go = False
