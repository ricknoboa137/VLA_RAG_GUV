"""The safety gate between CARMA and the motors, without ROS.

Everything above this gate (the arbiter, the policy server, the bridge) may
hang, crash or be wrong. The gate is the one place that decides whether a
velocity command reaches the wheels, so it is deliberately small:

* **Emergency stop.** While engaged, the output is zero. Releasing it does not
  resume the last command: motion restarts only with a command received after
  the release.
* **Watchdog.** A command older than ``command_timeout_s`` is replaced by zero,
  so a hung policy server or a dead bridge stops the robot instead of leaving
  it driving on its last command.
* **Limits.** Linear and angular speed are clamped to configured maxima.
* **Stall.** When a stall is reported (see :mod:`carma_bridge.motion`), the
  output is zero until a command asks for no motion, which is how the operator
  or the episode loop acknowledges it.

Time is whatever clock the caller passes, in seconds: ROS time on the robot,
simulation time in Gazebo, so a paused simulation never trips the watchdog.

No ROS imports, so the unit suite tests it on a plain machine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    """Configuration of the gate.

    Attributes:
        max_linear_mps: Largest forward or reverse speed passed through, m/s.
        max_angular_radps: Largest yaw rate passed through, rad/s.
        command_timeout_s: Age after which a command is treated as absent, s.
    """

    max_linear_mps: float = 0.6
    max_angular_radps: float = 1.0
    command_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if self.max_linear_mps <= 0 or self.max_angular_radps <= 0:
            raise ValueError("speed limits must be positive")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")


@dataclass(frozen=True)
class GateOutput:
    """What the gate sends to the motors, and why.

    Attributes:
        linear_mps: Forward speed to command, m/s.
        angular_radps: Yaw rate to command, rad/s.
        state: ``"ok"``, ``"limited"``, ``"estop"``, ``"timeout"``,
            ``"stalled"`` or ``"idle"`` (no command received yet).
    """

    linear_mps: float
    angular_radps: float
    state: str


_STOP_EPS = 1e-6


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class SafetyGate:
    """Decides, at every tick, the velocity that reaches the wheels."""

    def __init__(self, limits: SafetyLimits) -> None:
        self._limits = limits
        self._estop = False
        self._stalled = False
        self._command: tuple[float, float, float] | None = None  # (t, linear, angular)

    @property
    def estop(self) -> bool:
        """Whether the emergency stop is engaged."""
        return self._estop

    def set_estop(self, engaged: bool) -> None:
        """Engage or release the emergency stop.

        Engaging or releasing discards the held command, so the robot never
        resumes a command issued before the change.
        """
        if engaged != self._estop:
            self._command = None
        self._estop = engaged

    def report_stall(self) -> None:
        """Latch a stall; cleared by a command that asks for no motion."""
        self._stalled = True
        self._command = None

    def command(self, t_s: float, linear_mps: float, angular_radps: float) -> None:
        """Record a velocity request made at time ``t_s``."""
        at_rest = abs(linear_mps) < _STOP_EPS and abs(angular_radps) < _STOP_EPS
        if self._stalled and at_rest:
            self._stalled = False
        self._command = (t_s, linear_mps, angular_radps)

    def output(self, t_s: float) -> GateOutput:
        """The command to send at time ``t_s``."""
        if self._estop:
            return GateOutput(0.0, 0.0, "estop")
        if self._stalled:
            return GateOutput(0.0, 0.0, "stalled")
        if self._command is None:
            return GateOutput(0.0, 0.0, "idle")
        stamp, linear, angular = self._command
        age = t_s - stamp
        # A negative age means the clock went backwards (a simulation reset):
        # the command belongs to a world that no longer exists.
        if age < 0.0 or age > self._limits.command_timeout_s:
            return GateOutput(0.0, 0.0, "timeout")
        out_linear = _clamp(linear, self._limits.max_linear_mps)
        out_angular = _clamp(angular, self._limits.max_angular_radps)
        limited = out_linear != linear or out_angular != angular
        return GateOutput(out_linear, out_angular, "limited" if limited else "ok")
