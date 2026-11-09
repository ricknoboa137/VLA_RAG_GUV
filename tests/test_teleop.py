"""Keyboard and speech to velocity command: hand-computed sequences."""

from __future__ import annotations

from carma.sim.teleop import DriveController
from carma.types.core import Action, IntentKind


def _drain(c: DriveController, n: int) -> list[Action]:
    return [c.next_action() for _ in range(n)]


def test_idle_is_zero() -> None:
    assert DriveController().next_action() == Action(0.0, 0.0)


def test_a_key_press_lasts_exactly_its_hold_window() -> None:
    c = DriveController(linear_mps=0.5, key_hold_steps=3)
    assert c.key("w")
    assert _drain(c, 4) == [Action(0.5, 0.0)] * 3 + [Action(0.0, 0.0)]


def test_turn_keys_and_reverse() -> None:
    c = DriveController(linear_mps=0.5, angular_radps=0.8, key_hold_steps=1)
    c.key("a")
    assert c.next_action() == Action(0.0, 0.8)
    c.key("d")
    assert c.next_action() == Action(0.0, -0.8)
    c.key("s")
    assert c.next_action() == Action(-0.5, 0.0)


def test_unknown_key_is_ignored() -> None:
    assert not DriveController().key("z")


def test_spoken_go_latches_until_stop() -> None:
    c = DriveController(linear_mps=0.4)
    assert c.intent(IntentKind.GO)
    assert _drain(c, 50) == [Action(0.4, 0.0)] * 50
    c.intent(IntentKind.STOP)
    assert c.next_action() == Action(0.0, 0.0)


def test_spoken_turn_is_timed_then_returns_to_go() -> None:
    c = DriveController(linear_mps=0.5, angular_radps=0.8, turn_steps=2, turn_linear_mps=0.2)
    c.intent(IntentKind.GO)
    c.intent(IntentKind.TURN_RIGHT)
    assert _drain(c, 3) == [Action(0.2, -0.8), Action(0.2, -0.8), Action(0.5, 0.0)]


def test_a_held_key_beats_speech_and_stop_beats_everything() -> None:
    c = DriveController(key_hold_steps=2)
    c.intent(IntentKind.GO)
    c.key("a")
    assert c.next_action().angular == 0.8
    c.intent(IntentKind.STOP)
    assert _drain(c, 2) == [Action(0.0, 0.0)] * 2


def test_space_stops_a_latched_go() -> None:
    c = DriveController()
    c.intent(IntentKind.GO)
    c.key(" ")
    assert c.next_action() == Action(0.0, 0.0)
    assert c.status().source == "idle"


def test_non_driving_intents_are_not_consumed() -> None:
    c = DriveController()
    for kind in (IntentKind.YES, IntentKind.NO, IntentKind.CORRECTION, IntentKind.NONE):
        assert not c.intent(kind)
    assert c.next_action() == Action(0.0, 0.0)


def test_the_drive_loop_steps_with_the_controller_commands() -> None:
    """The script's stepping thread feeds controller commands to the simulator."""
    import importlib.util
    import threading
    from pathlib import Path

    from carma.sim.synthetic import SyntheticSim
    from carma.types.core import Pose

    path = Path(__file__).resolve().parents[1] / "scripts/drive_sim.py"
    spec = importlib.util.spec_from_file_location("drive_sim", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sent: list[Action] = []

    class Recording(SyntheticSim):
        def step(self, action: Action):  # type: ignore[no-untyped-def]
            sent.append(action)
            if len(sent) == 6:
                shared.stop = True
            return super().step(action)

    shared = module._Shared()
    ctrl = DriveController(key_hold_steps=2, linear_mps=0.5)
    ctrl.key("w")
    worker = threading.Thread(
        target=module._sim_loop, args=(Recording(dt=0.1), ctrl, shared, Pose(0.0, 0.0, 0.0))
    )
    worker.start()
    worker.join(timeout=5.0)
    assert shared.error is None
    assert sent[:3] == [Action(0.5, 0.0), Action(0.5, 0.0), Action(0.0, 0.0)]
    assert shared.obs is not None and shared.obs.pose.x > 0.0
