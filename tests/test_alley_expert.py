"""The scripted alley expert: lane keeping, stopping, and the instruction it needs."""

from __future__ import annotations

import numpy as np
import pytest

from carma.backbone.alley_expert import AlleyExpert
from carma.backbone.registry import BACKBONES
from carma.runtime.episodes import row_alley_episodes
from carma.sim.synthetic import SyntheticSim
from carma.types import BackboneError
from carma.types.core import Observation, PhenologyStage, Pose


def _obs(
    x: float = 1.0, y: float = 0.0, yaw: float = 0.0, text: str = "drive 14 metres"
) -> Observation:
    return Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        pose=Pose(x, y, yaw),
        t=0.0,
        instruction=text,
        stage=PhenologyStage.VEGETATIVE,
    )


def _act(expert: AlleyExpert, obs: Observation):  # type: ignore[no-untyped-def]
    return expert.act(obs, (), rng=np.random.default_rng(0))


def test_it_is_registered() -> None:
    assert BACKBONES.create("alley_expert") is not None


def test_on_the_centreline_it_drives_straight() -> None:
    action, uncertainty = _act(AlleyExpert(), _obs())
    assert (action.linear, action.angular, action.stop) == (0.5, 0.0, False)
    assert uncertainty == 0.0


def test_left_of_the_line_it_turns_right() -> None:
    # y = +0.2 m, heading 0: -(1.0 * 0.2) = -0.2 rad/s, a right turn (negative yaw rate).
    action, _ = _act(AlleyExpert(), _obs(y=0.2))
    assert action.angular == pytest.approx(-0.2)


def test_heading_error_adds_to_lateral_error() -> None:
    # y = -0.1, yaw = +0.2: -(1.0 * -0.1) - (1.5 * 0.2) = 0.1 - 0.3 = -0.2.
    action, _ = _act(AlleyExpert(), _obs(y=-0.1, yaw=0.2))
    assert action.angular == pytest.approx(-0.2)


def test_the_yaw_rate_is_capped() -> None:
    action, _ = _act(AlleyExpert(max_angular=0.8), _obs(y=3.0))
    assert action.angular == -0.8


def test_it_stops_a_margin_before_the_named_distance() -> None:
    expert = AlleyExpert(stop_margin_m=0.3)
    assert not _act(expert, _obs(x=13.6))[0].stop
    stopped, _ = _act(expert, _obs(x=13.7))
    assert stopped.stop and stopped.linear == 0.0


@pytest.mark.parametrize(
    "text", ["drive 14 metres down the row and stop", "go 14 m", "14.0 meters"]
)
def test_it_reads_the_distance_from_several_phrasings(text: str) -> None:
    assert _act(AlleyExpert(), _obs(x=14.0, text=text))[0].stop


def test_an_instruction_without_a_distance_is_an_error() -> None:
    with pytest.raises(BackboneError, match="distance"):
        _act(AlleyExpert(), _obs(text="follow the row"))


def test_generated_instructions_are_ones_the_expert_can_read() -> None:
    for spec in row_alley_episodes(20, rng=np.random.default_rng(1)):
        assert spec.instruction == f"drive {spec.goal.x:.0f} metres down the row and stop"
        action, _ = _act(AlleyExpert(), _obs(x=spec.goal.x, text=spec.instruction))
        assert action.stop


def test_on_a_straight_alley_it_reaches_the_goal_in_the_synthetic_sim() -> None:
    """Closed loop on kinematics only: from an offset start it converges and stops at the goal."""
    sim = SyntheticSim(dt=0.1)
    spec = row_alley_episodes(
        1, rng=np.random.default_rng(2), min_length_m=10.0, max_length_m=10.0
    )[0]
    obs = sim.reset(seed=0, start=Pose(0.0, 0.25, 0.2))
    expert = AlleyExpert()
    for _ in range(spec.max_steps):
        action, _ = _act(expert, _with_text(obs, spec.instruction))
        if action.stop:
            break
        obs = sim.step(action)
    assert action.stop
    assert abs(obs.pose.y) < 0.05  # converged onto the centreline
    assert obs.pose.distance_to(spec.goal) <= spec.success_radius


def _with_text(obs: Observation, text: str) -> Observation:
    return Observation(
        rgb=obs.rgb,
        depth=obs.depth,
        pose=obs.pose,
        t=obs.t,
        instruction=text,
        stage=obs.stage,
    )
