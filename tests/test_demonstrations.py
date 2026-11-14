"""Demonstration recording: labels stay clean, disturbances are seeded and recoverable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carma.backbone.alley_expert import AlleyExpert
from carma.runtime.demonstrations import Disturbance, record_episode, write_episode
from carma.runtime.episodes import row_alley_episodes
from carma.sim.synthetic import SyntheticSim


def _record(seed: int, disturbance: Disturbance | None):  # type: ignore[no-untyped-def]
    spec = row_alley_episodes(1, rng=np.random.default_rng(5), min_length_m=8.0, max_length_m=8.0)[
        0
    ]
    return record_episode(
        SyntheticSim(dt=0.1),
        AlleyExpert(),
        spec,
        seed=seed,
        rng=np.random.default_rng(seed),
        disturbance=disturbance,
    )


def test_a_clean_episode_ends_with_the_stop_step() -> None:
    episode = _record(0, None)
    assert episode.steps[-1].label.stop
    assert not any(s.disturbed for s in episode.steps)
    assert all(s.executed.angular == s.label.angular for s in episode.steps)


def test_disturbed_steps_execute_a_different_command_than_the_label() -> None:
    burst = Disturbance(start_probability=0.5, magnitude_radps=0.4)
    episode = _record(1, burst)
    disturbed = [s for s in episode.steps if s.disturbed]
    assert disturbed
    for s in disturbed:
        assert abs(s.executed.angular - s.label.angular) == np.float64(0.4) or np.isclose(
            abs(s.executed.angular - s.label.angular), 0.4
        )
    # The label on a disturbed state is the expert's own reply to that state.
    expert = AlleyExpert()
    for s in disturbed[:3]:
        assert (
            s.label.angular
            == expert.act(_obs_like(episode, s), (), rng=np.random.default_rng(0))[0].angular
        )


def _obs_like(episode, step):  # type: ignore[no-untyped-def]
    from carma.types.core import Observation

    return Observation(
        step.rgb, None, step.pose, step.t, episode.spec.instruction, episode.spec.stage
    )


def test_the_expert_recovers_after_disturbance_and_stops() -> None:
    episode = _record(2, Disturbance(start_probability=0.1))
    assert episode.steps[-1].label.stop
    assert abs(episode.steps[-1].pose.y) < 0.4


def test_same_seed_records_the_same_episode() -> None:
    d = Disturbance(start_probability=0.1)
    a, b = _record(3, d), _record(3, d)
    assert [s.executed for s in a.steps] == [s.executed for s in b.steps]
    assert [s.disturbed for s in a.steps] != [s.disturbed for s in _record(4, d).steps]


def test_written_episode_round_trips(tmp_path: Path) -> None:
    import cv2

    episode = _record(0, None)
    folder = write_episode(tmp_path, episode)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["steps"]) == len(episode.steps)
    assert meta["steps"][-1]["label"][2] is True
    assert meta["instruction"] == episode.spec.instruction
    first = cv2.imread(str(folder / "frames" / "000000.png"))
    assert np.array_equal(first, episode.steps[0].rgb)


def test_a_collision_ends_the_recording_and_is_flagged(tmp_path: Path) -> None:
    from dataclasses import replace

    from carma.types.core import Action, Observation

    class Crashing(SyntheticSim):
        count = 0

        def step(self, action: Action) -> Observation:
            Crashing.count += 1
            return replace(super().step(action), collided=Crashing.count >= 5)

    spec = row_alley_episodes(1, rng=np.random.default_rng(5), min_length_m=8.0, max_length_m=8.0)[
        0
    ]
    episode = record_episode(
        Crashing(dt=0.1),
        AlleyExpert(),
        spec,
        seed=0,
        rng=np.random.default_rng(0),
        disturbance=None,
    )
    assert episode.collided
    assert len(episode.steps) == 5  # the five states before the crash, not the crashed one
    meta = json.loads((write_episode(tmp_path, episode) / "meta.json").read_text(encoding="utf-8"))
    assert meta["collided"] is True
