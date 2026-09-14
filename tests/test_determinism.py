"""A run is defined by (config, seed, commit). This proves the seed part."""

from __future__ import annotations

from dataclasses import replace

from carma.config import ExperimentConfig
from carma.runtime import run_experiment


def _fingerprint(cfg: ExperimentConfig) -> list[tuple[int, float, float, str]]:
    result = run_experiment(cfg)
    return [
        (s.step, round(s.action.linear, 9), round(s.action.angular, 9), s.decision.choice.value)
        for e in result.episodes
        for s in e.steps
    ]


def test_same_seed_gives_identical_steps(smoke_config: ExperimentConfig) -> None:
    """Two runs at one seed produce identical step records."""
    assert _fingerprint(smoke_config) == _fingerprint(smoke_config)


def test_different_seed_changes_something(smoke_config: ExperimentConfig) -> None:
    """A different seed must actually change the trajectory.

    Guards the opposite failure: a harness that is deterministic because the
    seed is being ignored would pass the test above and be useless.
    """
    other = replace(smoke_config, seed=smoke_config.seed + 977)
    assert _fingerprint(smoke_config) != _fingerprint(other)


def test_seed_is_excluded_from_config_hash(smoke_config: ExperimentConfig) -> None:
    """Seeds pool; configuration changes do not."""
    other = replace(smoke_config, seed=smoke_config.seed + 1)
    assert smoke_config.hash() == other.hash()

    changed = replace(smoke_config, episodes=smoke_config.episodes + 1)
    assert smoke_config.hash() != changed.hash()
