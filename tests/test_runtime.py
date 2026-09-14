"""The episode loop, end to end, on the synthetic simulator."""

from __future__ import annotations

from carma.config import ExperimentConfig
from carma.runtime import run_experiment
from carma.types.core import Choice


def test_experiment_runs_and_reports(smoke_config: ExperimentConfig) -> None:
    """A smoke run produces one record per episode and a full metric bundle."""
    result = run_experiment(smoke_config)
    assert len(result.episodes) == smoke_config.episodes
    assert result.metrics.episodes == smoke_config.episodes
    assert 0.0 <= result.metrics.success_rate <= 1.0
    assert result.config_hash


def test_every_step_records_all_three_prices(smoke_config: ExperimentConfig) -> None:
    """The decision log must let a reader recompute the choice."""
    result = run_experiment(smoke_config)
    for episode in result.episodes:
        for step in episode.steps:
            assert set(step.decision.costs) == set(Choice)


def test_no_memory_condition_never_asks_or_retrieves(
    smoke_config: ExperimentConfig,
) -> None:
    """The floor baseline is obtained by swapping the arbiter, nothing else."""
    from dataclasses import replace

    from carma.config import ComponentSpec

    cfg = replace(
        smoke_config,
        condition="no_memory",
        arbiter=ComponentSpec(kind="always_act", params={}),
    )
    result = run_experiment(cfg)
    for episode in result.episodes:
        for step in episode.steps:
            assert step.decision.choice is Choice.ACT
            assert not step.asked
            assert step.retrieved_ids == ()
