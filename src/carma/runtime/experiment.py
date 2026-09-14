"""Running a full experiment and collecting its result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carma.config import ExperimentConfig, build
from carma.metrics import MetricBundle, summarise
from carma.runtime.episode import run_episode
from carma.runtime.episodes import synthetic_episodes
from carma.types.core import EpisodeRecord


@dataclass(frozen=True, slots=True)
class RunResult:
    """Episodes and their metrics for one experiment."""

    config_hash: str
    episodes: tuple[EpisodeRecord, ...]
    metrics: MetricBundle


def run_experiment(cfg: ExperimentConfig, *, run_id: str = "") -> RunResult:
    """Build the assembly, run every episode, and summarise.

    Episode seeds are derived from the master seed by a fixed rule
    (``seed + index``) so that adding an episode never changes the behaviour of
    the ones before it.
    """
    assembly = build(cfg)
    spec_rng = np.random.default_rng(cfg.seed)
    specs = synthetic_episodes(cfg.episodes, rng=spec_rng)

    records = tuple(
        run_episode(spec, assembly, seed=cfg.seed + i, run_id=run_id)
        for i, spec in enumerate(specs)
    )
    assembly.sim.close()
    return RunResult(
        config_hash=cfg.hash(),
        episodes=records,
        metrics=summarise(records),
    )
