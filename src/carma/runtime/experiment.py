"""Running a full experiment and collecting its result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carma.config import ExperimentConfig, build
from carma.metrics import MetricBundle, summarise
from carma.runtime.episode import run_episode
from carma.runtime.episodes import row_alley_episodes, synthetic_episodes
from carma.types import ConfigError
from carma.types.core import EpisodeRecord, EpisodeSpec, PhenologyStage


@dataclass(frozen=True, slots=True)
class RunResult:
    """Episodes and their metrics for one experiment."""

    config_hash: str
    episodes: tuple[EpisodeRecord, ...]
    metrics: MetricBundle


def _episodes(cfg: ExperimentConfig) -> tuple[EpisodeSpec, ...]:
    """The episode specs the config asks for."""
    spec_rng = np.random.default_rng(cfg.seed)
    if cfg.route is None:
        return synthetic_episodes(cfg.episodes, rng=spec_rng)
    if cfg.route.kind == "row_alley":
        # The stage labels the episodes; the world's actual stage is fixed at
        # launch, and must match what the sim section says.
        stage = PhenologyStage[str(cfg.sim.params.get("stage", "VEGETATIVE"))]
        return row_alley_episodes(cfg.episodes, rng=spec_rng, stage=stage, **cfg.route.params)
    msg = f"unknown route kind {cfg.route.kind!r}; expected 'row_alley'"
    raise ConfigError(msg)


def run_experiment(cfg: ExperimentConfig, *, run_id: str = "") -> RunResult:
    """Build the assembly, run every episode, and summarise.

    Episode seeds are derived from the master seed by a fixed rule
    (``seed + index``) so that adding an episode never changes the behaviour of
    the ones before it.
    """
    assembly = build(cfg)
    specs = _episodes(cfg)

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
