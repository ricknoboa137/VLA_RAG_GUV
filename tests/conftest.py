"""Shared fixtures. No network, no GPU, no simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carma.config import ExperimentConfig, load
from carma.types.core import (
    Action,
    Choice,
    Decision,
    EpisodeRecord,
    EpisodeSpec,
    Observation,
    PhenologyStage,
    Pose,
    StepRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator. Nothing in the suite uses global randomness."""
    return np.random.default_rng(1234)


@pytest.fixture
def smoke_config() -> ExperimentConfig:
    """The smoke experiment config, loaded from disk."""
    return load(REPO_ROOT / "configs" / "experiment" / "smoke.yaml")


@pytest.fixture
def observation() -> Observation:
    """A small synthetic observation with a vegetated lower half."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:30] = (200, 190, 175)
    img[30:] = (40, 130, 60)
    return Observation(
        rgb=img,
        depth=None,
        pose=Pose(0.0, 0.0, 0.0),
        t=0.0,
        instruction="follow the row and stop at the headland",
        stage=PhenologyStage.VEGETATIVE,
    )


def make_episode(
    *,
    final: Pose,
    goal: Pose,
    path_length: float,
    asks: int = 0,
    shortest: float = 10.0,
    success_radius: float = 2.0,
) -> EpisodeRecord:
    """Build an episode record with hand-chosen values, for metric tests."""
    decision = Decision(
        choice=Choice.ACT,
        costs={Choice.ACT: 0.0, Choice.RETRIEVE: 1.0, Choice.ASK: 2.0},
    )
    steps = tuple(
        StepRecord(
            step=i,
            t=float(i),
            pose=Pose(0.0, 0.0, 0.0),
            action=Action(0.5, 0.0),
            decision=decision,
            uncertainty=0.5,
            asked=i < asks,
            operator_latency_s=10.0 if i < asks else 0.0,
        )
        for i in range(max(asks, 1))
    )
    spec = EpisodeSpec(
        episode_id="t",
        instruction="i",
        start=Pose(0.0, 0.0, 0.0),
        goal=goal,
        success_radius=success_radius,
        shortest_path_length=shortest,
        stage=PhenologyStage.VEGETATIVE,
    )
    return EpisodeRecord(
        spec=spec,
        steps=steps,
        final_pose=final,
        path_length=path_length,
        wall_time_s=1.0,
    )
