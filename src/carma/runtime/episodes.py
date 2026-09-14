"""Episode generation.

Synthetic episodes exist so the harness has something to run in CI. Real
episodes are loaded from a route file recorded in the field; that loader lands
with the first campaign and must produce the same :class:`EpisodeSpec` type.
"""

from __future__ import annotations

import numpy as np

from carma.types.core import EpisodeSpec, PhenologyStage, Pose


def synthetic_episodes(
    count: int,
    *,
    rng: np.random.Generator,
    stage: PhenologyStage = PhenologyStage.VEGETATIVE,
    max_steps: int = 200,
) -> tuple[EpisodeSpec, ...]:
    """Generate ``count`` reproducible episodes from ``rng``.

    Goals are placed on a ring so that shortest-path length is known exactly,
    which makes SPL meaningful without a planner.
    """
    specs: list[EpisodeSpec] = []
    for i in range(count):
        radius = float(rng.uniform(8.0, 20.0))
        bearing = float(rng.uniform(-np.pi, np.pi))
        goal = Pose(radius * float(np.cos(bearing)), radius * float(np.sin(bearing)), 0.0)
        specs.append(
            EpisodeSpec(
                episode_id=f"syn-{i:03d}",
                instruction="follow the row to the end and stop at the headland",
                start=Pose(0.0, 0.0, 0.0),
                goal=goal,
                success_radius=2.0,
                shortest_path_length=radius,
                stage=stage,
                max_steps=max_steps,
            )
        )
    return tuple(specs)
