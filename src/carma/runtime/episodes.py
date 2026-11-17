"""Episode generation.

Synthetic episodes exist so the harness has something to run in CI. Real
episodes are loaded from a route file recorded in the field; that loader lands
with the first campaign and must produce the same :class:`EpisodeSpec` type.
"""

from __future__ import annotations

import numpy as np

from carma.types.core import EpisodeSpec, Hazard, PhenologyStage, Pose


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


# The row-crop world (ros2_ws/src/carma_sim/worlds/row_crop.sdf) puts the robot
# at the origin facing +x in a clear alley centred on y = 0, with crop segments
# from x = 2 m. Its furniture sits in the y = +/-1.5 m alleys, so a straight
# drive along y = 0 is the task and the alley's width is the difficulty.
ROW_ALLEY_START_X_M = 0.0


def row_alley_episodes(
    count: int,
    *,
    rng: np.random.Generator,
    stage: PhenologyStage = PhenologyStage.VEGETATIVE,
    min_length_m: float = 10.0,
    max_length_m: float = 20.0,
    start_offset_m: float = 0.25,
    start_yaw_rad: float = 0.2,
    success_radius_m: float = 1.5,
    max_steps: int = 400,
    hazard_probability: float = 0.0,
    hazard_keep_out_m: float = 1.0,
    hazard_stop_before_m: float = 2.0,
    hazard_min_x_m: float = 6.0,
    hazard_x_m: float | None = None,
) -> tuple[EpisodeSpec, ...]:
    """Generate ``count`` reproducible down-the-alley episodes from ``rng``.

    The goal is on the alley centreline, ``length`` whole metres from the origin
    with ``length`` drawn uniformly from ``[min_length_m, max_length_m]`` and
    rounded. The instruction names the distance ("drive 14 metres down the row
    and stop"), because a policy cannot know where to stop otherwise; the goal
    itself is never given to it. The robot starts
    with a lateral offset in ``[-start_offset_m, start_offset_m]`` and a
    heading error in ``[-start_yaw_rad, start_yaw_rad]``, so the policy has to
    correct itself rather than replay one straight drive. ``shortest_path_length``
    is the straight line from the start to the goal.

    With ``hazard_probability`` > 0, each episode has that chance of a hidden
    hazard (:class:`~carma.types.core.Hazard`) on the centreline, at a whole
    metre in ``[hazard_min_x_m, length - 1]``. The instruction still names the
    full distance; the goal moves to the stop point ``hazard_stop_before_m``
    before the hazard, and the hazard's advice says so ("wet patch at 8
    metres: stop at 6 metres"). Only the operator, asked or remembered, knows
    it. ``hazard_x_m`` fixes the hazard at one place for every episode that
    has it, the way a wet patch stays where it is from one pass to the next;
    that is what lets memory from an earlier pass help a later one. At
    probability 0 no extra numbers are drawn, so existing configs keep
    producing exactly the episodes they did.
    """
    specs: list[EpisodeSpec] = []
    for i in range(count):
        length = float(round(rng.uniform(min_length_m, max_length_m)))
        start = Pose(
            ROW_ALLEY_START_X_M,
            float(rng.uniform(-start_offset_m, start_offset_m)),
            float(rng.uniform(-start_yaw_rad, start_yaw_rad)),
        )
        goal = Pose(length, 0.0, 0.0)
        hazards: tuple[Hazard, ...] = ()
        if hazard_probability > 0.0 and rng.random() < hazard_probability:
            if hazard_x_m is not None:
                hx = float(hazard_x_m)
            else:
                hx = float(rng.integers(int(hazard_min_x_m), int(length)))  # up to length - 1
            stop = hx - hazard_stop_before_m
            hazards = (
                Hazard(
                    position=Pose(hx, 0.0, 0.0),
                    keep_out_m=hazard_keep_out_m,
                    advice=f"wet patch at {hx:.0f} metres: stop at {stop:.0f} metres",
                ),
            )
            goal = Pose(stop, 0.0, 0.0)
        specs.append(
            EpisodeSpec(
                episode_id=f"row-{i:03d}",
                instruction=f"drive {length:.0f} metres down the row and stop",
                start=start,
                goal=goal,
                success_radius=success_radius_m,
                shortest_path_length=start.distance_to(goal),
                stage=stage,
                max_steps=max_steps,
                hazards=hazards,
            )
        )
    return tuple(specs)
