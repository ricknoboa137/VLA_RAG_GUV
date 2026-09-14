"""The control loop.

One function, deliberately. Everything that differs between experimental
conditions is a component handed in by :func:`carma.config.build`, so this loop
is identical across all five conditions. If a change to this file needs an
``if condition ==`` branch, the change belongs in an arbiter instead.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from carma.config import Assembly
from carma.types.core import (
    Choice,
    EpisodeRecord,
    EpisodeSpec,
    Observation,
    OperatorQuery,
    StepRecord,
)

_log = structlog.get_logger(__name__)

_RETRIEVE_K = 3
_REVIEW_BUDGET = 1


def _maybe_reset(component: object) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def run_episode(
    spec: EpisodeSpec,
    assembly: Assembly,
    *,
    seed: int,
    run_id: str = "",
) -> EpisodeRecord:
    """Run one episode and return its record.

    The loop is: observe, poll for volunteered corrections, ask the backbone for
    an action and an uncertainty, let the arbiter price the three options, act on
    its choice, record everything.

    Args:
        spec: The episode to run.
        assembly: Constructed components for this condition.
        seed: Episode seed. Derived from the master seed by the caller so that
            episode ``i`` is reproducible independently of the others.
        run_id: Recorded in the decision log.
    """
    rng = np.random.default_rng(seed)
    _maybe_reset(assembly.backbone)
    _maybe_reset(assembly.arbiter)

    obs: Observation = assembly.sim.reset(seed=seed)
    steps: list[StepRecord] = []
    path_length = 0.0
    previous_pose = obs.pose
    started = time.perf_counter()

    for step_idx in range(spec.max_steps):
        for correction in assembly.operator.poll_corrections(obs):
            assembly.retriever.ingest(correction)

        action, uncertainty = assembly.backbone.act(obs, (), rng=rng)
        decision = assembly.arbiter.decide(obs, uncertainty)

        retrieved_ids: tuple[str, ...] = ()
        asked = False
        operator_latency = 0.0

        if decision.choice is Choice.RETRIEVE:
            result = assembly.retriever.retrieve(obs, _RETRIEVE_K)
            live = tuple(e for e in result.entries if assembly.hygiene.staleness(e, obs) < 0.75)
            retrieved_ids = tuple(e.entry_id for e in live)
            if live:
                action, uncertainty = assembly.backbone.act(
                    obs, tuple(e.text for e in live), rng=rng
                )

        elif decision.choice is Choice.ASK:
            response = assembly.operator.ask(
                OperatorQuery(question="which way at this point?", observation=obs)
            )
            asked = True
            operator_latency = response.latency_s
            note_ask = getattr(assembly.arbiter, "note_ask", None)
            if callable(note_ask):
                note_ask()
            if not response.refused:
                action, uncertainty = assembly.backbone.act(obs, (response.text,), rng=rng)
                for entry in assembly.hygiene.select_for_review(
                    assembly.store.all(), obs, _REVIEW_BUDGET
                ):
                    assembly.store.invalidate(entry.entry_id)

        steps.append(
            StepRecord(
                step=step_idx,
                t=obs.t,
                pose=obs.pose,
                action=action,
                decision=decision,
                uncertainty=uncertainty,
                retrieved_ids=retrieved_ids,
                asked=asked,
                operator_latency_s=operator_latency,
            )
        )
        _log.debug(
            "decision",
            run_id=run_id,
            episode_id=spec.episode_id,
            step=step_idx,
            choice=decision.choice.value,
            costs={c.value: v for c, v in decision.costs.items()},
            uncertainty=uncertainty,
        )

        if action.stop or obs.pose.distance_to(spec.goal) <= spec.success_radius:
            break

        obs = assembly.sim.step(action)
        path_length += obs.pose.distance_to(previous_pose)
        previous_pose = obs.pose

    return EpisodeRecord(
        spec=spec,
        steps=tuple(steps),
        final_pose=obs.pose,
        path_length=path_length,
        wall_time_s=time.perf_counter() - started,
    )
