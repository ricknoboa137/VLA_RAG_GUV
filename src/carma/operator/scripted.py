"""A simulated operator for reproducible experiments.

Answers from a fixed script with a sampled latency, and volunteers corrections
at a configurable rate. Its purpose is to make the simulation study
reproducible; it is not a model of a person, and no claim about human behaviour
may rest on it. The field campaigns use a live channel instead.
"""

from __future__ import annotations

import numpy as np

from carma.operator.registry import OPERATORS
from carma.types.core import (
    Correction,
    EpisodeSpec,
    Hazard,
    Observation,
    OperatorQuery,
    OperatorResponse,
)

_DEFAULT_ANSWERS = (
    "keep to the left of the row",
    "the ground ahead is soft, go right",
    "that gap is passable, continue straight",
    "stop and wait, the row is blocked",
)

#: What the scripted operator says when nothing specific is ahead. Public so the
#: demonstration recorder can train a policy to drive on through such remarks.
DEFAULT_ANSWERS = _DEFAULT_ANSWERS

_DEFAULT_CORRECTIONS = (
    "stay left when the canopy narrows",
    "avoid the wet patch near the headland",
    "slow down before the turn at the row end",
)


@OPERATORS.register("scripted")
class ScriptedOperator:
    """Deterministic stand-in for a human operator.

    Args:
        mean_latency_s: Mean of the log-normal response latency.
        sigma: Shape parameter of that log-normal.
        refusal_rate: Probability the operator does not answer at all.
        correction_rate: Per-step probability of volunteering a correction.
        operator_id: Recorded as entry provenance, so per-operator effects stay
            separable in analysis.
        seed: Seed for this operator's own generator.
        advice_range_m: A hidden hazard this many metres ahead or less is what
            the operator talks about when asked, instead of a scripted answer.
        hazard_warning_rate: Probability, per hazard per episode, that the
            operator volunteers its advice unasked as the robot comes within
            ``advice_range_m``. Those warnings are what fills memory for later
            passes. Zero keeps the operator's draws exactly as before hazards
            existed.

    Hazard knowledge is an oracle: the operator is told the episode's hazards
    (``begin_episode``) the way a person who works the field knows them. It
    still answers only after its sampled latency, and still refuses at
    ``refusal_rate``.
    """

    def __init__(
        self,
        mean_latency_s: float = 18.0,
        sigma: float = 0.4,
        refusal_rate: float = 0.05,
        correction_rate: float = 0.04,
        operator_id: str = "scripted-0",
        seed: int = 0,
        advice_range_m: float = 6.0,
        hazard_warning_rate: float = 0.0,
    ) -> None:
        self._mean = mean_latency_s
        self._sigma = sigma
        self._refusal_rate = refusal_rate
        self._correction_rate = correction_rate
        self._operator_id = operator_id
        self._rng = np.random.default_rng(seed)
        self._advice_range_m = advice_range_m
        self._warning_rate = hazard_warning_rate
        self._hazards: tuple[Hazard, ...] = ()
        self._will_warn: dict[int, bool] = {}

    @property
    def operator_id(self) -> str:
        """Provenance recorded on everything this operator says."""
        return self._operator_id

    def begin_episode(self, spec: EpisodeSpec) -> None:
        """Learn the episode's hidden hazards and decide which to warn about."""
        self._hazards = spec.hazards
        self._will_warn = {
            i: bool(self._rng.random() < self._warning_rate) for i in range(len(spec.hazards))
        }

    def _hazard_ahead(self, obs: Observation) -> int | None:
        """Index of the nearest hazard ahead within ``advice_range_m``, if any."""
        ahead = [
            (h.position.x - obs.pose.x, i)
            for i, h in enumerate(self._hazards)
            if 0.0 < h.position.x - obs.pose.x <= self._advice_range_m
        ]
        return min(ahead)[1] if ahead else None

    def ask(self, query: OperatorQuery) -> OperatorResponse:
        """Answer a query after a sampled delay."""
        latency = float(self._rng.lognormal(np.log(self._mean), self._sigma))
        if self._rng.random() < self._refusal_rate:
            return OperatorResponse(text="", latency_s=latency, refused=True)
        idx = int(self._rng.integers(0, len(_DEFAULT_ANSWERS)))
        hazard = self._hazard_ahead(query.observation)
        text = _DEFAULT_ANSWERS[idx] if hazard is None else self._hazards[hazard].advice
        return OperatorResponse(text=text, latency_s=latency, refused=False)

    def poll_corrections(self, obs: Observation) -> tuple[Correction, ...]:
        """Warn about a hazard ahead when decided to, else occasionally volunteer a correction."""
        hazard = self._hazard_ahead(obs)
        if hazard is not None and self._will_warn.get(hazard, False):
            self._will_warn[hazard] = False  # once per hazard per episode
            return (
                Correction(
                    text=self._hazards[hazard].advice,
                    observation=obs,
                    operator_id=self._operator_id,
                    took_over=False,
                ),
            )
        if self._rng.random() >= self._correction_rate:
            return ()
        idx = int(self._rng.integers(0, len(_DEFAULT_CORRECTIONS)))
        return (
            Correction(
                text=_DEFAULT_CORRECTIONS[idx],
                observation=obs,
                operator_id=self._operator_id,
                took_over=False,
            ),
        )
