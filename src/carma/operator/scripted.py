"""A simulated operator for reproducible experiments.

Answers from a fixed script with a sampled latency, and volunteers corrections
at a configurable rate. Its purpose is to make the simulation study
reproducible; it is not a model of a person, and no claim about human behaviour
may rest on it. The field campaigns use a live channel instead.
"""

from __future__ import annotations

import numpy as np

from carma.operator.registry import OPERATORS
from carma.types.core import Correction, Observation, OperatorQuery, OperatorResponse

_DEFAULT_ANSWERS = (
    "keep to the left of the row",
    "the ground ahead is soft, go right",
    "that gap is passable, continue straight",
    "stop and wait, the row is blocked",
)

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
    """

    def __init__(
        self,
        mean_latency_s: float = 18.0,
        sigma: float = 0.4,
        refusal_rate: float = 0.05,
        correction_rate: float = 0.04,
        operator_id: str = "scripted-0",
        seed: int = 0,
    ) -> None:
        self._mean = mean_latency_s
        self._sigma = sigma
        self._refusal_rate = refusal_rate
        self._correction_rate = correction_rate
        self._operator_id = operator_id
        self._rng = np.random.default_rng(seed)

    def ask(self, query: OperatorQuery) -> OperatorResponse:
        """Answer a query after a sampled delay."""
        latency = float(self._rng.lognormal(np.log(self._mean), self._sigma))
        if self._rng.random() < self._refusal_rate:
            return OperatorResponse(text="", latency_s=latency, refused=True)
        idx = int(self._rng.integers(0, len(_DEFAULT_ANSWERS)))
        return OperatorResponse(text=_DEFAULT_ANSWERS[idx], latency_s=latency, refused=False)

    def poll_corrections(self, obs: Observation) -> tuple[Correction, ...]:
        """Occasionally volunteer a correction."""
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
