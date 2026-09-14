"""One struct carrying every reported number for a set of episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from carma.metrics.interaction import (
    interventions_per_hectare,
    mean_neglect_time,
    operator_latency_total,
    queries_per_100m,
)
from carma.metrics.retrieval import retrieval_rate
from carma.metrics.systems import decision_latency_summary
from carma.metrics.task import navigation_error, spl, success_rate
from carma.types.core import EpisodeRecord

DEFAULT_SWATH_M = 2.0


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """Every number reported for one condition.

    Adding a field here means adding it to ``summarise`` and to
    ``tests/test_metrics.py`` with a hand-computed expected value. A metric
    with no hand-computed test is not trusted.
    """

    episodes: int
    success_rate: float
    navigation_error_m: float
    spl: float
    queries_per_100m: float
    interventions_per_hectare: float
    mean_neglect_time_s: float
    operator_latency_total_s: float
    retrieval_rate: float
    decision_latency_s: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary, suitable for YAML or JSON serialisation."""
        return asdict(self)


def summarise(
    episodes: Sequence[EpisodeRecord],
    swath_width_m: float = DEFAULT_SWATH_M,
) -> MetricBundle:
    """Compute every metric for a set of episodes."""
    return MetricBundle(
        episodes=len(episodes),
        success_rate=success_rate(episodes),
        navigation_error_m=navigation_error(episodes),
        spl=spl(episodes),
        queries_per_100m=queries_per_100m(episodes),
        interventions_per_hectare=interventions_per_hectare(episodes, swath_width_m),
        mean_neglect_time_s=mean_neglect_time(episodes),
        operator_latency_total_s=operator_latency_total(episodes),
        retrieval_rate=retrieval_rate(episodes),
        decision_latency_s=decision_latency_summary(episodes),
    )
