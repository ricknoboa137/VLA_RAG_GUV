"""Interaction-economy metrics.

This family is the project's distinctive measurement. The retrieval literature
claims to reduce operator load and does not report it; these functions are how
that claim is tested rather than asserted.

Every quantity here is defined per unit of *work done*, never per episode, so
that a method cannot look cheap simply by giving up early.
"""

from __future__ import annotations

from collections.abc import Sequence

from carma.types.core import EpisodeRecord

_SQUARE_METRES_PER_HECTARE = 10_000.0


def _total_path_length(episodes: Sequence[EpisodeRecord]) -> float:
    return sum(e.path_length for e in episodes)


def queries_per_100m(episodes: Sequence[EpisodeRecord]) -> float:
    """Operator queries issued per hundred metres travelled.

    The headline interaction measure. Lower is better only when task success is
    held level, which is why it is never reported on its own.
    """
    distance = _total_path_length(episodes)
    if distance <= 0.0:
        return 0.0
    asks = sum(1 for e in episodes for s in e.steps if s.asked)
    return 100.0 * asks / distance


def interventions_per_hectare(
    episodes: Sequence[EpisodeRecord],
    swath_width_m: float,
) -> float:
    """Operator queries per hectare covered, given an implement swath width.

    The unit an agronomist recognises. ``swath_width_m`` is a property of the
    machine, not of the algorithm, so it is passed in rather than assumed.
    """
    if swath_width_m <= 0.0:
        msg = "swath_width_m must be positive"
        raise ValueError(msg)
    area = _total_path_length(episodes) * swath_width_m
    if area <= 0.0:
        return 0.0
    asks = sum(1 for e in episodes for s in e.steps if s.asked)
    return asks * _SQUARE_METRES_PER_HECTARE / area


def mean_neglect_time(episodes: Sequence[EpisodeRecord]) -> float:
    """Mean seconds of autonomous operation between operator queries.

    The complement of query frequency, and the quantity that determines how
    many robots one person can supervise.
    """
    gaps: list[float] = []
    for episode in episodes:
        last_ask_t = 0.0
        for step in episode.steps:
            if step.asked:
                gaps.append(step.t - last_ask_t)
                last_ask_t = step.t
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)


def operator_latency_total(episodes: Sequence[EpisodeRecord]) -> float:
    """Total seconds the robot spent waiting on a human, across all episodes."""
    return sum(s.operator_latency_s for e in episodes for s in e.steps)
