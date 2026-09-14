"""Task-performance metrics.

Defined to match the published agricultural vision-and-language navigation
baselines exactly, so that numbers from this project can be placed beside them
without a footnote.
"""

from __future__ import annotations

from collections.abc import Sequence

from carma.types.core import EpisodeRecord


def success_rate(episodes: Sequence[EpisodeRecord]) -> float:
    """Fraction of episodes finishing inside the goal radius, in ``[0, 1]``."""
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.success) / len(episodes)


def navigation_error(episodes: Sequence[EpisodeRecord]) -> float:
    """Mean metres between final pose and goal, over all episodes.

    Reported over all episodes including successes, which is the convention the
    baselines use; reporting it over failures only would flatter the method.
    """
    if not episodes:
        return 0.0
    return sum(e.navigation_error for e in episodes) / len(episodes)


def spl(episodes: Sequence[EpisodeRecord]) -> float:
    """Success weighted by normalised inverse path length, in ``[0, 1]``.

    An episode that fails contributes zero. An episode that succeeds by a
    wandering route contributes less than one.
    """
    if not episodes:
        return 0.0
    total = 0.0
    for e in episodes:
        if not e.success:
            continue
        travelled = max(e.path_length, 1e-6)
        shortest = max(e.spec.shortest_path_length, 1e-6)
        total += shortest / max(travelled, shortest)
    return total / len(episodes)
