"""Embedded-feasibility metrics.

These establish that the loop is deployable rather than only simulable. Power
and energy are measured on the platform by ``scripts/profile_power.py``; this
module covers only what can be derived from step records.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from carma.types.core import EpisodeRecord


def decision_latency_summary(episodes: Sequence[EpisodeRecord]) -> dict[str, float]:
    """Wall-clock decision latency in seconds.

    Returns mean, median and the 95th percentile. The tail matters more than
    the mean: a policy whose worst case exceeds the control period is not
    deployable however good its average.
    """
    latencies = [s.operator_latency_s for e in episodes for s in e.steps if s.asked]
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
