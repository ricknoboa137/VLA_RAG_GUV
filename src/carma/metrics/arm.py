"""Arm-policy metrics: how well a manipulator VLA's chunks match recorded actions.

The arm is outside the arbitration and the five experimental conditions
(``docs/arm.md``), so nothing here enters :class:`~carma.metrics.MetricBundle`.
These numbers compare arm policies with each other -- SmolVLA against X-VLA --
and live here because every reported number does.

Open-loop means each prediction is scored against what the demonstrator did
next, from the demonstrator's own observation. It measures imitation of the
recording, not task success: a policy can drift from the demonstration and
still pick the cube up, or match it closely and miss. Closed-loop success on
the arm is a separate measurement that needs the arm or its simulation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def mean_joint_error(
    predicted_rad: FloatArray, truth_rad: FloatArray, valid: npt.NDArray[np.bool_]
) -> float:
    """Mean absolute joint-angle error over the valid steps of one chunk, in radians.

    Args:
        predicted_rad: ``(steps, joints)`` predicted joint targets.
        truth_rad: ``(steps, joints)`` recorded joint targets, the same steps.
        valid: ``(steps,)`` True where the recording has that step; the last
            chunks of an episode run past its end.

    Returns:
        The mean over valid steps and all joints; NaN when no step is valid,
        so an empty comparison cannot pass for a perfect one.
    """
    if predicted_rad.shape != truth_rad.shape or predicted_rad.shape[0] != valid.shape[0]:
        msg = (
            f"shapes disagree: predicted {predicted_rad.shape}, truth {truth_rad.shape}, "
            f"valid {valid.shape}"
        )
        raise ValueError(msg)
    if not valid.any():
        return float("nan")
    return float(np.abs(predicted_rad[valid] - truth_rad[valid]).mean())


def _average_ranks(values: FloatArray) -> FloatArray:
    """Ranks from 1, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation between two paired samples, in ``[-1, 1]``.

    Used for whether a policy's uncertainty orders its errors: +1 means the
    frames it was least sure of were exactly the ones it got most wrong.
    NaN for fewer than two pairs or a constant sample, where no ordering
    exists.
    """
    if len(x) != len(y):
        msg = f"paired samples differ in length: {len(x)} and {len(y)}"
        raise ValueError(msg)
    if len(x) < 2:
        return float("nan")
    rx = _average_ranks(np.asarray(x, dtype=np.float64))
    ry = _average_ranks(np.asarray(y, dtype=np.float64))
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def latency_summary(latencies_s: Sequence[float]) -> dict[str, float]:
    """Mean, median and 95th percentile of measured inference times, in seconds."""
    if not latencies_s:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    arr = np.asarray(latencies_s, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
