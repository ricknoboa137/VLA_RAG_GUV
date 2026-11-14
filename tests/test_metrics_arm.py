"""Arm-policy metrics, each against a hand-computed value (AGENTS.md section 8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from carma.metrics.arm import latency_summary, mean_joint_error, rank_correlation


def test_mean_joint_error_over_valid_steps_only() -> None:
    predicted = np.array([[0.1, 0.0], [0.3, -0.2], [9.0, 9.0]])
    truth = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    valid = np.array([True, True, False])
    # |0.1| + |0| + |0.3| + |-0.2| = 0.6 over 4 values = 0.15; the padded step is ignored.
    assert mean_joint_error(predicted, truth, valid) == pytest.approx(0.15)


def test_mean_joint_error_with_nothing_valid_is_not_zero() -> None:
    assert math.isnan(mean_joint_error(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros(2, bool)))


def test_mean_joint_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="disagree"):
        mean_joint_error(np.zeros((2, 6)), np.zeros((3, 6)), np.ones(2, bool))


def test_rank_correlation_of_a_perfect_ordering() -> None:
    assert rank_correlation([0.1, 0.5, 0.9], [1.0, 2.0, 30.0]) == pytest.approx(1.0)
    assert rank_correlation([0.1, 0.5, 0.9], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_rank_correlation_with_ties() -> None:
    # x ranks (1, 2.5, 2.5, 4), y ranks (1, 2, 3, 4).
    # Centred: x (-1.5, 0, 0, 1.5), y (-1.5, -0.5, 0.5, 1.5).
    # Covariance sum 4.5; sqrt(4.5 * 5) = 4.7434; r = 0.948683.
    assert rank_correlation([1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(
        0.948683, abs=1e-6
    )


def test_rank_correlation_is_undefined_without_an_ordering() -> None:
    assert math.isnan(rank_correlation([1.0], [2.0]))
    assert math.isnan(rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="length"):
        rank_correlation([1.0, 2.0], [1.0])


def test_latency_summary() -> None:
    summary = latency_summary([0.2, 0.4, 0.6, 0.8, 1.0])
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["p50"] == pytest.approx(0.6)
    # numpy's linear percentile: 0.8 + 0.8 * (1.0 - 0.8) = 0.96
    assert summary["p95"] == pytest.approx(0.96)
    assert math.isnan(latency_summary([])["mean"])
