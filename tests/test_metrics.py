"""Every metric is checked against a hand-computed value.

A metric with no hand-computed test is not trusted; see AGENTS.md section 8.
"""

from __future__ import annotations

import math

import pytest
from tests.conftest import make_episode

from carma.metrics import (
    interventions_per_hectare,
    mean_neglect_time,
    navigation_error,
    precision_at_k,
    queries_per_100m,
    spl,
    staleness_rate,
    success_rate,
)
from carma.types.core import Pose


def test_success_rate_counts_goal_radius() -> None:
    """One episode inside the radius, one outside, gives 0.5."""
    inside = make_episode(final=Pose(1.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    outside = make_episode(final=Pose(9.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    assert success_rate([inside, outside]) == pytest.approx(0.5)


def test_navigation_error_is_the_mean_over_all_episodes() -> None:
    """Distances 1.0 and 9.0 average to 5.0, successes included."""
    a = make_episode(final=Pose(1.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    b = make_episode(final=Pose(9.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    assert navigation_error([a, b]) == pytest.approx(5.0)


def test_spl_penalises_a_wandering_success() -> None:
    """Shortest 10 m, travelled 20 m, one episode: SPL is 0.5."""
    wandering = make_episode(
        final=Pose(0.5, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=20.0, shortest=10.0
    )
    assert spl([wandering]) == pytest.approx(0.5)


def test_spl_is_zero_for_failure() -> None:
    """A failed episode contributes nothing however short its path."""
    failed = make_episode(
        final=Pose(50.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=5.0, shortest=10.0
    )
    assert spl([failed]) == pytest.approx(0.0)


def test_queries_per_100m() -> None:
    """Four asks over 200 m is two per hundred metres."""
    ep = make_episode(
        final=Pose(0.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=200.0, asks=4
    )
    assert queries_per_100m([ep]) == pytest.approx(2.0)


def test_interventions_per_hectare() -> None:
    """1000 m at a 2 m swath is 2000 m^2; two asks is ten per hectare."""
    ep = make_episode(
        final=Pose(0.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=1000.0, asks=2
    )
    assert interventions_per_hectare([ep], swath_width_m=2.0) == pytest.approx(10.0)


def test_interventions_per_hectare_rejects_bad_swath() -> None:
    """A non-positive swath is a caller error, not a silent zero."""
    ep = make_episode(final=Pose(0.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0)
    with pytest.raises(ValueError, match="swath_width_m"):
        interventions_per_hectare([ep], swath_width_m=0.0)


def test_mean_neglect_time() -> None:
    """Asks at t=0,1,2 give gaps 0,1,1 and a mean of 2/3."""
    ep = make_episode(final=Pose(0.0, 0.0, 0.0), goal=Pose(0.0, 0.0, 0.0), path_length=10.0, asks=3)
    assert mean_neglect_time([ep]) == pytest.approx(2.0 / 3.0)


def test_precision_at_k() -> None:
    """One of the top two is relevant in each query: precision 0.5."""
    retrieved = [["a", "b"], ["c", "d"]]
    relevant = [{"a"}, {"c"}]
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)


def test_precision_at_k_validates_inputs() -> None:
    """Mismatched lengths and non-positive k are errors."""
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k([["a"]], [{"a"}], k=0)
    with pytest.raises(ValueError, match="equal length"):
        precision_at_k([["a"]], [{"a"}, {"b"}], k=1)


def test_staleness_rate() -> None:
    """Two of four retrieved entries are stale."""
    assert staleness_rate(["a", "b", "c", "d"], {"a", "c"}) == pytest.approx(0.5)


def test_empty_inputs_are_zero_not_nan() -> None:
    """Empty sequences return 0.0, never NaN, so reports never break."""
    for value in (success_rate([]), navigation_error([]), spl([]), queries_per_100m([])):
        assert not math.isnan(value)
        assert value == pytest.approx(0.0)
