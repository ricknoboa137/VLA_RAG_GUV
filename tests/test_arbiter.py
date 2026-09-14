"""The scientific core: pricing and choosing."""

from __future__ import annotations

import pytest

from carma.arbiter.cost_model import LinearCostModel
from carma.arbiter.policies import AlwaysAct, AlwaysRetrieve, UncertaintyAsk
from carma.types.core import Choice, Observation


def test_acting_is_free_when_certain() -> None:
    """Zero uncertainty means no expected recovery cost."""
    model = LinearCostModel(recovery_s=45.0)
    assert model.cost_act(0.0) == pytest.approx(0.0)
    assert model.cost_act(1.0) == pytest.approx(45.0)


def test_retrieval_without_a_match_removes_no_risk() -> None:
    """A retriever that holds nothing costs latency and saves nothing.

    This is what stops unconditional retrieval from looking free.
    """
    model = LinearCostModel(recovery_s=45.0, retrieval_s=0.1, retrieval_efficacy=0.6)
    barren = model.cost_retrieve(uncertainty=1.0, expected_gain=0.0)
    assert barren == pytest.approx(0.1 + 45.0)


def test_retrieval_with_a_strong_match_beats_acting() -> None:
    """A confident, relevant hit should price below acting blind."""
    model = LinearCostModel(recovery_s=45.0, retrieval_s=0.1, retrieval_efficacy=0.6)
    assert model.cost_retrieve(1.0, 1.0) < model.cost_act(1.0)


def test_asking_costs_the_operator_even_when_certain() -> None:
    """The interruption is paid in full regardless of uncertainty.

    Scaling the operator term by uncertainty would let a confident-but-wrong
    policy ask for free, which is exactly the failure the project measures.
    """
    model = LinearCostModel(operator_s=20.0)
    assert model.cost_ask(0.0) == pytest.approx(20.0)


def test_baseline_arbiters_are_unconditional(observation: Observation) -> None:
    """The two baselines ignore uncertainty entirely."""
    assert AlwaysAct().decide(observation, 0.99).choice is Choice.ACT
    assert AlwaysRetrieve().decide(observation, 0.01).choice is Choice.RETRIEVE


def test_uncertainty_ask_respects_its_threshold(observation: Observation) -> None:
    """Below the threshold it acts; at or above it asks."""
    policy = UncertaintyAsk(threshold=0.5)
    assert policy.decide(observation, 0.49).choice is Choice.ACT
    assert policy.decide(observation, 0.50).choice is Choice.ASK
