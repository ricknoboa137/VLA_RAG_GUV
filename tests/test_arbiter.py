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
    barren = model.cost_retrieve(risk=1.0, expected_gain=0.0)
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


def test_risk_is_the_uncertainty_when_memory_risk_is_zero() -> None:
    """The default reproduces the original model: risk = u, whatever memory holds."""
    model = LinearCostModel()
    assert model.risk(0.3, 0.9) == pytest.approx(0.3)
    assert model.risk(0.0, 1.0) == 0.0


def test_remembered_warnings_raise_the_risk_of_acting_blind() -> None:
    """rho 0.5, gain 0.5: risk max(0.0, 0.25) = 0.25. With R 45, L 0.12, e_r 0.6,
    A 20, e_a 0.95: act 11.25, retrieve 0.12 + 0.25 * 0.7 * 45 = 7.995,
    ask 20 + 0.25 * 0.05 * 45 = 20.5625. Retrieve is cheapest."""
    model = LinearCostModel(memory_risk=0.5)
    risk = model.risk(0.0, 0.5)
    assert risk == pytest.approx(0.25)
    assert model.cost_act(risk) == pytest.approx(11.25)
    assert model.cost_retrieve(risk, 0.5) == pytest.approx(7.995)
    assert model.cost_ask(risk) == pytest.approx(20.5625)
    # A confident backbone with nothing in memory still acts for free.
    assert model.risk(0.0, 0.0) == 0.0
    # The larger source wins; they are not added.
    assert model.risk(0.4, 0.5) == pytest.approx(0.4)


def test_memory_risk_must_be_a_probability() -> None:
    with pytest.raises(ValueError):
        LinearCostModel(memory_risk=1.5)
