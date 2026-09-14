"""Pricing the three options in one currency.

The currency is **expected additional traverse time in seconds**. Choosing a
single unit is what makes operator attention, retrieval latency and the risk of
acting wrongly comparable at all; the reasoning is set out in
``docs/cost-model.md``, and any change here requires that document to change in
the same commit.
"""

from __future__ import annotations

from carma.registry import Registry
from carma.types import CostModel

COST_MODELS: Registry[CostModel] = Registry("cost_model")


@COST_MODELS.register("linear")
class LinearCostModel:
    """Linear-in-uncertainty pricing with fixed overheads.

    Args:
        recovery_s: Seconds lost to a recovery traverse when the robot acts
            wrongly. Measured on the platform, not guessed: see
            ``scripts/measure_recovery.py``.
        retrieval_s: Wall-clock cost of one retrieval, including embedding.
        operator_s: Expected seconds for an operator to answer, including the
            time they spend re-orienting to the robot's situation. This is
            deliberately larger than raw response latency, because interrupting
            a person costs more than the words they say.
        retrieval_efficacy: Fraction of the acting risk that a good retrieval
            removes. Estimated from held-out data, never assumed to be 1.0.
        ask_efficacy: Fraction of the acting risk an operator answer removes.
    """

    def __init__(
        self,
        recovery_s: float = 45.0,
        retrieval_s: float = 0.12,
        operator_s: float = 20.0,
        retrieval_efficacy: float = 0.6,
        ask_efficacy: float = 0.95,
    ) -> None:
        self._recovery_s = recovery_s
        self._retrieval_s = retrieval_s
        self._operator_s = operator_s
        self._retrieval_efficacy = retrieval_efficacy
        self._ask_efficacy = ask_efficacy

    def cost_act(self, uncertainty: float) -> float:
        """Expected seconds lost by acting on the backbone's output directly."""
        return uncertainty * self._recovery_s

    def cost_retrieve(self, uncertainty: float, expected_gain: float) -> float:
        """Expected seconds when retrieving first.

        ``expected_gain`` in ``[0, 1]`` is the retriever's own estimate that it
        holds something relevant, normally the top similarity score. A retriever
        that returns nothing costs its latency and removes no risk, which is the
        behaviour that stops unconditional retrieval from being free.
        """
        residual = uncertainty * (1.0 - self._retrieval_efficacy * expected_gain)
        return self._retrieval_s + residual * self._recovery_s

    def cost_ask(self, uncertainty: float) -> float:
        """Expected seconds when querying the operator.

        The operator term is paid in full whatever the uncertainty: a person is
        interrupted regardless of how the robot felt about the situation.
        """
        residual = uncertainty * (1.0 - self._ask_efficacy)
        return self._operator_s + residual * self._recovery_s
