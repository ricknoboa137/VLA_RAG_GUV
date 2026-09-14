"""The four arbiters that define the experimental conditions.

Each condition in ``configs/condition/`` names one of these. Adding a fifth
means adding a claim to the paper; see ``AGENTS.md`` section 5.
"""

from __future__ import annotations

from carma.registry import Registry
from carma.types import Arbiter
from carma.types.core import Choice, Decision, Observation
from carma.types.protocols import CostModel, Retriever

ARBITERS: Registry[Arbiter] = Registry("arbiter")


@ARBITERS.register("always_act")
class AlwaysAct:
    """Baseline floor: never retrieves, never asks."""

    def decide(self, obs: Observation, uncertainty: float) -> Decision:
        """Always choose ACT."""
        return Decision(
            choice=Choice.ACT,
            costs={Choice.ACT: 0.0, Choice.RETRIEVE: float("inf"), Choice.ASK: float("inf")},
            rationale="no-memory baseline",
        )


@ARBITERS.register("always_retrieve")
class AlwaysRetrieve:
    """The published retrieval-augmented pattern: retrieve on every step."""

    def decide(self, obs: Observation, uncertainty: float) -> Decision:
        """Always choose RETRIEVE."""
        return Decision(
            choice=Choice.RETRIEVE,
            costs={Choice.ACT: float("inf"), Choice.RETRIEVE: 0.0, Choice.ASK: float("inf")},
            rationale="unconditional retrieval baseline",
        )


@ARBITERS.register("uncertainty_ask")
class UncertaintyAsk:
    """Ask when uncertain, otherwise act. No memory is consulted.

    Args:
        threshold: Uncertainty above which the operator is queried.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def decide(self, obs: Observation, uncertainty: float) -> Decision:
        """Choose ASK above the threshold, otherwise ACT."""
        ask = uncertainty >= self._threshold
        return Decision(
            choice=Choice.ASK if ask else Choice.ACT,
            costs={
                Choice.ACT: 0.0 if not ask else float("inf"),
                Choice.RETRIEVE: float("inf"),
                Choice.ASK: 0.0 if ask else float("inf"),
            },
            rationale=f"uncertainty {uncertainty:.2f} vs threshold {self._threshold:.2f}",
        )


@ARBITERS.register("cost_aware")
class CostAwareArbiter:
    """The proposed policy: price all three options and take the cheapest.

    Retrieval's expected gain is estimated by a cheap probe against the store,
    so the arbiter knows whether memory holds anything relevant *before* paying
    for a full retrieval. That probe is what separates this from unconditional
    retrieval; without it the comparison would be unfair to the baseline.

    Args:
        probe_k: How many entries the gain probe looks at.
        ask_budget: Maximum queries per episode. Exceeding it makes ASK
            infinitely expensive, which models an operator's finite patience.
    """

    def __init__(self, probe_k: int = 3, ask_budget: int = 10) -> None:
        self._probe_k = probe_k
        self._ask_budget = ask_budget
        self._asks_used = 0
        self._cost_model: CostModel | None = None
        self._retriever: Retriever | None = None

    def bind(self, cost_model: CostModel, retriever: Retriever) -> None:
        """Attach collaborators. Called once by ``carma.config.build``."""
        self._cost_model = cost_model
        self._retriever = retriever

    def reset(self) -> None:
        """Clear the per-episode ask budget."""
        self._asks_used = 0

    def note_ask(self) -> None:
        """Record that a query was actually issued."""
        self._asks_used += 1

    def decide(self, obs: Observation, uncertainty: float) -> Decision:
        """Price ACT, RETRIEVE and ASK, and return the cheapest."""
        if self._cost_model is None or self._retriever is None:
            msg = "arbiter used before bind()"
            raise RuntimeError(msg)

        probe = self._retriever.retrieve(obs, self._probe_k)
        expected_gain = probe.best_score

        costs = {
            Choice.ACT: self._cost_model.cost_act(uncertainty),
            Choice.RETRIEVE: self._cost_model.cost_retrieve(uncertainty, expected_gain),
            Choice.ASK: (
                self._cost_model.cost_ask(uncertainty)
                if self._asks_used < self._ask_budget
                else float("inf")
            ),
        }
        # Ties resolve toward the cheaper-for-the-human option by enumeration
        # order: ACT, then RETRIEVE, then ASK.
        choice = min(costs, key=lambda c: (costs[c], list(Choice).index(c)))
        return Decision(
            choice=choice,
            costs=costs,
            rationale=(
                f"u={uncertainty:.2f} gain={expected_gain:.2f} "
                f"asks={self._asks_used}/{self._ask_budget}"
            ),
        )
