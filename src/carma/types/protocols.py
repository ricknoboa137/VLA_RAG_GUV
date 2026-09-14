"""Structural interfaces for every replaceable component.

Adding an implementation means satisfying one of these protocols and
registering it, never widening a protocol with implementation-specific keyword
arguments. See ``AGENTS.md`` section 4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from carma.types.core import (
    Action,
    AnalysisResult,
    Correction,
    Decision,
    Embedding,
    MemoryEntry,
    Observation,
    OperatorQuery,
    OperatorResponse,
    RetrievalResult,
)


@runtime_checkable
class NavigationBackbone(Protocol):
    """A vision-language-action navigation policy."""

    def act(
        self,
        obs: Observation,
        context: tuple[str, ...] = (),
        *,
        rng: np.random.Generator,
    ) -> tuple[Action, float]:
        """Return an action and an uncertainty in ``[0, 1]``.

        ``context`` carries retrieved correction texts. A backbone that ignores
        context must still accept it, so that conditions differ only by config.
        """
        ...

    def reset(self) -> None:
        """Clear any per-episode internal state."""
        ...


@runtime_checkable
class SceneAnalyzer(Protocol):
    """Describes the surroundings from one observation.

    The extension point for user-trained perception models: plant health, weed
    detection, obstacle classes. An analyzer reports; it never acts and never
    reads memory, so adding one cannot change an experiment's decisions.
    """

    @property
    def name(self) -> str:
        """Registry key, copied into every result."""
        ...

    def analyze(self, obs: Observation) -> AnalysisResult:
        """Analyse one observation. Must be deterministic for a given image."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a unit-norm vector."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, texts: list[str]) -> Embedding:
        """Embed a batch, returning shape ``(len(texts), dim)``, float32."""
        ...


@runtime_checkable
class Store(Protocol):
    """Persistent collection of memory entries."""

    def add(self, entry: MemoryEntry) -> None:
        """Insert an entry. Raises if ``entry_id`` already exists."""
        ...

    def get(self, entry_id: str) -> MemoryEntry:
        """Return one entry. Raises ``MemoryError_`` when absent."""
        ...

    def invalidate(self, entry_id: str) -> None:
        """Mark an entry retired. Retired entries are never retrieved."""
        ...

    def all(self) -> tuple[MemoryEntry, ...]:
        """Every entry in insertion order, including invalidated ones."""
        ...

    def search(self, query: Embedding, k: int) -> tuple[tuple[MemoryEntry, float], ...]:
        """Return up to ``k`` live entries with similarity, best first.

        Ties must preserve insertion order or determinism breaks.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Turns an observation into retrieved context."""

    def retrieve(self, obs: Observation, k: int) -> RetrievalResult:
        """Retrieve up to ``k`` entries relevant to ``obs``."""
        ...

    def ingest(self, correction: Correction) -> MemoryEntry:
        """Convert an operator correction into a stored entry."""
        ...


@runtime_checkable
class HygienePolicy(Protocol):
    """Decides which entries have gone stale and which to put to the operator."""

    def staleness(self, entry: MemoryEntry, obs: Observation) -> float:
        """Staleness in ``[0, 1]``; 1.0 means certainly no longer applicable."""
        ...

    def select_for_review(
        self,
        entries: tuple[MemoryEntry, ...],
        obs: Observation,
        budget: int,
    ) -> tuple[MemoryEntry, ...]:
        """Choose at most ``budget`` entries worth an operator confirmation."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Prices the three options in a common currency.

    The currency is expected additional traverse time in seconds. Expressing
    operator attention and compute latency in the same unit is what makes them
    comparable; see ``docs/cost-model.md``.
    """

    def cost_act(self, uncertainty: float) -> float:
        """Expected cost of acting on the backbone's output directly."""
        ...

    def cost_retrieve(self, uncertainty: float, expected_gain: float) -> float:
        """Expected cost of retrieving, including retrieval latency."""
        ...

    def cost_ask(self, uncertainty: float) -> float:
        """Expected cost of querying the operator, including their latency."""
        ...


@runtime_checkable
class Arbiter(Protocol):
    """Selects between acting, retrieving and asking."""

    def decide(self, obs: Observation, uncertainty: float) -> Decision:
        """Return the decision for this step, with all three costs populated."""
        ...


@runtime_checkable
class OperatorInterface(Protocol):
    """The channel to the human.

    Implementations include a scripted operator for simulation, a replayed
    operator for offline analysis, and a live tablet or voice interface in the
    field. All three must report honest latencies, because those latencies are
    the interaction-economy measurements.
    """

    def ask(self, query: OperatorQuery) -> OperatorResponse:
        """Put a question and block until answered or timed out."""
        ...

    def poll_corrections(self, obs: Observation) -> tuple[Correction, ...]:
        """Return any unsolicited corrections offered since the last call."""
        ...


@runtime_checkable
class SimAdapter(Protocol):
    """A simulator or a physical robot, behind one interface."""

    def reset(self, *, seed: int) -> Observation:
        """Start a new episode and return the first observation."""
        ...

    def step(self, action: Action) -> Observation:
        """Apply an action for one control period."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...
