"""Turning observations into retrieved context, and corrections into entries."""

from __future__ import annotations

import time
import uuid

from carma.registry import Registry
from carma.types import Retriever
from carma.types.core import Correction, MemoryEntry, Observation, RetrievalResult
from carma.types.protocols import Embedder, Store

RETRIEVERS: Registry[Retriever] = Registry("retriever")


@RETRIEVERS.register("similarity")
class SimilarityRetriever:
    """Embeds the instruction and returns the nearest live entries.

    Args:
        min_score: Entries scoring below this are dropped. A retriever that
            returns weak matches is worse than one that returns nothing,
            because the arbiter prices retrieval on its expected gain.
    """

    def __init__(self, min_score: float = 0.15) -> None:
        self._min_score = min_score
        self._embedder: Embedder | None = None
        self._store: Store | None = None

    def bind(self, embedder: Embedder, store: Store) -> None:
        """Attach the embedder and store.

        Called once by ``carma.config.build``. Kept separate from ``__init__``
        so that a retriever can be named in config without the config file
        needing to know how the store was constructed.
        """
        self._embedder = embedder
        self._store = store

    def _require(self) -> tuple[Embedder, Store]:
        if self._embedder is None or self._store is None:
            msg = "retriever used before bind()"
            raise RuntimeError(msg)
        return self._embedder, self._store

    def retrieve(self, obs: Observation, k: int) -> RetrievalResult:
        """Retrieve up to ``k`` entries relevant to ``obs``."""
        embedder, store = self._require()
        started = time.perf_counter()
        query = embedder.embed([obs.instruction])[0]
        hits = store.search(query, k)
        kept = [(e, s) for e, s in hits if s >= self._min_score]
        latency = time.perf_counter() - started
        return RetrievalResult(
            entries=tuple(e for e, _ in kept),
            scores=tuple(s for _, s in kept),
            latency_s=latency,
        )

    def ingest(self, correction: Correction) -> MemoryEntry:
        """Convert an operator correction into a stored entry.

        The entry text is the operator's words. Paraphrasing to strip
        task-specific detail is a deliberate later step and belongs in its own
        module, not hidden here.
        """
        embedder, store = self._require()
        vec = embedder.embed([correction.text])[0]
        entry = MemoryEntry(
            entry_id=uuid.uuid4().hex[:12],
            text=correction.text,
            embedding=vec,
            pose=correction.observation.pose,
            created_at=time.time(),
            stage=correction.observation.stage,
            provenance=correction.operator_id,
            meta={"took_over": correction.took_over},
        )
        store.add(entry)
        return entry
