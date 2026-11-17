"""Turning observations into retrieved context, and corrections into entries."""

from __future__ import annotations

import time

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
        self._ingested = 0

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

    def ingest(self, correction: Correction, *, run_t: float) -> MemoryEntry:
        """Convert an operator correction into a stored entry.

        The entry text is the operator's words. Paraphrasing to strip
        task-specific detail is a deliberate later step and belongs in its own
        module, not hidden here.

        Args:
            correction: The operator's correction.
            run_t: Run clock in simulated seconds; stored as ``created_at``.
        """
        embedder, store = self._require()
        vec = embedder.embed([correction.text])[0]
        entry = MemoryEntry(
            # A counter, not a random id: entry ids reach the decision log
            # (retrieved_ids), and the same (config, seed, commit) must
            # produce the same log.
            entry_id=f"e{self._ingested:06d}",
            text=correction.text,
            embedding=vec,
            pose=correction.observation.pose,
            created_at=run_t,
            stage=correction.observation.stage,
            provenance=correction.operator_id,
            meta={"took_over": correction.took_over},
        )
        store.add(entry)
        self._ingested += 1
        return entry


@RETRIEVERS.register("spatial")
class SpatialRetriever(SimilarityRetriever):
    """Returns the live entries given near where the robot is now.

    For knowledge tied to a place rather than to a phrase: an operator's
    warning about a hidden wet patch is useful exactly where it was given, and
    the instruction text ("drive 13 metres down the row") says nothing about
    it, so instruction similarity cannot find it.

    The score of an entry is ``max(0, 1 - d / radius_m)``, with ``d`` the
    distance in metres between the robot's pose and the pose the entry was
    given at: 1 on the spot, 0 at ``radius_m`` and beyond. The arbiter's probe
    reads it as retrieval's expected gain, so retrieval is cheap to choose only
    where memory actually holds something about this place. Ties keep
    insertion order.

    Args:
        radius_m: Distance in metres at which an entry stops being relevant.
        min_score: Entries scoring below this are dropped.
    """

    def __init__(self, radius_m: float = 4.0, min_score: float = 0.15) -> None:
        if radius_m <= 0.0:
            msg = f"radius_m must be positive, got {radius_m}"
            raise ValueError(msg)
        super().__init__(min_score=min_score)
        self._radius_m = radius_m

    def retrieve(self, obs: Observation, k: int) -> RetrievalResult:
        """Retrieve up to ``k`` live entries within ``radius_m`` of ``obs.pose``."""
        _embedder, store = self._require()
        started = time.perf_counter()
        scored = [
            (e, max(0.0, 1.0 - obs.pose.distance_to(e.pose) / self._radius_m))
            for e in store.all()
            if not e.invalidated
        ]
        kept = [(e, s) for e, s in scored if s >= self._min_score]
        kept.sort(key=lambda pair: -pair[1])  # stable: equal scores keep insertion order
        kept = kept[:k]
        return RetrievalResult(
            entries=tuple(e for e, _ in kept),
            scores=tuple(s for _, s in kept),
            latency_s=time.perf_counter() - started,
        )
