"""In-process store with exact cosine search.

Exact search over a few thousand entries is fast enough for this project and
removes an index as a source of non-determinism. A FAISS-backed store becomes
worthwhile only if the corpus grows past roughly ten thousand entries; it must
preserve insertion order on ties when it arrives.
"""

from __future__ import annotations

import numpy as np

from carma.memory.stores.registry import STORES
from carma.types import MemoryError_
from carma.types.core import Embedding, MemoryEntry


@STORES.register("in_memory")
class InMemoryStore:
    """Insertion-ordered store with exact cosine similarity search."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._order: list[str] = []

    def add(self, entry: MemoryEntry) -> None:
        """Insert an entry. Raises when ``entry_id`` is already present."""
        if entry.entry_id in self._entries:
            msg = f"duplicate entry_id {entry.entry_id!r}"
            raise MemoryError_(msg)
        self._entries[entry.entry_id] = entry
        self._order.append(entry.entry_id)

    def get(self, entry_id: str) -> MemoryEntry:
        """Return one entry, invalidated or not."""
        try:
            return self._entries[entry_id]
        except KeyError as exc:
            msg = f"no entry {entry_id!r}"
            raise MemoryError_(msg) from exc

    def invalidate(self, entry_id: str) -> None:
        """Mark an entry retired; it will no longer be returned by search."""
        entry = self.get(entry_id)
        self._entries[entry_id] = MemoryEntry(
            entry_id=entry.entry_id,
            text=entry.text,
            embedding=entry.embedding,
            pose=entry.pose,
            created_at=entry.created_at,
            stage=entry.stage,
            provenance=entry.provenance,
            invalidated=True,
            meta=entry.meta,
        )

    def all(self) -> tuple[MemoryEntry, ...]:
        """Every entry in insertion order, including invalidated ones."""
        return tuple(self._entries[eid] for eid in self._order)

    def search(self, query: Embedding, k: int) -> tuple[tuple[MemoryEntry, float], ...]:
        """Return up to ``k`` live entries with similarity, best first.

        Ties resolve by insertion order, which determinism depends on: the sort
        is stable and candidates are built in insertion order.
        """
        if k <= 0:
            return ()
        live = [self._entries[eid] for eid in self._order if not self._entries[eid].invalidated]
        if not live:
            return ()
        vec = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return ()
        vec = vec / norm
        matrix = np.stack([e.embedding for e in live]).astype(np.float32)
        scores = matrix @ vec
        ranked = sorted(zip(live, scores.tolist(), strict=True), key=lambda p: -p[1])
        return tuple((entry, float(score)) for entry, score in ranked[:k])
