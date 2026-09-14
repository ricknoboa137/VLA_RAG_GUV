"""Store, embedder, retriever and hygiene behaviour."""

from __future__ import annotations

import time

import numpy as np
import pytest

from carma.memory.embedders import HashingEmbedder
from carma.memory.hygiene import NoHygiene, StageDecayHygiene
from carma.memory.retriever import SimilarityRetriever
from carma.memory.stores import InMemoryStore
from carma.types import MemoryError_
from carma.types.core import Correction, MemoryEntry, Observation, PhenologyStage, Pose


def _entry(eid: str, text: str, embedder: HashingEmbedder, **kw: object) -> MemoryEntry:
    return MemoryEntry(
        entry_id=eid,
        text=text,
        embedding=embedder.embed([text])[0],
        pose=kw.get("pose", Pose(0.0, 0.0, 0.0)),  # type: ignore[arg-type]
        created_at=kw.get("created_at", 0.0),  # type: ignore[arg-type]
        stage=kw.get("stage", PhenologyStage.VEGETATIVE),  # type: ignore[arg-type]
        provenance="op-0",
    )


def test_hashing_embedder_is_unit_norm_and_stable() -> None:
    """Embeddings are L2-normalised and identical across calls."""
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["stay left at the row end"])
    b = emb.embed(["stay left at the row end"])
    assert a.shape == (1, 64)
    assert np.allclose(np.linalg.norm(a[0]), 1.0)
    assert np.array_equal(a, b)


def test_hashing_embedder_rejects_bad_dim() -> None:
    """A non-positive dimension is a construction error."""
    with pytest.raises(ValueError, match="dim must be positive"):
        HashingEmbedder(dim=0)


def test_store_rejects_duplicate_ids() -> None:
    """Duplicate entry ids would silently shadow knowledge, so they raise."""
    emb = HashingEmbedder(dim=32)
    store = InMemoryStore()
    store.add(_entry("a", "one", emb))
    with pytest.raises(MemoryError_, match="duplicate"):
        store.add(_entry("a", "two", emb))


def test_invalidated_entries_are_never_retrieved() -> None:
    """Retirement removes an entry from search but not from the record."""
    emb = HashingEmbedder(dim=32)
    store = InMemoryStore()
    store.add(_entry("a", "stay left at the row end", emb))
    query = emb.embed(["stay left at the row end"])[0]

    assert len(store.search(query, 3)) == 1
    store.invalidate("a")
    assert store.search(query, 3) == ()
    assert len(store.all()) == 1


def test_search_ties_preserve_insertion_order() -> None:
    """Determinism depends on stable tie-breaking."""
    emb = HashingEmbedder(dim=32)
    store = InMemoryStore()
    for eid in ("first", "second", "third"):
        store.add(_entry(eid, "identical text", emb))
    query = emb.embed(["identical text"])[0]
    got = [e.entry_id for e, _ in store.search(query, 3)]
    assert got == ["first", "second", "third"]


def test_retriever_ingests_and_finds(observation: Observation) -> None:
    """A correction becomes an entry that its own words retrieve."""
    emb = HashingEmbedder(dim=128)
    store = InMemoryStore()
    retriever = SimilarityRetriever(min_score=0.0)
    retriever.bind(emb, store)

    correction = Correction(
        text=observation.instruction,
        observation=observation,
        operator_id="op-0",
    )
    entry = retriever.ingest(correction)
    result = retriever.retrieve(observation, k=3)

    assert entry.entry_id in {e.entry_id for e in result.entries}
    assert result.best_score > 0.9


def test_retriever_requires_binding(observation: Observation) -> None:
    """Using a retriever before the composition root binds it is a bug."""
    with pytest.raises(RuntimeError, match="bind"):
        SimilarityRetriever().retrieve(observation, k=1)


def test_staleness_rises_with_stage_distance(observation: Observation) -> None:
    """An entry from bare soil is staler than one from the current stage."""
    emb = HashingEmbedder(dim=32)
    hygiene = StageDecayHygiene()
    same = _entry("same", "x", emb, stage=PhenologyStage.VEGETATIVE)
    far = _entry("far", "x", emb, stage=PhenologyStage.BARE_SOIL)
    assert hygiene.staleness(far, observation) > hygiene.staleness(same, observation)


def test_staleness_rises_with_distance(observation: Observation) -> None:
    """An entry recorded far away is staler than one recorded here."""
    emb = HashingEmbedder(dim=32)
    hygiene = StageDecayHygiene()
    near = _entry("near", "x", emb, pose=Pose(0.0, 0.0, 0.0))
    far = _entry("far", "x", emb, pose=Pose(100.0, 0.0, 0.0))
    assert hygiene.staleness(far, observation) > hygiene.staleness(near, observation)


def test_stage_decay_weights_must_sum_to_one() -> None:
    """Weights that do not sum to one would make staleness uninterpretable."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        StageDecayHygiene(stage_weight=0.9, age_weight=0.9, distance_weight=0.9)


def test_no_hygiene_ablation_is_inert(observation: Observation) -> None:
    """The ablation reports nothing stale and selects nothing for review."""
    emb = HashingEmbedder(dim=32)
    entry = _entry("a", "x", emb, created_at=time.time() - 10_000_000)
    hygiene = NoHygiene()
    assert hygiene.staleness(entry, observation) == 0.0
    assert hygiene.select_for_review((entry,), observation, 5) == ()
