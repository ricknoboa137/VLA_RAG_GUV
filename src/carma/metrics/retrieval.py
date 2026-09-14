"""Retrieval-quality metrics.

These separate *retrieved the right thing* from *acted well anyway*. Without
them a method can score well on task metrics while its retrieval contributes
nothing, which would make the central claim unfalsifiable.
"""

from __future__ import annotations

from collections.abc import Sequence

from carma.types.core import EpisodeRecord


def retrieval_rate(episodes: Sequence[EpisodeRecord]) -> float:
    """Fraction of steps on which retrieval was chosen, in ``[0, 1]``."""
    steps = [s for e in episodes for s in e.steps]
    if not steps:
        return 0.0
    return sum(1 for s in steps if s.retrieved_ids) / len(steps)


def precision_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    k: int,
) -> float:
    """Mean fraction of the top ``k`` retrieved ids that are relevant.

    Args:
        retrieved: Per-query ranked entry ids.
        relevant: Per-query ground-truth relevant ids.
        k: Cutoff.

    Relevance labels come from operator adjudication of a held-out sample; they
    are not inferred from whether the episode succeeded, which would be
    circular.
    """
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    if len(retrieved) != len(relevant):
        msg = "retrieved and relevant must have equal length"
        raise ValueError(msg)
    if not retrieved:
        return 0.0
    scores = [
        sum(1 for rid in got[:k] if rid in gold) / k
        for got, gold in zip(retrieved, relevant, strict=True)
    ]
    return sum(scores) / len(scores)


def staleness_rate(
    retrieved_ids: Sequence[str],
    stale_ids: set[str],
) -> float:
    """Fraction of retrieved entries that were stale, in ``[0, 1]``.

    The drift measurement. A rising staleness rate across a season with a flat
    success rate means the policy is compensating for bad retrieval, which is
    worth knowing.
    """
    if not retrieved_ids:
        return 0.0
    return sum(1 for rid in retrieved_ids if rid in stale_ids) / len(retrieved_ids)
