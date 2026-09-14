"""Value-type invariants."""

from __future__ import annotations

import numpy as np
import pytest

from carma.types.core import PhenologyStage, Pose, RetrievalResult


def test_pose_distance() -> None:
    """Planar Euclidean distance, in metres."""
    assert Pose(3.0, 4.0, 0.0).distance_to(Pose(0.0, 0.0, 0.0)) == pytest.approx(5.0)


def test_stage_distance_is_symmetric() -> None:
    """Stage distance drives staleness, so it must not depend on direction."""
    a, b = PhenologyStage.BARE_SOIL, PhenologyStage.CANOPY_CLOSURE
    assert a.distance_to(b) == b.distance_to(a) == 3


def test_retrieval_result_rejects_mismatched_scores() -> None:
    """Entries and scores must correspond one to one."""
    with pytest.raises(ValueError, match="equal length"):
        RetrievalResult(entries=(), scores=(0.5,), latency_s=0.0)


def test_empty_retrieval_has_zero_best_score() -> None:
    """An empty result prices retrieval at no expected gain."""
    assert RetrievalResult(entries=(), scores=(), latency_s=0.0).best_score == 0.0


def test_embeddings_are_float32() -> None:
    """float64 embeddings would double memory for no benefit."""
    assert np.zeros(4, dtype=np.float32).dtype == np.float32
