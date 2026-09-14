"""Staleness scoring and operator review selection.

A field robot's corpus decays in a way a tabletop's does not: the same
instruction at the same coordinates means something different once the canopy
closes. This module prices that decay and decides which entries are worth an
operator's confirmation.
"""

from __future__ import annotations

import numpy as np

from carma.registry import Registry
from carma.types import HygienePolicy
from carma.types.core import MemoryEntry, Observation

HYGIENE: Registry[HygienePolicy] = Registry("hygiene")

_SECONDS_PER_DAY = 86_400.0


@HYGIENE.register("stage_decay")
class StageDecayHygiene:
    """Staleness from phenological distance, age and spatial displacement.

    Args:
        stage_weight: Contribution of crop-stage distance.
        age_weight: Contribution of entry age.
        distance_weight: Contribution of distance between where the entry was
            recorded and where it is being retrieved.
        half_life_days: Age at which the age term reaches one half.
        distance_scale: Metres at which the distance term reaches one half.
    """

    def __init__(
        self,
        stage_weight: float = 0.5,
        age_weight: float = 0.2,
        distance_weight: float = 0.3,
        half_life_days: float = 30.0,
        distance_scale: float = 25.0,
    ) -> None:
        total = stage_weight + age_weight + distance_weight
        if not np.isclose(total, 1.0):
            msg = f"weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        self._stage_w = stage_weight
        self._age_w = age_weight
        self._dist_w = distance_weight
        self._half_life = half_life_days
        self._dist_scale = distance_scale

    def staleness(self, entry: MemoryEntry, obs: Observation) -> float:
        """Staleness in ``[0, 1]``; 1.0 means certainly no longer applicable."""
        stage_term = min(entry.stage.distance_to(obs.stage) / 3.0, 1.0)

        age_days = max(obs.t - entry.created_at, 0.0) / _SECONDS_PER_DAY
        age_term = 1.0 - 0.5 ** (age_days / self._half_life) if self._half_life > 0 else 0.0

        metres = entry.pose.distance_to(obs.pose)
        dist_term = metres / (metres + self._dist_scale)

        score = self._stage_w * stage_term + self._age_w * age_term + self._dist_w * dist_term
        return float(np.clip(score, 0.0, 1.0))

    def select_for_review(
        self,
        entries: tuple[MemoryEntry, ...],
        obs: Observation,
        budget: int,
    ) -> tuple[MemoryEntry, ...]:
        """Choose at most ``budget`` entries worth an operator confirmation.

        Ranks by staleness but skips entries already retired, and breaks ties by
        entry id so the selection is reproducible.
        """
        live = [e for e in entries if not e.invalidated]
        scored = sorted(live, key=lambda e: (-self.staleness(e, obs), e.entry_id))
        return tuple(scored[: max(budget, 0)])


@HYGIENE.register("none")
class NoHygiene:
    """Ablation: nothing is ever stale and nothing is ever reviewed.

    Used by the ``carma_no_hygiene`` condition to isolate the contribution of
    drift handling from that of arbitration.
    """

    def staleness(self, entry: MemoryEntry, obs: Observation) -> float:
        """Always 0.0."""
        return 0.0

    def select_for_review(
        self,
        entries: tuple[MemoryEntry, ...],
        obs: Observation,
        budget: int,
    ) -> tuple[MemoryEntry, ...]:
        """Always empty."""
        return ()
