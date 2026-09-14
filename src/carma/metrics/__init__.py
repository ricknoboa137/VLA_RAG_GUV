"""Every number that appears in a paper is computed here.

No other module computes a success rate, a query count or a latency summary. If
you find that logic elsewhere, it is a bug: move it here and leave a test
behind. See ``AGENTS.md`` section 2, rule 5.

The four families correspond to the four measurement families in the research
design. The interaction family is the one absent from the existing retrieval
literature, and it is the reason this package exists as a first-class concern
rather than a reporting afterthought.
"""

from __future__ import annotations

from carma.metrics.bundle import MetricBundle, summarise
from carma.metrics.interaction import (
    interventions_per_hectare,
    mean_neglect_time,
    operator_latency_total,
    queries_per_100m,
)
from carma.metrics.retrieval import precision_at_k, retrieval_rate, staleness_rate
from carma.metrics.systems import decision_latency_summary
from carma.metrics.task import navigation_error, spl, success_rate

__all__ = [
    "MetricBundle",
    "decision_latency_summary",
    "interventions_per_hectare",
    "mean_neglect_time",
    "navigation_error",
    "operator_latency_total",
    "precision_at_k",
    "queries_per_100m",
    "retrieval_rate",
    "spl",
    "staleness_rate",
    "success_rate",
    "summarise",
]
