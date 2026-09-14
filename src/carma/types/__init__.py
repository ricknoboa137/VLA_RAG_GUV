"""Shared dataclasses and protocols.

This package depends on nothing else in ``carma``. Every other package may
import it; it may import none of them. See ``AGENTS.md`` section 3.
"""

from __future__ import annotations

from carma.types.core import (
    Action,
    AnalysisResult,
    Choice,
    Correction,
    Decision,
    EpisodeRecord,
    EpisodeSpec,
    IntentKind,
    MemoryEntry,
    Observation,
    OperatorIntent,
    OperatorQuery,
    OperatorResponse,
    PhenologyStage,
    Pose,
    RetrievalResult,
    StepRecord,
)
from carma.types.errors import (
    AnalyzerError,
    CarmaError,
    ConfigError,
    DeterminismError,
    MemoryError_,
    OperatorUnavailableError,
    SimulatorError,
)
from carma.types.protocols import (
    Arbiter,
    CostModel,
    Embedder,
    HygienePolicy,
    NavigationBackbone,
    OperatorInterface,
    Retriever,
    SceneAnalyzer,
    SimAdapter,
    Store,
)

__all__ = [
    "Action",
    "AnalysisResult",
    "AnalyzerError",
    "Arbiter",
    "CarmaError",
    "Choice",
    "ConfigError",
    "Correction",
    "CostModel",
    "Decision",
    "DeterminismError",
    "Embedder",
    "EpisodeRecord",
    "EpisodeSpec",
    "HygienePolicy",
    "IntentKind",
    "MemoryEntry",
    "MemoryError_",
    "NavigationBackbone",
    "Observation",
    "OperatorIntent",
    "OperatorInterface",
    "OperatorQuery",
    "OperatorResponse",
    "OperatorUnavailableError",
    "PhenologyStage",
    "Pose",
    "RetrievalResult",
    "Retriever",
    "SceneAnalyzer",
    "SimAdapter",
    "SimulatorError",
    "StepRecord",
    "Store",
]
