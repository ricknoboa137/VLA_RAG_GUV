"""Core value types.

Units are SI throughout and are stated in every docstring: distances in metres,
angles in radians, time in seconds. A field whose unit differs carries the unit
in its name (``timeout_ms``, ``heading_deg``). See ``AGENTS.md`` section 7.

Image arrays follow the OpenCV convention: ``HWC``, ``BGR``, ``uint8``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

BgrImage = npt.NDArray[np.uint8]
DepthImage = npt.NDArray[np.float32]
Embedding = npt.NDArray[np.float32]


class PhenologyStage(enum.Enum):
    """Crop development stage at the time an observation or entry was made.

    Retrieval relevance decays across stages, so every memory entry records the
    stage it was created in. Extend deliberately: the ordering is used to
    compute stage distance in the hygiene policy.
    """

    BARE_SOIL = 0
    EMERGENCE = 1
    VEGETATIVE = 2
    CANOPY_CLOSURE = 3
    SENESCENCE = 4
    POST_HARVEST = 5

    def distance_to(self, other: PhenologyStage) -> int:
        """Number of stages between ``self`` and ``other``."""
        return abs(self.value - other.value)


class Choice(enum.Enum):
    """The three options the arbiter selects between."""

    ACT = "act"
    RETRIEVE = "retrieve"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class Pose:
    """Planar robot pose in the map frame.

    Attributes:
        x: Easting in metres.
        y: Northing in metres.
        yaw: Heading in radians, counter-clockwise from the x axis.
    """

    x: float
    y: float
    yaw: float

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance to ``other`` in metres."""
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True, slots=True)
class Observation:
    """One synchronised sensor reading.

    Attributes:
        rgb: Colour image, HWC, BGR, uint8.
        depth: Depth image in metres, or ``None`` when unavailable.
        pose: Robot pose in the map frame.
        t: Seconds since episode start.
        instruction: The natural-language navigation instruction in force.
        stage: Crop stage at capture time.
        rgb_right: Right image of a stereo pair, HWC, BGR, uint8, or ``None``
            for a monocular camera. ``rgb`` is always the left image.
    """

    rgb: BgrImage
    depth: DepthImage | None
    pose: Pose
    t: float
    instruction: str
    stage: PhenologyStage
    rgb_right: BgrImage | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The output of one scene analyzer on one observation.

    Analyzers describe the surroundings (plant health, obstacles, weeds); they
    never choose actions. Scores are free-form per analyzer but must be plain
    floats so results serialise to JSON without a custom encoder.

    Attributes:
        analyzer: Registry key of the analyzer that produced this result.
        t: Seconds since episode start, copied from the observation.
        pose: Robot pose at capture, copied from the observation.
        label: The analyzer's headline verdict, e.g. ``"healthy"``.
        scores: Named quantities, each documented by its analyzer with units.
        meta: Free-form extras; must be JSON-serialisable.
    """

    analyzer: str
    t: float
    pose: Pose
    label: str
    scores: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action:
    """A velocity command, optionally terminating the episode.

    Attributes:
        linear: Forward velocity in metres per second.
        angular: Yaw rate in radians per second, positive counter-clockwise.
        stop: True when the policy declares the goal reached.
    """

    linear: float
    angular: float
    stop: bool = False


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One retrievable unit of operator-derived knowledge.

    Entries are created from operator corrections, never from the robot's own
    rollouts. ``provenance`` records which operator gave it, so per-operator
    effects can be separated in analysis.

    Attributes:
        entry_id: Stable identifier, unique within a store.
        text: The paraphrased correction.
        embedding: Unit-norm text embedding, float32.
        pose: Where the correction was given.
        created_at: Seconds since the Unix epoch.
        stage: Crop stage at creation.
        provenance: Operator identifier.
        invalidated: True once an operator has retired this entry.
        meta: Free-form extras; never read by the arbiter.
    """

    entry_id: str
    text: str
    embedding: Embedding
    pose: Pose
    created_at: float
    stage: PhenologyStage
    provenance: str
    invalidated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Entries returned for one query, best first.

    ``scores[i]`` is the similarity of ``entries[i]`` to the query, in ``[0, 1]``.
    Ties preserve insertion order, which determinism depends on.
    """

    entries: tuple[MemoryEntry, ...]
    scores: tuple[float, ...]
    latency_s: float

    def __post_init__(self) -> None:
        if len(self.entries) != len(self.scores):
            msg = "entries and scores must have equal length"
            raise ValueError(msg)

    @property
    def best_score(self) -> float:
        """Similarity of the top entry, or 0.0 when nothing was returned."""
        return self.scores[0] if self.scores else 0.0


@dataclass(frozen=True, slots=True)
class OperatorQuery:
    """A question put to the human operator."""

    question: str
    observation: Observation
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OperatorResponse:
    """The operator's reply, and what it cost to obtain.

    Attributes:
        text: Free-form answer.
        latency_s: Seconds between asking and receiving. This is the quantity
            the interaction-economy metrics are built on.
        refused: True when the operator declined or timed out.
    """

    text: str
    latency_s: float
    refused: bool = False


@dataclass(frozen=True, slots=True)
class Correction:
    """An unsolicited operator intervention.

    Distinct from :class:`OperatorResponse`, which answers a question the robot
    asked. A correction is volunteered, and is the raw material from which
    memory entries are built.
    """

    text: str
    observation: Observation
    operator_id: str
    took_over: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    """The arbiter's output for one step.

    Attributes:
        choice: Which of the three options was selected.
        costs: Expected cost of each option, in seconds of additional traverse
            time. The chosen option is the argmin.
        rationale: Short human-readable explanation, for the decision log.
    """

    choice: Choice
    costs: dict[Choice, float]
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Everything worth keeping about one control step.

    This is a research artefact. Its schema is documented in
    ``docs/data-schema.md`` and is treated as public.
    """

    step: int
    t: float
    pose: Pose
    action: Action
    decision: Decision
    uncertainty: float
    retrieved_ids: tuple[str, ...] = ()
    asked: bool = False
    operator_latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """The definition of one navigation episode.

    Attributes:
        episode_id: Stable identifier.
        instruction: Natural-language instruction given to the backbone.
        start: Initial pose.
        goal: Goal pose used only for scoring, never given to the policy.
        success_radius: Metres within which the goal counts as reached.
        shortest_path_length: Metres, used as the denominator of SPL.
        stage: Crop stage this episode was recorded in.
        max_steps: Hard cap on episode length.
    """

    episode_id: str
    instruction: str
    start: Pose
    goal: Pose
    success_radius: float
    shortest_path_length: float
    stage: PhenologyStage
    max_steps: int = 500


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """The outcome of running one :class:`EpisodeSpec`."""

    spec: EpisodeSpec
    steps: tuple[StepRecord, ...]
    final_pose: Pose
    path_length: float
    wall_time_s: float

    @property
    def success(self) -> bool:
        """True when the robot finished inside the goal radius."""
        return self.final_pose.distance_to(self.spec.goal) <= self.spec.success_radius

    @property
    def navigation_error(self) -> float:
        """Metres between the final pose and the goal."""
        return self.final_pose.distance_to(self.spec.goal)
