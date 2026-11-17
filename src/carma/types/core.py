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
        collided: True when the robot touched something other than the ground
            since the previous observation (a crop row, an obstacle). The
            episode ends there and counts as a failure.
        stalled: True when the robot was driven but did not move (its camera
            view stayed static). On a real robot this is how a collision shows
            up; the episode ends there and counts as a failure.
        true_pose: Ground-truth pose, where a simulator has one, for scoring
            only: the policy and the arbiter use ``pose``, which may be the
            robot's own estimate. ``None`` means ``pose`` is the best there is
            (a real robot), and episodes are scored on it.
    """

    rgb: BgrImage
    depth: DepthImage | None
    pose: Pose
    t: float
    instruction: str
    stage: PhenologyStage
    rgb_right: BgrImage | None = None
    collided: bool = False
    stalled: bool = False
    true_pose: Pose | None = None

    @property
    def scored_pose(self) -> Pose:
        """The pose episodes are scored on: the truth when known, else ``pose``."""
        return self.true_pose if self.true_pose is not None else self.pose


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
        created_at: Run clock at creation: simulated seconds since the start
            of the run, accumulated across its episodes (an episode's
            ``Observation.t`` plus the durations of the episodes before it).
            Never wall-clock time, so the same (config, seed, commit) gives
            the same entries.
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


class IntentKind(enum.Enum):
    """What an operator utterance asks of the robot."""

    STOP = "stop"
    GO = "go"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    YES = "yes"
    NO = "no"
    CORRECTION = "correction"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class OperatorIntent:
    """An operator utterance reduced to something the system can act on.

    Attributes:
        kind: The recognised intent. ``CORRECTION`` carries free text destined
            for memory; ``NONE`` means nothing usable was said.
        text: The utterance as transcribed, whitespace-trimmed.
        language: ISO 639-1 code reported by speech recognition, or ``""``.
    """

    kind: IntentKind
    text: str
    language: str = ""


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
class Hazard:
    """A spot the robot must not enter and cannot see: a wet patch, a buried stone.

    Hidden on purpose. The camera shows nothing there, so the backbone alone
    cannot avoid it; only knowledge from the operator, asked now or remembered
    from an earlier pass, can. Entering it ends the episode as a failure.

    Attributes:
        position: Centre of the hazard, map frame.
        keep_out_m: The robot fails when its centre comes within this many
            metres of ``position``.
        advice: What an operator who knows the field says about it, e.g.
            "wet patch at 8 metres: stop at 6 metres". The scripted operator
            answers and volunteers this text; it is never given to the policy
            directly.
    """

    position: Pose
    keep_out_m: float
    advice: str


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
        hazards: Hidden hazards on the route (:class:`Hazard`). When there is
            one, ``goal`` is the stop point before it, not the distance named
            in ``instruction``: stopping short on the operator's advice is the
            right outcome, driving on as instructed is not.
    """

    episode_id: str
    instruction: str
    start: Pose
    goal: Pose
    success_radius: float
    shortest_path_length: float
    stage: PhenologyStage
    max_steps: int = 500
    hazards: tuple[Hazard, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """The outcome of running one :class:`EpisodeSpec`.

    ``collided`` is true when the episode ended because the robot hit
    something, and ``stalled`` when it ended because the robot was driven but
    did not move; either is a failure wherever it ended. ``final_pose`` and
    ``path_length`` are measured on the scored pose
    (:attr:`Observation.scored_pose`).
    """

    spec: EpisodeSpec
    steps: tuple[StepRecord, ...]
    final_pose: Pose
    path_length: float
    wall_time_s: float
    collided: bool = False
    stalled: bool = False
    entered_hazard: bool = False

    @property
    def success(self) -> bool:
        """True when the robot ended inside the goal radius with no collision, stall or hazard."""
        if self.collided or self.stalled or self.entered_hazard:
            return False
        return self.final_pose.distance_to(self.spec.goal) <= self.spec.success_radius

    @property
    def navigation_error(self) -> float:
        """Metres between the final pose and the goal."""
        return self.final_pose.distance_to(self.spec.goal)
