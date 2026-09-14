"""Configuration loading and object construction.

An experiment is fully described by a YAML file. Nothing that changes
scientific behaviour is a command-line flag. The config hash recorded in a run
manifest is computed over the *resolved* config, so a default that changes
between versions produces a different hash, as it should.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from carma.types import ConfigError, SceneAnalyzer


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A registry key plus its constructor parameters."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The complete description of one experiment.

    Attributes:
        name: Short identifier, used in run directory names.
        condition: One of the five keys in ``configs/condition/``.
        seed: Master seed. Every component RNG is derived from it.
        episodes: How many synthetic episodes to generate when the episode
            source is synthetic.
        backbone, embedder, store, retriever, hygiene, cost_model, arbiter,
        operator, sim: Component specifications.
    """

    name: str
    condition: str
    seed: int
    episodes: int
    backbone: ComponentSpec
    embedder: ComponentSpec
    store: ComponentSpec
    retriever: ComponentSpec
    hygiene: ComponentSpec
    cost_model: ComponentSpec
    arbiter: ComponentSpec
    operator: ComponentSpec
    sim: ComponentSpec

    def resolved(self) -> dict[str, Any]:
        """The config as a plain dictionary, with defaults applied."""
        return asdict(self)

    def hash(self) -> str:
        """Stable 16-character hash over the resolved config, excluding the seed.

        The seed is excluded deliberately: repeated seeds of one configuration
        are exactly what should be aggregated, while any other difference must
        prevent aggregation. Two runs may only be pooled when their hashes
        agree; ``carma report`` enforces that and has no override flag.
        """
        payload = {k: v for k, v in self.resolved().items() if k != "seed"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


_COMPONENTS = (
    "backbone",
    "embedder",
    "store",
    "retriever",
    "hygiene",
    "cost_model",
    "arbiter",
    "operator",
    "sim",
)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "backbone": {"kind": "heuristic", "params": {}},
    "embedder": {"kind": "hashing", "params": {"dim": 256}},
    "store": {"kind": "in_memory", "params": {}},
    "retriever": {"kind": "similarity", "params": {"min_score": 0.15}},
    "hygiene": {"kind": "stage_decay", "params": {}},
    "cost_model": {"kind": "linear", "params": {}},
    "arbiter": {"kind": "cost_aware", "params": {}},
    "operator": {"kind": "scripted", "params": {}},
    "sim": {"kind": "synthetic", "params": {}},
}


def _as_spec(raw: object, name: str) -> ComponentSpec:
    if not isinstance(raw, dict):
        msg = f"component {name!r} must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    kind = raw.get("kind")
    if not isinstance(kind, str):
        msg = f"component {name!r} is missing a string 'kind'"
        raise ConfigError(msg)
    params = raw.get("params", {})
    if not isinstance(params, dict):
        msg = f"component {name!r} has non-mapping 'params'"
        raise ConfigError(msg)
    return ComponentSpec(kind=kind, params=dict(params))


def load(path: Path) -> ExperimentConfig:
    """Read and validate an experiment config.

    Unknown top-level keys are an error rather than a warning, so a typo in a
    config never silently produces a run with default behaviour.
    """
    if not path.exists():
        msg = f"config not found: {path}"
        raise ConfigError(msg)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        msg = f"config {path} must contain a mapping at the top level"
        raise ConfigError(msg)

    allowed = {"name", "condition", "seed", "episodes", *_COMPONENTS}
    unknown = set(raw) - allowed
    if unknown:
        msg = f"unknown config keys: {', '.join(sorted(unknown))}"
        raise ConfigError(msg)

    for required in ("name", "condition"):
        if required not in raw:
            msg = f"config is missing required key {required!r}"
            raise ConfigError(msg)

    merged: dict[str, Any] = {}
    for comp in _COMPONENTS:
        base = dict(_DEFAULTS[comp])
        override = raw.get(comp)
        if override is not None:
            if not isinstance(override, dict):
                msg = f"component {comp!r} must be a mapping"
                raise ConfigError(msg)
            base = {**base, **override, "params": {**base["params"], **override.get("params", {})}}
        merged[comp] = _as_spec(base, comp)

    return ExperimentConfig(
        name=str(raw["name"]),
        condition=str(raw["condition"]),
        seed=int(raw.get("seed", 0)),
        episodes=int(raw.get("episodes", 5)),
        **merged,
    )


@dataclass(frozen=True, slots=True)
class Assembly:
    """Every constructed component for one experiment.

    Produced by :func:`build`, which is the project's single composition root.
    No module instantiates a concrete class from another package by name; that
    rule is what keeps a condition a config file rather than a code path.
    """

    backbone: Any
    embedder: Any
    store: Any
    retriever: Any
    hygiene: Any
    cost_model: Any
    arbiter: Any
    operator: Any
    sim: Any


def build(cfg: ExperimentConfig) -> Assembly:
    """Construct every component named by ``cfg`` and wire them together.

    Imports are deferred to call time so that ``carma.config`` stays importable
    from any package without an import cycle.
    """
    from carma.arbiter.cost_model import COST_MODELS
    from carma.arbiter.policies import ARBITERS
    from carma.backbone.registry import BACKBONES
    from carma.memory.embedders.registry import EMBEDDERS
    from carma.memory.hygiene import HYGIENE
    from carma.memory.retriever import RETRIEVERS
    from carma.memory.stores.registry import STORES
    from carma.operator.registry import OPERATORS
    from carma.sim.registry import SIMS

    embedder = EMBEDDERS.create(cfg.embedder.kind, **cfg.embedder.params)
    store = STORES.create(cfg.store.kind, **cfg.store.params)
    retriever = RETRIEVERS.create(cfg.retriever.kind, **cfg.retriever.params)
    cost_model = COST_MODELS.create(cfg.cost_model.kind, **cfg.cost_model.params)
    arbiter = ARBITERS.create(cfg.arbiter.kind, **cfg.arbiter.params)

    bind_retriever = getattr(retriever, "bind", None)
    if callable(bind_retriever):
        bind_retriever(embedder, store)

    bind_arbiter = getattr(arbiter, "bind", None)
    if callable(bind_arbiter):
        bind_arbiter(cost_model, retriever)

    return Assembly(
        backbone=BACKBONES.create(cfg.backbone.kind, **cfg.backbone.params),
        embedder=embedder,
        store=store,
        retriever=retriever,
        hygiene=HYGIENE.create(cfg.hygiene.kind, **cfg.hygiene.params),
        cost_model=cost_model,
        arbiter=arbiter,
        operator=OPERATORS.create(cfg.operator.kind, **cfg.operator.params),
        sim=SIMS.create(cfg.sim.kind, **cfg.sim.params),
    )


def load_analyzers(path: Path) -> tuple[ComponentSpec, ...]:
    """Read an analyzer list from ``configs/analyzers/<name>.yaml``.

    Analyzers live in their own file rather than in :class:`ExperimentConfig`
    because they describe the scene without influencing decisions: adding one
    to the robot must not change any experiment's config hash.
    """
    if not path.exists():
        msg = f"analyzer config not found: {path}"
        raise ConfigError(msg)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"analyzers"}:
        msg = f"analyzer config {path} must contain exactly one key, 'analyzers'"
        raise ConfigError(msg)
    items = raw["analyzers"]
    if not isinstance(items, list):
        msg = f"'analyzers' in {path} must be a list"
        raise ConfigError(msg)
    return tuple(_as_spec(item, f"analyzers[{i}]") for i, item in enumerate(items))


def build_analyzers(specs: Sequence[ComponentSpec]) -> tuple[SceneAnalyzer, ...]:
    """Construct scene analyzers by registry key, in the order given."""
    from carma.perception.analyzers import ANALYZERS

    return tuple(ANALYZERS.create(spec.kind, **spec.params) for spec in specs)
