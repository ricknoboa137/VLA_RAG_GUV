"""Config loading, validation and the composition root."""

from __future__ import annotations

from pathlib import Path

import pytest

from carma.config import ExperimentConfig, build, load
from carma.types import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_shipped_config_loads() -> None:
    """A config in the repository that does not load is a broken repository."""
    for path in sorted((REPO_ROOT / "configs" / "experiment").glob("*.yaml")):
        cfg = load(path)
        assert cfg.name
        assert cfg.condition


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    """A typo must fail loudly rather than silently running defaults."""
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\ncondition: carma\nseeed: 3\n")
    with pytest.raises(ConfigError, match="unknown config keys"):
        load(path)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    """``condition`` has no safe default."""
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\n")
    with pytest.raises(ConfigError, match="condition"):
        load(path)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    """A missing config raises our own error type, not OSError."""
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "absent.yaml")


def test_unknown_component_names_its_alternatives(tmp_path: Path) -> None:
    """The error tells you what you could have written instead."""
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\ncondition: carma\nbackbone:\n  kind: nonesuch\n")
    cfg = load(path)
    with pytest.raises(ConfigError, match="registered:"):
        build(cfg)


def test_build_wires_collaborators(smoke_config: ExperimentConfig) -> None:
    """The composition root binds the retriever and the arbiter."""
    assembly = build(smoke_config)
    assert assembly.retriever is not None
    # A bound arbiter can price a decision; an unbound one raises.
    assert hasattr(assembly.arbiter, "decide")
