"""A run without a manifest is not a result."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carma.cli.manifest import REQUIRED_FIELDS, build_manifest, read_manifest, write_manifest
from carma.config import ExperimentConfig


def test_manifest_has_every_required_field(smoke_config: ExperimentConfig, tmp_path: Path) -> None:
    """Provenance is complete enough to reject an incomparable run."""
    manifest = build_manifest(smoke_config, run_id="test-run")
    write_manifest(manifest, tmp_path)
    data = read_manifest(tmp_path)
    for field in REQUIRED_FIELDS:
        assert field in data


def test_missing_manifest_raises(tmp_path: Path) -> None:
    """A run directory with no manifest cannot be used."""
    with pytest.raises(FileNotFoundError, match="not usable as a result"):
        read_manifest(tmp_path)


def test_incomplete_manifest_raises(tmp_path: Path) -> None:
    """A truncated manifest is rejected, naming what is missing."""
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"run_id": "x"}))
    with pytest.raises(ValueError, match="missing"):
        read_manifest(tmp_path)


def test_manifest_records_dependency_versions(smoke_config: ExperimentConfig) -> None:
    """Dependency drift is the commonest cause of unreproducible numbers."""
    manifest = build_manifest(smoke_config, run_id="test-run")
    assert "numpy" in {name.lower() for name in manifest.dependencies}
