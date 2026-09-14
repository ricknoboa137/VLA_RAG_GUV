"""Run manifests.

A run without a manifest is not a result. The manifest records everything
needed to decide whether two runs may be compared: the resolved config and its
hash, the commit the code was at, whether the tree was dirty, the seed, the
platform, and the version of every installed dependency.

``carma report`` refuses to aggregate runs whose manifests disagree on config
hash. That refusal is the point; do not add a flag to override it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import yaml

from carma.config import ExperimentConfig

REQUIRED_FIELDS = (
    "run_id",
    "created_at",
    "config_hash",
    "config",
    "seed",
    "git_commit",
    "git_dirty",
    "python",
    "platform",
    "dependencies",
)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def _dependencies() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]
    }


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to reproduce or reject a run."""

    run_id: str
    created_at: str
    config_hash: str
    config: dict[str, Any]
    seed: int
    git_commit: str
    git_dirty: bool
    python: str
    platform: str
    dependencies: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary for serialisation."""
        return asdict(self)


def build_manifest(cfg: ExperimentConfig, run_id: str) -> Manifest:
    """Collect provenance for a run about to start."""
    return Manifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        config_hash=cfg.hash(),
        config=cfg.resolved(),
        seed=cfg.seed,
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        python=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=_dependencies(),
    )


def write_manifest(manifest: Manifest, run_dir: Path) -> Path:
    """Write ``manifest.yaml`` into ``run_dir`` and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=True))
    return path


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """Read and validate a manifest from a run directory."""
    path = run_dir / "manifest.yaml"
    if not path.exists():
        msg = f"no manifest in {run_dir}; the run is not usable as a result"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        msg = f"malformed manifest in {run_dir}"
        raise ValueError(msg)
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        msg = f"manifest in {run_dir} is missing: {', '.join(missing)}"
        raise ValueError(msg)
    return data
