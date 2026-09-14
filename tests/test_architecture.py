"""Enforce the module boundaries declared in AGENTS.md section 3.

A boundary violation is caught here rather than in review, because the
no-memory baseline stops being honest the moment ``backbone`` can see
``memory``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "carma"

# package -> packages it may not import
FORBIDDEN: dict[str, set[str]] = {
    "types": {
        "perception",
        "memory",
        "backbone",
        "operator",
        "sim",
        "arbiter",
        "runtime",
        "metrics",
        "cli",
        "config",
        "registry",
    },
    "perception": {"arbiter", "runtime", "cli", "memory", "backbone"},
    "memory": {"arbiter", "runtime", "backbone", "cli"},
    "backbone": {"arbiter", "runtime", "memory", "cli"},
    "operator": {"arbiter", "runtime", "cli"},
    "sim": {"arbiter", "runtime", "memory", "backbone", "operator", "cli"},
    "arbiter": {"runtime", "cli"},
    "runtime": {"cli"},
    "metrics": {"runtime", "cli", "arbiter", "memory", "backbone", "sim", "operator"},
}


def _imported_carma_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("carma"):
            parts = node.module.split(".")
            if len(parts) > 1:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("carma."):
                    found.add(alias.name.split(".")[1])
    return found


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_respects_boundaries(package: str) -> None:
    """No module imports a package it is forbidden to depend on."""
    pkg_dir = SRC / package
    paths = sorted(pkg_dir.rglob("*.py")) if pkg_dir.is_dir() else [SRC / f"{package}.py"]
    violations: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for imported in _imported_carma_packages(path) & FORBIDDEN[package]:
            # Deferred imports inside a function body are the documented escape
            # hatch for the composition root only.
            violations.append(f"{path.relative_to(SRC)} imports carma.{imported}")
    assert not violations, "boundary violations:\n" + "\n".join(violations)


def test_composition_root_is_the_only_wiring_point() -> None:
    """Only ``config.build`` may import every registry."""
    text = (SRC / "config.py").read_text()
    assert "def build(" in text
    # The deferred imports must sit inside build(), not at module scope.
    module_scope = text.split("def build(")[0]
    assert "from carma.arbiter" not in module_scope
    assert "from carma.memory" not in module_scope
