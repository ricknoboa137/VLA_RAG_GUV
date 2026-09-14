"""Fail when a runtime dependency is not pinned to an exact version.

Reproducibility gate. A range specifier means two people running the same
commit can get different numbers, which defeats the manifest.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

PIN = re.compile(r"^[A-Za-z0-9._-]+(\[[^\]]+\])?==[^,\s]+$")


def main() -> int:
    """Return 0 when every dependency is exactly pinned, 1 otherwise."""
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    project = data.get("project", {})

    groups: dict[str, list[str]] = {"dependencies": project.get("dependencies", [])}
    for name, deps in project.get("optional-dependencies", {}).items():
        groups[f"optional:{name}"] = deps

    bad: list[str] = []
    for group, deps in groups.items():
        for dep in deps:
            if not PIN.match(dep.strip()):
                bad.append(f"{group}: {dep}")

    if bad:
        print("unpinned dependencies found:", file=sys.stderr)
        for item in bad:
            print(f"  {item}", file=sys.stderr)
        return 1

    total = sum(len(v) for v in groups.values())
    print(f"all {total} dependencies are exactly pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
