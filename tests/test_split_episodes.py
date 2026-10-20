"""The seeded held-out episode split used by the arm-policy benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def split() -> ModuleType:
    path = _ROOT / "training" / "split_episodes.py"
    spec = importlib.util.spec_from_file_location("split_episodes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_is_a_disjoint_cover_with_equal_held_out_halves(split: ModuleType) -> None:
    parts = split.split_episodes(182, seed=0)
    assert (len(parts["train"]), len(parts["calibration"]), len(parts["test"])) == (146, 18, 18)
    every = parts["train"] + parts["calibration"] + parts["test"]
    assert sorted(every) == list(range(182))


def test_split_depends_only_on_the_seed(split: ModuleType) -> None:
    assert split.split_episodes(50, seed=3) == split.split_episodes(50, seed=3)
    assert split.split_episodes(50, seed=3) != split.split_episodes(50, seed=4)


def test_the_recorded_split_is_the_one_the_seed_gives(split: ModuleType) -> None:
    recorded = json.loads(
        (_ROOT / "training" / "splits" / "so101_red_cube_seed0.json").read_text(encoding="utf-8")
    )
    parts = split.split_episodes(recorded["total_episodes"], recorded["seed"], recorded["held_out"])
    assert {k: recorded[k] for k in parts} == parts


def test_too_few_episodes_is_rejected(split: ModuleType) -> None:
    with pytest.raises(ValueError, match="at least 4"):
        split.split_episodes(3, seed=0)
