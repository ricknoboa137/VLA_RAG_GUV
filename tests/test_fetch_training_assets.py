"""The training-asset fetcher stays pinned, open-licensed and offline-capable.

``scripts/fetch_training_assets.py`` is the only network step of the training
environment. These tests need neither the network nor that environment: they
check the asset table and the rewrite that lets SmolVLA load from ``assets/``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_training_assets.py"
_OPEN_LICENCES = {"Apache-2.0", "MIT", "BSD-3-Clause"}


@pytest.fixture(scope="module")
def fetch() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_training_assets", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_asset_is_pinned_to_a_commit(fetch: ModuleType) -> None:
    """A branch name would let two fetches of the same name differ."""
    for name, asset in fetch.ASSETS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", asset.revision), name


def test_every_asset_is_openly_licensed(fetch: ModuleType) -> None:
    """AGENTS.md section 2: open components only."""
    for name, asset in fetch.ASSETS.items():
        assert asset.licence in _OPEN_LICENCES, name


def test_localise_points_smolvla_at_the_local_vlm(fetch: ModuleType, tmp_path: Path) -> None:
    """After the rewrite nothing in the checkpoint names a Hub repository."""
    hub_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    (tmp_path / "config.json").write_text(
        json.dumps({"vlm_model_name": hub_id, "load_vlm_weights": True, "chunk_size": 50})
    )
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"registry_name": "to_batch_processor", "config": {}},
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": hub_id, "max_length": 48},
                    },
                ]
            }
        )
    )
    vlm_dir = Path("assets") / fetch.VLM_ASSET

    fetch._localise_smolvla(tmp_path, vlm_dir)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {
        "vlm_model_name": "assets/smolvlm2-500m-video-instruct",
        "load_vlm_weights": False,
        "chunk_size": 50,
    }
    steps = json.loads((tmp_path / "policy_preprocessor.json").read_text())["steps"]
    assert steps[0] == {"registry_name": "to_batch_processor", "config": {}}
    assert steps[1]["config"] == {
        "tokenizer_name": "assets/smolvlm2-500m-video-instruct",
        "max_length": 48,
    }


def test_localise_points_xvla_at_the_local_tokenizer(fetch: ModuleType, tmp_path: Path) -> None:
    """X-VLA would otherwise fetch facebook/bart-large's tokenizer at load time."""
    (tmp_path / "config.json").write_text(
        json.dumps({"tokenizer_name": "facebook/bart-large", "chunk_size": 30})
    )
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"registry_name": "rename_observations_processor", "config": {}},
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "facebook/bart-large", "max_length": 50},
                    },
                ]
            }
        )
    )

    fetch._localise_xvla(tmp_path, Path("assets") / fetch.TOKENIZER_ASSET)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {"tokenizer_name": "assets/bart-large-tokenizer", "chunk_size": 30}
    steps = json.loads((tmp_path / "policy_preprocessor.json").read_text())["steps"]
    assert steps[0] == {"registry_name": "rename_observations_processor", "config": {}}
    assert steps[1]["config"] == {"tokenizer_name": "assets/bart-large-tokenizer", "max_length": 50}


def test_every_checkpoint_that_names_a_hub_dependency_is_localised(fetch: ModuleType) -> None:
    """A SmolVLA or X-VLA checkpoint left unrewritten would reach the network at load."""
    for name, asset in fetch.ASSETS.items():
        if asset.repo_type != "model":
            continue
        if "smolvla" in name or "svla" in name:
            assert asset.localise == "smolvla", name
        if "xvla" in name:
            assert asset.localise == "xvla", name
