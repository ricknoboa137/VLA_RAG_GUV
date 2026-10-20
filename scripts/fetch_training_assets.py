"""Download the open VLA weights and datasets used by ``training/`` into ``assets/``.

Sibling of ``fetch_models.py`` for the fine-tuning environment, which runs in its
own interpreter (see ``training/README.md``). Every entry is openly licensed,
ungated and pinned to a commit, so two fetches of the same name give the same
bytes. Do not add a gated repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

VLM_ASSET = "smolvlm2-500m-video-instruct"


@dataclass(frozen=True)
class Asset:
    """One Hugging Face repository snapshot, pinned to a commit."""

    repo: str
    revision: str
    licence: str
    repo_type: str = "model"
    # None fetches every file; otherwise only files matching these globs.
    allow_patterns: tuple[str, ...] | None = None


# name -> asset. Kept in one place so the licence audit is a diff.
ASSETS: dict[str, Asset] = {
    # SmolVLA pretrained policy, about 0.45B parameters.
    "smolvla_base": Asset(
        repo="lerobot/smolvla_base",
        revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.safetensors", "README.md"),
    ),
    # SmolVLA builds its VLM from this repo's config and processor. Its weights
    # come from smolvla_base, so the 7.5 GB of checkpoints here are skipped.
    VLM_ASSET: Asset(
        repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.txt", "README.md"),
    ),
    # SO-101 pick-and-place, LeRobot dataset v3.0: the smoke-test fine-tuning set.
    "svla_so101_pickplace": Asset(
        repo="lerobot/svla_so101_pickplace",
        revision="f641879e22172be7e8161d5e6c1503c2d2feb657",
        licence="Apache-2.0",
        repo_type="dataset",
    ),
}


def _localise_smolvla(dest: Path, vlm_dir: Path) -> None:
    """Point a fetched SmolVLA checkpoint at the local VLM instead of the Hub.

    Upstream names its VLM and tokenizer by Hub id and sets ``load_vlm_weights``,
    which downloads the full VLM at load time only to overwrite it with the
    policy's own weights. Rewriting the three fields lets training and inference
    run offline; the policy weights themselves are untouched and load strictly.
    """
    vlm = vlm_dir.as_posix()

    config_file = dest / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["vlm_model_name"] = vlm
    config["load_vlm_weights"] = False
    config_file.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    pre_file = dest / "policy_preprocessor.json"
    pre = json.loads(pre_file.read_text(encoding="utf-8"))
    for step in pre["steps"]:
        if step["registry_name"] == "tokenizer_processor":
            step["config"]["tokenizer_name"] = vlm
    pre_file.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"localised {dest.name}: VLM and tokenizer read from {vlm}")


def main() -> int:
    """Fetch the requested asset into ``assets/<name>``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(ASSETS), help="asset to fetch")
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("assets"),
        help="destination directory (gitignored)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("run this from the training environment (training/README.md)", file=sys.stderr)
        return 1

    asset = ASSETS[args.name]
    dest = args.assets / args.name
    if dest.exists():
        print(f"{dest} already present")
        return 0

    print(f"fetching {asset.repo}@{asset.revision[:12]} ({asset.licence}) -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.repo,
        repo_type=asset.repo_type,
        revision=asset.revision,
        allow_patterns=list(asset.allow_patterns) if asset.allow_patterns else None,
        local_dir=str(dest),
    )
    if args.name == "smolvla_base":
        _localise_smolvla(dest, args.assets / VLM_ASSET)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
