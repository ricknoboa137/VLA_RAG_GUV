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
TOKENIZER_ASSET = "bart-large-tokenizer"


@dataclass(frozen=True)
class Asset:
    """One Hugging Face repository snapshot, pinned to a commit."""

    repo: str
    revision: str
    licence: str
    repo_type: str = "model"
    # None fetches every file; otherwise only files matching these globs.
    allow_patterns: tuple[str, ...] | None = None
    # How the fetched checkpoint is pointed at local files: "smolvla", "xvla" or None.
    localise: str | None = None


# name -> asset. Kept in one place so the licence audit is a diff.
ASSETS: dict[str, Asset] = {
    # SmolVLA pretrained policy, about 0.45B parameters.
    "smolvla_base": Asset(
        repo="lerobot/smolvla_base",
        revision="5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.safetensors", "README.md"),
        localise="smolvla",
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
    # The arm-policy benchmark pair: SmolVLA and X-VLA fine-tuned by the same
    # person on the same SO-101 dataset (pick the red cube, top and gripper
    # cameras, 20 000 steps each). Community checkpoints, used to exercise the
    # serving path; the fair benchmark fine-tunes both bases here instead.
    "svla_so101_red_cube": Asset(
        repo="whosricky/svla-so101-pick-red-cube-2cam",
        revision="5302eecfe373a3225642c3c9c94e385d6e32c139",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.safetensors", "README.md"),
        localise="smolvla",
    ),
    "xvla_so101_red_cube": Asset(
        repo="whosricky/xvla-so101-pick-red-cube-2cam",
        revision="7a007f64dd5910dd5cde76c8d31d9d7f7e6c0778",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.safetensors", "README.md"),
        localise="xvla",
    ),
    # X-VLA pretrained policy, about 0.9B parameters: the base to fine-tune from.
    "xvla_base": Asset(
        repo="lerobot/xvla-base",
        revision="cdb7964e4fe842935d671bfab5a5ebe00a96648c",
        licence="Apache-2.0",
        allow_patterns=("*.json", "*.safetensors", "README.md"),
        localise="xvla",
    ),
    # X-VLA tokenises its instruction with BART's tokenizer. Only the tokenizer
    # files; the 1-2 GB weight files in four formats are not needed.
    TOKENIZER_ASSET: Asset(
        repo="facebook/bart-large",
        revision="cb48c1365bd826bd521f650dc2e0940aee54720c",
        licence="Apache-2.0",
        allow_patterns=("config.json", "tokenizer*.json", "vocab.json", "merges.txt", "README.md"),
    ),
    # The dataset both red-cube checkpoints were trained on (published as
    # whosricky/pick-red-cube-v3, since renamed): 182 episodes, 30 fps.
    "so101_red_cube": Asset(
        repo="whosricky/so101_pick_mixed_cubes_2cams",
        revision="f77f31614b10b9cf4ceb4c5f4c4edb503b12b74c",
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


def _localise_xvla(dest: Path, tokenizer_dir: Path) -> None:
    """Point a fetched X-VLA checkpoint at the local BART tokenizer instead of the Hub.

    X-VLA builds its Florence-2 backbone from the config embedded in its own
    ``config.json``, so the tokenizer is the only thing it would download.
    """
    tokenizer = tokenizer_dir.as_posix()

    config_file = dest / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["tokenizer_name"] = tokenizer
    config_file.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    pre_file = dest / "policy_preprocessor.json"
    pre = json.loads(pre_file.read_text(encoding="utf-8"))
    for step in pre["steps"]:
        if "tokenizer_name" in step.get("config", {}):
            step["config"]["tokenizer_name"] = tokenizer
    pre_file.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
    print(f"localised {dest.name}: tokenizer read from {tokenizer}")


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
    if asset.localise == "smolvla":
        _localise_smolvla(dest, args.assets / VLM_ASSET)
    elif asset.localise == "xvla":
        _localise_xvla(dest, args.assets / TOKENIZER_ASSET)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
