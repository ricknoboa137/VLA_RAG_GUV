"""Download open model weights into ``assets/``.

Fetching is explicit and auditable so that no test or experiment silently
reaches the network. Everything listed here is openly licensed and downloadable
without an account; do not add a gated checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# name -> (repo id, licence, kind). Kept in one place so the licence audit is a diff.
MODELS: dict[str, tuple[str, str, str]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0", "sentence-transformers"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "Apache-2.0", "sentence-transformers"),
    # Speech-to-text for operator utterances (CTranslate2 conversions of Whisper).
    "faster-whisper-base": ("Systran/faster-whisper-base", "MIT", "faster-whisper"),
    "faster-whisper-small": ("Systran/faster-whisper-small", "MIT", "faster-whisper"),
}


def _fetch_sentence_transformer(repo: str, dest: Path) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("install the 'retrieval' extra first: pip install -e '.[retrieval]'", file=sys.stderr)
        return 1
    SentenceTransformer(repo).save(str(dest))
    return 0


def _fetch_snapshot(repo: str, dest: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("install the 'voice' extra first: pip install -e '.[voice]'", file=sys.stderr)
        return 1
    snapshot_download(repo_id=repo, local_dir=str(dest))
    return 0


def main() -> int:
    """Fetch the requested model into ``assets/<name>``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(MODELS), help="model to fetch")
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("assets"),
        help="destination directory (gitignored)",
    )
    args = parser.parse_args()

    repo, licence, kind = MODELS[args.name]
    dest = args.assets / args.name
    if dest.exists():
        print(f"{dest} already present")
        return 0

    print(f"fetching {repo} ({licence}) -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch = _fetch_sentence_transformer if kind == "sentence-transformers" else _fetch_snapshot
    status = fetch(repo, dest)
    if status == 0:
        print("done")
    return status


if __name__ == "__main__":
    sys.exit(main())
