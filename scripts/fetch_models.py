"""Download open model weights into ``assets/``.

Fetching is explicit and auditable so that no test or experiment silently
reaches the network. Everything listed here is openly licensed and downloadable
without an account; do not add a gated checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# name -> (repo id, licence). Kept in one place so the licence audit is a diff.
MODELS: dict[str, tuple[str, str]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "Apache-2.0"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "Apache-2.0"),
}


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

    repo, licence = MODELS[args.name]
    dest = args.assets / args.name
    if dest.exists():
        print(f"{dest} already present")
        return 0

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "install the 'retrieval' extra first: pip install -e '.[retrieval]'",
            file=sys.stderr,
        )
        return 1

    print(f"fetching {repo} ({licence}) -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(repo).save(str(dest))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
