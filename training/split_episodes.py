"""Split a LeRobot dataset's episodes into train, calibration and test, by seed.

    python training/split_episodes.py --dataset assets/so101_red_cube --seed 0 \
        --out training/splits/so101_red_cube_seed0.json

The split is a permutation of the episode indices drawn from
``np.random.default_rng(seed)``: the first ``held_out`` fraction is held out,
and that is halved into a calibration subset (where the uncertainty map is
fitted) and a test subset (where the models are scored). Whole episodes are
held out, never frames, so no test frame has a neighbour in training.

The output names the dataset's pinned revision and the episode count it was
drawn from, so a split cannot silently be applied to a different dataset.
Pure numpy; runs in either environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_training_assets import ASSETS  # a sibling script, found above


def split_episodes(n_episodes: int, seed: int, held_out: float = 0.2) -> dict[str, list[int]]:
    """Seeded train / calibration / test split of ``range(n_episodes)``.

    Args:
        n_episodes: How many episodes the dataset has.
        seed: Seed of the permutation.
        held_out: Fraction of episodes held out, rounded to the nearest even
            count so calibration and test get the same number.

    Returns:
        ``train``, ``calibration`` and ``test`` episode indices, each sorted.
    """
    if n_episodes < 4 or not 0.0 < held_out < 1.0:
        msg = f"need at least 4 episodes and 0 < held_out < 1, got {n_episodes}, {held_out}"
        raise ValueError(msg)
    n_held = 2 * max(1, round(n_episodes * held_out / 2))
    order = np.random.default_rng(seed).permutation(n_episodes)
    half = n_held // 2
    return {
        "train": sorted(int(i) for i in order[n_held:]),
        "calibration": sorted(int(i) for i in order[:half]),
        "test": sorted(int(i) for i in order[half:n_held]),
    }


def main() -> int:
    """Write the split as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="assets/<fetch entry>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads((args.dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    n_episodes = int(info["total_episodes"])
    asset = ASSETS[args.dataset.name]
    split = split_episodes(n_episodes, args.seed, args.held_out)
    record = {
        "dataset": asset.repo,
        "revision": asset.revision,
        "fetch_entry": args.dataset.name,
        "total_episodes": n_episodes,
        "seed": args.seed,
        "held_out": args.held_out,
        "method": "np.random.default_rng(seed).permutation(total_episodes); "
        "first n_held held out, first half of those calibration, second half test",
        **split,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.out}: {len(split['train'])} train, {len(split['calibration'])} calibration, "
        f"{len(split['test'])} test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
