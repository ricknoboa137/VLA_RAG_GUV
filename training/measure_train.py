"""Run ``lerobot-train`` in-process and record peak GPU memory and wall time.

Every argument is passed through to ``lerobot-train`` unchanged, except three of
our own: ``--measure-out`` names the JSON summary (default
``<output_dir>.measure.json``), ``--vram-limit-gib`` caps PyTorch's allocator, and
``--split`` trains on the ``train`` episodes of a split written by
``training/split_episodes.py``.

``--split`` does two things. It passes those episodes as ``--dataset.episodes``,
and it replaces the dataset's normalisation statistics for its numeric
features (state, action) with ones aggregated over the training episodes only.
LeRobot otherwise normalises with ``meta/stats.json``, computed over every
episode, so the held-out episodes' action range would reach the model.
Camera statistics are left alone: with ``use_imagenet_stats`` (the default)
they are ImageNet's, not the dataset's.

The cap matters on Windows: past the card's capacity the driver spills into
shared system memory instead of raising out-of-memory, so an oversized batch
"runs" several times slower rather than failing. With the cap it fails fast and
is recorded as not fitting. Steps per second are taken from LeRobot's own
per-step timers (``updt_s`` + ``data_s``) in the log, not from wall time, so model
loading and dataset indexing do not dilute the number.

    python training/measure_train.py --policy.path=assets/smolvla_base ... --measure-out x.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import windows_compat
from lerobot.scripts import lerobot_train


def _pop_arg(argv: list[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg.startswith(f"{name}="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def _train_only_stats(dataset: Any, episodes: list[int]) -> list[str]:
    """Replace ``dataset.meta.stats`` for numeric features with training-episode stats.

    Per-episode statistics are stored in ``meta/episodes/*.parquet``, which
    LeRobot loads without them, so they are read from the files. Rows are
    looked up by ``episode_index``, so stale duplicate rows a dataset may carry
    after re-recording (``so101_red_cube`` has 35, for episodes 132-166) are
    not counted; when an index has several rows, the first is the one LeRobot
    itself reads.

    Returns:
        The features whose statistics were replaced.
    """
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats

    files = sorted((Path(dataset.root) / "meta" / "episodes").glob("*/*.parquet"))
    rows: dict[int, dict[str, Any]] = {}
    for file in files:
        for row in pq.read_table(file).to_pylist():
            rows.setdefault(int(row["episode_index"]), row)
    keys = [
        key
        for key, feature in dataset.meta.features.items()
        if feature["dtype"] in ("float32", "float64")
        and key in dataset.meta.stats
        and f"stats/{key}/mean" in rows[episodes[0]]
    ]
    per_episode = [
        {
            key: {
                stat: np.asarray(rows[index][f"stats/{key}/{stat}"], dtype=np.float64)
                for stat in dataset.meta.stats[key]
                if f"stats/{key}/{stat}" in rows[index]
            }
            for key in keys
        }
        for index in episodes
    ]
    aggregated = aggregate_stats(per_episode)
    for key in keys:
        dataset.meta.stats[key] = aggregated[key]
    return keys


def _install_split(split_path: Path) -> list[int]:
    """Train on the split's ``train`` episodes, normalised by their own statistics."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    episodes = [int(i) for i in split["train"]]
    sys.argv.append(f"--dataset.episodes=[{','.join(str(i) for i in episodes)}]")
    original = lerobot_train.make_train_eval_datasets

    def make_with_train_stats(cfg: Any) -> Any:
        dataset, eval_dataset = original(cfg)
        replaced = _train_only_stats(dataset, episodes)
        print(f"split {split_path}: {len(episodes)} episodes, train-only stats for {replaced}")
        return dataset, eval_dataset

    lerobot_train.make_train_eval_datasets = make_with_train_stats
    return episodes


def main() -> int:
    """Train, then write peak memory and wall time; record out-of-memory as a result."""
    out_arg = _pop_arg(sys.argv, "--measure-out")
    out = Path(out_arg) if out_arg is not None else None
    split_arg = _pop_arg(sys.argv, "--split")
    split_episodes = _install_split(Path(split_arg)) if split_arg is not None else None
    limit_arg = _pop_arg(sys.argv, "--vram-limit-gib")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if limit_arg is not None:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(limit_arg) / total_gib))
    output_dir = next(
        (Path(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--output_dir=")), None
    )
    if out is None and output_dir is not None:
        out = output_dir.parent / f"{output_dir.name}.measure.json"

    windows_compat.install()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    error: str | None = None
    try:
        lerobot_train.main()
    except torch.OutOfMemoryError as exc:
        error = f"OutOfMemoryError: {str(exc).splitlines()[0]}"
    except OSError as exc:
        # A filesystem error after training (the checkpoints/last link was one,
        # before windows_compat) must not lose the measurement of a finished run.
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "argv": sys.argv[1:],
        "ok": error is None,
        "error": error,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(total_gib, 2),
        "vram_limit_gib": float(limit_arg) if limit_arg is not None else None,
        "split": split_arg,
        "train_episodes": len(split_episodes) if split_episodes is not None else None,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
