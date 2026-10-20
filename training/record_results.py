"""Copy a set of training measurements into the tracked record ``training/results/``.

``runs/`` is gitignored, so nothing there survives for a report. This makes a
dated folder ``training/results/<date>_<label>/`` holding the given JSON
results, each training log reduced to a loss curve (CSV), and ``env.json``:
the commit the code came from, whether the tree was dirty, the package
versions, the GPU and driver, and the pinned asset revisions. Then add a
dated entry to ``training/results/RESULTS.md`` by hand, citing that folder.

    python training/record_results.py --label smolvla-smoke \
        --json runs/training/sweep/sweep.json --log runs/training/smoke300.log
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata as md
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_LINE = re.compile(
    r"step:(\d+) smpl:(\S+) .*?loss:([\d.]+) grdn:([\d.]+) lr:([\d.e+-]+) "
    r"updt_s:([\d.]+) data_s:([\d.]+)"
)
PACKAGES = ("torch", "torchvision", "torchcodec", "lerobot", "transformers", "accelerate")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _environment() -> dict[str, object]:
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_training_assets import ASSETS

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "recorded": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: md.version(name) for name in PACKAGES},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "driver": driver or None,
        "assets": {name: f"{a.repo}@{a.revision}" for name, a in ASSETS.items()},
    }


def _curve(log: Path, dest: Path) -> int:
    rows = LOG_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "samples", "loss", "grad_norm", "lr", "updt_s", "data_s"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    """Create the dated results folder and print its path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="short kebab-case name of the record")
    parser.add_argument("--json", type=Path, nargs="*", default=[], help="result files to copy")
    parser.add_argument("--log", type=Path, nargs="*", default=[], help="lerobot-train logs")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "results")
    args = parser.parse_args()

    dest = args.out / f"{args.date}_{args.label}"
    dest.mkdir(parents=True, exist_ok=True)
    env = _environment()
    (dest / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    for src in args.json:
        shutil.copyfile(src, dest / src.name)
    for log in args.log:
        n = _curve(log, dest / f"{log.stem}.curve.csv")
        print(f"{log.name}: {n} logged steps")
    print(dest.relative_to(ROOT).as_posix())
    if env["dirty"]:
        print("note: working tree is dirty; say which commit holds the code", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
