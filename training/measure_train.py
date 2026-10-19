"""Run ``lerobot-train`` in-process and record peak GPU memory and wall time.

Every argument is passed through to ``lerobot-train`` unchanged, except two of
our own: ``--measure-out`` names the JSON summary (default
``<output_dir>.measure.json``), and ``--vram-limit-gib`` caps PyTorch's allocator.
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

import torch
from lerobot.scripts import lerobot_train


def _pop_arg(argv: list[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg.startswith(f"{name}="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def main() -> int:
    """Train, then write peak memory and wall time; record out-of-memory as a result."""
    out_arg = _pop_arg(sys.argv, "--measure-out")
    out = Path(out_arg) if out_arg is not None else None
    limit_arg = _pop_arg(sys.argv, "--vram-limit-gib")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if limit_arg is not None:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(limit_arg) / total_gib))
    output_dir = next(
        (Path(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--output_dir=")), None
    )
    if out is None and output_dir is not None:
        out = output_dir.parent / f"{output_dir.name}.measure.json"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    error: str | None = None
    try:
        lerobot_train.main()
    except torch.OutOfMemoryError as exc:
        error = f"OutOfMemoryError: {str(exc).splitlines()[0]}"
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
    }
    text = json.dumps(result, indent=2)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
