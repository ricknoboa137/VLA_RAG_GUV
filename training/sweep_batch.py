"""Batch-size sweep for SmolVLA fine-tuning on one GPU.

For each variant and batch size, runs a short ``lerobot-train`` through
``measure_train.py`` and records peak memory and steady-state throughput in
``<out>/sweep.json``. PyTorch's allocator is capped at ``--vram-budget-gib``
(see ``measure_train.py``): on Windows the driver otherwise spills to shared
system memory instead of raising OOM, which "fits" but runs several times
slower. A variant stops at the first batch size that runs out of memory.

    python training/sweep_batch.py --variant expert --batches 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS: dict[str, list[str]] = {
    # The upstream SmolVLA fine-tuning recipe: action expert trained, VLM frozen.
    # Precision is the checkpoint's own: VLM and expert weights in bf16, the
    # small state/action/time projections in fp32, no autocast.
    "expert": [],
    # VLM language layers trained as well; the vision encoder stays frozen.
    "full": ["--policy.train_expert_only=false"],
    # LeRobot's AMP switch. Kept to document that it fails for SmolVLA in 0.6.1:
    # with no policy dtype it autocasts to fp16 and its GradScaler cannot unscale
    # the bf16 gradients ("not implemented for 'BFloat16'").
    "expert_amp": ["--policy.use_amp=true"],
    "full_amp": ["--policy.train_expert_only=false", "--policy.use_amp=true"],
}

LOG_LINE = re.compile(r"step:(\d+) .*?loss:([\d.]+).*?updt_s:([\d.]+) data_s:([\d.]+)")


def _steady_state(log: str, skip: int) -> dict[str, float] | None:
    """Median step time over log windows after the first ``skip`` (warm-up)."""
    rows = [tuple(float(x) for x in m.groups()) for m in LOG_LINE.finditer(log)]
    rows = rows[skip:]
    if not rows:
        return None
    step_s = statistics.median(r[2] + r[3] for r in rows)
    return {
        "updt_s": round(statistics.median(r[2] for r in rows), 3),
        "data_s": round(statistics.median(r[3] for r in rows), 3),
        "steps_per_s": round(1.0 / step_s, 3),
        "last_loss": rows[-1][1],
    }


def main() -> int:
    """Run the sweep and write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 48, 64])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--log-freq", type=int, default=5)
    parser.add_argument("--vram-budget-gib", type=float, default=7.4)
    parser.add_argument("--policy", default="assets/smolvla_base")
    parser.add_argument("--dataset", default="assets/svla_so101_pickplace")
    parser.add_argument("--out", type=Path, default=Path("runs/training/sweep"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_file = args.out / "sweep.json"
    results: list[dict[str, object]] = (
        json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else []
    )

    for variant in args.variant:
        for batch in args.batches:
            name = f"{variant}_b{batch}"
            run_dir = args.out / name
            cmd = [
                sys.executable,
                str(HERE / "measure_train.py"),
                f"--policy.path={args.policy}",
                "--policy.push_to_hub=false",
                "--dataset.repo_id=lerobot/svla_so101_pickplace",
                f"--dataset.root={args.dataset}",
                '--rename_map={"observation.images.up": "observation.images.camera1", '
                '"observation.images.side": "observation.images.camera2"}',
                f"--batch_size={batch}",
                f"--steps={args.steps}",
                f"--log_freq={args.log_freq}",
                "--save_checkpoint=false",
                "--num_workers=2",
                "--seed=0",
                "--wandb.enable=false",
                f"--output_dir={run_dir.as_posix()}",
                f"--measure-out={(args.out / f'{name}.measure.json').as_posix()}",
                f"--vram-limit-gib={args.vram_budget_gib}",
                *VARIANTS[variant],
            ]
            print(f"== {name}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            log = proc.stdout + proc.stderr
            (args.out / f"{name}.log").write_text(log, encoding="utf-8")

            measure_file = args.out / f"{name}.measure.json"
            measure = (
                json.loads(measure_file.read_text(encoding="utf-8"))
                if measure_file.exists()
                else {"ok": False, "error": f"exit {proc.returncode}, see {name}.log"}
            )
            row: dict[str, object] = {"variant": variant, "batch": batch, **measure}
            row.pop("argv", None)
            speed = _steady_state(log, skip=1)
            if speed is not None:
                row.update(speed)
                row["samples_per_s"] = round(speed["steps_per_s"] * batch, 1)
            row["fits"] = bool(row.get("ok"))
            print(json.dumps(row), flush=True)
            results = [r for r in results if (r["variant"], r["batch"]) != (variant, batch)]
            results.append(row)
            summary_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            if not row["fits"]:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
