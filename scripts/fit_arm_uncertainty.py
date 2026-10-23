"""Fit each arm policy's logistic uncertainty map on benchmark rows from calibration episodes.

    python scripts/fit_arm_uncertainty.py \
        --predictions training/results/<date>_arm-benchmark/calibration/predictions.jsonl \
        --out training/results/<date>_arm-benchmark/uncertainty_fit.json

Reads the ``predictions.jsonl`` that ``scripts/benchmark_arm_policies.py``
writes, and for every model fits ``u = sigmoid(a * ln(spread) + b)`` to the
event "this chunk's mean joint error exceeded ``tolerance_rad``"
(:func:`arm_policy.fit_logistic`). The tolerance is one number for every
model: by default the median chunk error over all models' calibration rows
pooled, so the event is neither rare nor certain and the same for both. The
fitted ``a``, ``b`` and ``tolerance_rad`` go into each model's spec as
``uncertainty: {method: logistic, ...}``.

Fit only on episodes kept out of training and out of the test set
(``training/split_episodes.py``), or the calibration is circular.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_policy import UncertaintyMap, fit_logistic  # a sibling script, found above


def main() -> int:
    """Fit and write one map per model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tolerance-rad", type=float, default=None, help="default: pooled median")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if np.isfinite(r["error_rad"]) and (r["spread"] or 0.0) > 0.0]
    tolerance = (
        args.tolerance_rad
        if args.tolerance_rad is not None
        else float(np.median([r["error_rad"] for r in rows]))
    )
    fits: dict[str, object] = {}
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        spreads = np.array([r["spread"] for r in mine])
        exceeded = np.array([r["error_rad"] > tolerance for r in mine])
        a, b = fit_logistic(spreads, exceeded)
        u = UncertaintyMap(method="logistic", a=a, b=b, tolerance_rad=tolerance)
        predicted = np.array([u(float(s)) for s in spreads])
        fits[model] = {
            "a": a,
            "b": b,
            "frames": len(mine),
            "exceeded_rate": float(exceeded.mean()),
            "mean_predicted": float(predicted.mean()),
            "brier": float(np.mean((predicted - exceeded) ** 2)),
            "brier_constant": float(exceeded.mean() * (1.0 - exceeded.mean())),
        }
    record = {
        "predictions": args.predictions.as_posix(),
        "tolerance_rad": tolerance,
        "event": "chunk mean joint error > tolerance_rad",
        "map": "u = sigmoid(a * ln(spread) + b)",
        "models": fits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
