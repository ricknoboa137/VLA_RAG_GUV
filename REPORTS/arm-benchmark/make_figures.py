"""Figures for the arm benchmark report, drawn from the data in this folder alone.

    python make_figures.py            # reads ./data, writes ./figures

Reads data/test_predictions.jsonl and the two training curves
(data/*_train.curve.csv). Needs matplotlib and numpy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
res, out = here / "data", here / "figures"
out.mkdir(parents=True, exist_ok=True)
COLOR = {"smolvla": "#1f6f8b", "xvla": "#d1722e"}
LABEL = {"smolvla": "SmolVLA", "xvla": "X-VLA"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

rows = [json.loads(line) for line in (res / "test_predictions.jsonl").read_text().splitlines()]
by = {m: [r for r in rows if r["model"] == m] for m in COLOR}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


# 1. Error per joint
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(JOINTS))
for k, m in enumerate(COLOR):
    vals = [np.mean([r["per_joint_error_rad"][j] for r in by[m]]) for j in JOINTS]
    bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=COLOR[m], label=LABEL[m])
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
ax.set_xticks(x, [j.replace("_", " ") for j in JOINTS], rotation=20)
ax.set_ylabel("mean joint error (rad)")
ax.legend(frameon=False)
save(fig, "fig1_error_per_joint.png")

# 2. Paired per-episode difference
eps = sorted({r["episode"] for r in rows})
mean = {m: {e: np.mean([r["error_rad"] for r in by[m] if r["episode"] == e]) for e in eps} for m in COLOR}
diff = np.array([mean["smolvla"][e] - mean["xvla"][e] for e in eps])
order = np.argsort(diff)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.bar(range(len(eps)), diff[order], color=[COLOR["smolvla"] if d < 0 else COLOR["xvla"] for d in diff[order]])
ax.axhline(0, color="black", lw=0.8)
ax.axhline(diff.mean(), color="gray", ls="--", lw=1, label=f"mean {diff.mean():+.4f} rad")
ax.set_xticks(range(len(eps)), [str(eps[i]) for i in order], rotation=90, fontsize=8)
ax.set_xlabel("test episode (sorted)")
ax.set_ylabel("SmolVLA minus X-VLA (rad)")
ax.set_title("Below zero: SmolVLA closer to the demonstration", fontsize=9)
ax.legend(frameon=False)
save(fig, "fig2_paired_difference.png")

# 3. Training curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, m in zip(axes, COLOR):
    with (res / f"{m}_train.curve.csv").open() as fh:
        data = list(csv.DictReader(fh))
    ax.plot([int(d["step"]) for d in data], [float(d["loss"]) for d in data], color=COLOR[m])
    ax.set_title(LABEL[m], fontsize=10)
    ax.set_xlabel("step")
    ax.set_yscale("log")
axes[0].set_ylabel("training loss (log)")
fig.suptitle("Losses are different quantities; compare shape, not level", fontsize=9)
save(fig, "fig3_training_loss.png")

# 4. Does uncertainty order the errors?
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, m in zip(axes, COLOR):
    u = np.array([r["uncertainty"] for r in by[m]])
    e = np.array([r["error_rad"] for r in by[m]])
    ax.scatter(u, e, s=8, alpha=0.4, color=COLOR[m])
    bins = np.quantile(u, np.linspace(0, 1, 6))
    idx = np.clip(np.digitize(u, bins[1:-1]), 0, 4)
    ax.plot([u[idx == i].mean() for i in range(5)], [e[idx == i].mean() for i in range(5)],
            "o-", color="black", ms=4, lw=1, label="mean per fifth")
    ax.set_title(LABEL[m], fontsize=10)
    ax.set_xlabel("predicted uncertainty")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("chunk joint error (rad)")
save(fig, "fig4_uncertainty_vs_error.png")

# 5. Latency
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.boxplot([[r["inference_s"] for r in by[m]] for m in COLOR], tick_labels=[LABEL[m] for m in COLOR],
           vert=False, showfliers=False)
ax.set_xlabel("inference time per request (s), GPU shared with other processes")
save(fig, "fig5_latency.png")
print("figures:", sorted(p.name for p in out.glob("*.png")))
