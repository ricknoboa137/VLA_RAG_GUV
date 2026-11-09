#!/usr/bin/env python3
"""Record scripted-expert demonstrations from the lockstep Gazebo world.

    python scripts/record_nav_demos.py --out assets/nav_demos --episodes 40 --seed 100

Writes one folder per episode (PNG frames plus ``meta.json``; format in
``carma.runtime.demonstrations``) and a ``manifest.json`` naming the seed,
route parameters, disturbance settings and git commit, so the dataset can be
regenerated. Needs the world launched with ``lockstep:=true`` and
``carma_bridge`` running (``docs/simulation.md``). ``assets/`` is gitignored.
Convert for training with ``training/build_nav_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from carma.backbone.alley_expert import AlleyExpert
from carma.runtime.demonstrations import Disturbance, record_episode, write_episode
from carma.runtime.episodes import row_alley_episodes
from carma.sim.gazebo import GazeboAdapter
from carma.types.core import PhenologyStage


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    """Record the episodes; the return value is the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--stage", default="VEGETATIVE", choices=[s.name for s in PhenologyStage])
    parser.add_argument("--min-length", type=float, default=6.0)
    parser.add_argument("--max-length", type=float, default=14.0)
    parser.add_argument("--start-offset", type=float, default=0.4, help="max lateral start, m")
    parser.add_argument("--start-yaw", type=float, default=0.35, help="max start heading, rad")
    parser.add_argument("--disturb", type=float, default=0.02, help="burst chance per step")
    parser.add_argument("--no-disturb", action="store_true")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--bridge-url", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5555)
    args = parser.parse_args()

    disturbance = None if args.no_disturb else Disturbance(start_probability=args.disturb)
    route = {
        "min_length_m": args.min_length,
        "max_length_m": args.max_length,
        "start_offset_m": args.start_offset,
        "start_yaw_rad": args.start_yaw,
        "max_steps": 400,
    }
    stage = PhenologyStage[args.stage]
    specs = row_alley_episodes(
        args.episodes, rng=np.random.default_rng(args.seed), stage=stage, **route
    )
    args.out.mkdir(parents=True, exist_ok=True)
    sim = GazeboAdapter(
        control_hz=args.control_hz,
        stage=args.stage,
        timeout_s=300.0,
        bridge_url=args.bridge_url,
        bridge_port=args.bridge_port,
    )
    expert = AlleyExpert()
    written, frames = [], 0
    try:
        for index, spec in enumerate(specs):
            rng = np.random.default_rng(args.seed + 1000 + index)
            episode = record_episode(
                sim, expert, spec, seed=args.seed + index, rng=rng, disturbance=disturbance
            )
            write_episode(args.out, episode)
            written.append(spec.episode_id)
            frames += len(episode.steps)
            bursts = sum(s.disturbed for s in episode.steps)
            print(f"{spec.episode_id}: {len(episode.steps)} steps, {bursts} disturbed", flush=True)
    finally:
        sim.close()
    manifest = {
        "seed": args.seed,
        "commit": _commit(),
        "stage": args.stage,
        "control_hz": args.control_hz,
        "route": route,
        "disturbance": None if disturbance is None else asdict(disturbance),
        "episodes": written,
        "frames": frames,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"recorded {len(written)} episodes, {frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
