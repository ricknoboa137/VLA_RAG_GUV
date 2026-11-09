#!/usr/bin/env python3
"""Check that a running lockstep Gazebo world replays an episode exactly.

    python scripts/gazebo_replay_check.py --episodes 3 --steps 30

Runs the same scripted action sequence through ``GazeboAdapter`` several times
(each preceded by a reset) and compares every pose, image and depth map with
the first episode's. Exits 1 if anything differs. Needs the world launched with
``lockstep:=true`` and ``carma_bridge`` running; see ``docs/simulation.md``.
Repeat it after changing the bridge, the world, the physics step or the sensor
rates, since any of them can break replay without breaking a single test.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from carma.sim.gazebo import GazeboAdapter
from carma.types.core import Action, Observation, Pose


def _episode(sim: GazeboAdapter, steps: int, seed: int) -> list[Observation]:
    observations = [sim.reset(seed=seed, start=Pose(0.0, 0.0, 0.0))]
    for index in range(steps):
        turn = 0.1 if index % 10 < 5 else -0.1
        observations.append(sim.step(Action(linear=0.5, angular=turn)))
    return observations


def _differences(reference: list[Observation], other: list[Observation]) -> list[str]:
    problems: list[str] = []
    for index, (a, b) in enumerate(zip(reference, other, strict=True)):
        if a.pose != b.pose:
            problems.append(f"frame {index}: pose {a.pose} != {b.pose}")
        if not np.array_equal(a.rgb, b.rgb):
            worst = int(np.abs(a.rgb.astype(int) - b.rgb.astype(int)).max())
            problems.append(f"frame {index}: image differs by up to {worst} grey levels")
        if not np.array_equal(a.depth, b.depth, equal_nan=True):
            problems.append(f"frame {index}: depth differs")
    return problems


def main() -> int:
    """Run the check and report; the return value is the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=3, help="episodes to compare")
    parser.add_argument("--steps", type=int, default=30, help="steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--bridge-url", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=5555)
    args = parser.parse_args()

    sim = GazeboAdapter(
        control_hz=args.control_hz,
        timeout_s=300.0,
        bridge_url=args.bridge_url,
        bridge_port=args.bridge_port,
    )
    try:
        episodes = [_episode(sim, args.steps, args.seed) for _ in range(args.episodes)]
    finally:
        sim.close()

    failed = False
    for number, episode in enumerate(episodes[1:], start=2):
        problems = _differences(episodes[0], episode)
        status = "identical" if not problems else f"{len(problems)} difference(s)"
        print(f"episode {number} against episode 1: {status}")
        for line in problems[:10]:
            print(f"  {line}")
        failed = failed or bool(problems)
    frames = args.episodes * (args.steps + 1)
    print(f"{'FAIL' if failed else 'PASS'}: {frames} frames over {args.episodes} episodes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
