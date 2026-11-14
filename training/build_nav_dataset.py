"""Convert recorded navigation demonstrations into a LeRobot dataset.

    python training/build_nav_dataset.py --demos assets/nav_demos --out assets/nav_alley_v1

Runs in the ``carma-train`` environment (needs LeRobot). Reads what
``scripts/record_nav_demos.py`` wrote and produces:

* ``observation.images.front``: the left camera frame, HWC RGB uint8 (the
  recorder stores BGR, the OpenCV convention; the conversion is here, at the
  boundary, as AGENTS.md section 7 requires).
* ``observation.state``: ``[x, y, yaw]`` in metres and radians, in the world
  frame with the start pose included.
* ``action``: ``[linear_mps, angular_radps, stop]``; ``stop`` is 0 or 1 and a
  serving adapter thresholds it at 0.5.
* ``task``: the episode's instruction text.

The *label* (the expert's command) is the action, never the disturbed command
that was executed, so recovery from disturbed states is what gets learned.
Images are stored as PNG inside the dataset (``use_videos=False``): the set is
small and lossless frames keep training bit-faithful to what the policy sees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 10
IMAGE_KEY = "observation.images.front"


def _features(height: int, width: int) -> dict[str, dict[str, object]]:
    return {
        IMAGE_KEY: {"dtype": "image", "shape": (height, width, 3), "names": ["h", "w", "c"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "yaw"]},
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["linear_mps", "angular_radps", "stop"],
        },
    }


def convert(demos: Path, out: Path, repo_id: str, stop_hold: int = 0) -> tuple[int, int]:
    """Write the dataset; returns (episodes, frames).

    ``stop_hold`` repeats each episode's final (stop) frame that many extra times. The
    platform is stationary once it stops, so the repeats are what the camera would
    keep showing; without them stop is one frame in 100 to 260 and is never learned.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((demos / "manifest.json").read_text(encoding="utf-8"))
    names: list[str] = [
        name
        for name in manifest["episodes"]
        if not json.loads((demos / name / "meta.json").read_text(encoding="utf-8")).get("collided")
    ]
    if not names:
        raise ValueError(f"every episode in {demos} ended in a collision")
    first = cv2.imread(str(demos / names[0] / "frames" / "000000.png"))
    height, width = first.shape[:2]
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(manifest.get("control_hz", FPS))),
        features=_features(height, width),
        root=out,
        robot_type="wheeled_ugv",
        use_videos=False,
    )
    frames = 0
    for name in names:
        meta = json.loads((demos / name / "meta.json").read_text(encoding="utf-8"))
        for index, step in enumerate(meta["steps"]):
            bgr = cv2.imread(str(demos / name / "frames" / f"{index:06d}.png"))
            if bgr is None:
                raise FileNotFoundError(f"missing frame {index} of {name}")
            linear, angular, stop = step["label"]
            dataset.add_frame(
                {
                    IMAGE_KEY: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    "observation.state": np.asarray(step["pose"], dtype=np.float32),
                    "action": np.asarray([linear, angular, float(stop)], dtype=np.float32),
                    "task": meta["instruction"],
                }
            )
            frames += 1
            if index == len(meta["steps"]) - 1 and stop and stop_hold > 0:
                for _ in range(stop_hold):
                    dataset.add_frame(
                        {
                            IMAGE_KEY: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                            "observation.state": np.asarray(step["pose"], dtype=np.float32),
                            "action": np.asarray([linear, angular, 1.0], dtype=np.float32),
                            "task": meta["instruction"],
                        }
                    )
                    frames += 1
        dataset.save_episode()
    dataset.finalize()
    return len(names), frames


def main() -> int:
    """Convert and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="carma/nav_alley_v1")
    parser.add_argument("--stop-hold", type=int, default=0, help="extra copies of the stop frame")
    args = parser.parse_args()
    episodes, frames = convert(args.demos, args.out, args.repo_id, args.stop_hold)
    print(f"{episodes} episodes, {frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
