"""Record expert demonstrations for fine-tuning a navigation policy.

A demonstration is the sequence of (image, pose, instruction) the robot saw and
the action the expert *would take* there. To teach recovery, the executed
command is sometimes disturbed (a steering burst), while the stored label stays
the expert's own command computed on the state actually reached. Training on
those pairs shows the policy how to get back from states a clean demonstration
never visits (DART-style noise injection).

Everything is seeded: the disturbance schedule comes from the ``rng`` handed
in, so the same ``(config, seed, commit)`` records the same dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from carma.types.core import Action, EpisodeSpec, Observation, Pose
from carma.types.protocols import NavigationBackbone, SimAdapter

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Disturbance:
    """When and how the executed steering is disturbed.

    Attributes:
        start_probability: Chance per step, while idle, that a burst begins.
        min_steps: Shortest burst, in control steps.
        max_steps: Longest burst, in control steps.
        magnitude_radps: Yaw rate in rad/s added during a burst; sign is random.
    """

    start_probability: float = 0.02
    min_steps: int = 3
    max_steps: int = 8
    magnitude_radps: float = 0.2


@dataclass(frozen=True)
class DemoStep:
    """One recorded step.

    Attributes:
        rgb: Image seen before acting, HWC BGR uint8.
        pose: Pose seen before acting, metres and radians.
        t: Simulated seconds.
        label: The expert's command on this state; the training target.
        executed: What was actually sent to the simulator (label plus disturbance).
        disturbed: True while a steering burst was active.
    """

    rgb: np.ndarray  # type: ignore[type-arg]
    pose: Pose
    t: float
    label: Action
    executed: Action
    disturbed: bool


@dataclass
class DemoEpisode:
    """A recorded episode: its spec and steps, the final pose, and whether it ended in a crash.

    A collided episode is kept on disk so the rate is visible, and left out of
    training by ``training/build_nav_dataset.py``: its last frames show the
    expert's "keep going" label against a crop row.
    """

    spec: EpisodeSpec
    seed: int
    steps: list[DemoStep] = field(default_factory=list)
    final_pose: Pose | None = None
    collided: bool = False


def record_episode(
    sim: SimAdapter,
    expert: NavigationBackbone,
    spec: EpisodeSpec,
    *,
    seed: int,
    rng: np.random.Generator,
    disturbance: Disturbance | None,
) -> DemoEpisode:
    """Drive one episode with the expert and record every step.

    The episode ends when the expert declares stop, the robot collides, or
    ``spec.max_steps`` is hit. The stop step is recorded too, so the policy
    learns when to stop.
    """
    episode = DemoEpisode(spec=spec, seed=seed)
    obs: Observation = sim.reset(seed=seed, start=spec.start)
    obs = _with_instruction(obs, spec.instruction)
    remaining, sign = 0, 1.0
    for _ in range(spec.max_steps):
        if obs.collided:
            episode.collided = True
            break
        label, _ = expert.act(obs, (), rng=rng)
        if (
            disturbance is not None
            and not label.stop
            and remaining == 0
            and rng.random() < disturbance.start_probability
        ):
            remaining = int(rng.integers(disturbance.min_steps, disturbance.max_steps + 1))
            sign = 1.0 if rng.random() < 0.5 else -1.0
        disturbed = remaining > 0 and not label.stop
        executed = label
        if disturbed and disturbance is not None:
            executed = Action(label.linear, label.angular + sign * disturbance.magnitude_radps)
            remaining -= 1
        episode.steps.append(DemoStep(obs.rgb, obs.pose, obs.t, label, executed, disturbed))
        if label.stop:
            break
        obs = _with_instruction(sim.step(executed), spec.instruction)
    episode.final_pose = obs.pose
    return episode


def write_episode(root: Path, episode: DemoEpisode) -> Path:
    """Write one episode as ``<root>/<episode_id>/{meta.json, frames/NNNNNN.png}``.

    PNG is lossless, so the converter sees exactly what the policy will see.
    """
    folder = root / episode.spec.episode_id
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(episode.steps):
        if not cv2.imwrite(str(folder / "frames" / f"{index:06d}.png"), step.rgb):
            raise OSError(f"could not write frame {index} of {episode.spec.episode_id}")
    spec = episode.spec
    meta = {
        "schema": SCHEMA_VERSION,
        "episode_id": spec.episode_id,
        "seed": episode.seed,
        "instruction": spec.instruction,
        "stage": spec.stage.name,
        "start": [spec.start.x, spec.start.y, spec.start.yaw],
        "goal": [spec.goal.x, spec.goal.y, spec.goal.yaw],
        "collided": episode.collided,
        "final_pose": None
        if episode.final_pose is None
        else [episode.final_pose.x, episode.final_pose.y, episode.final_pose.yaw],
        "steps": [
            {
                "t": s.t,
                "pose": [s.pose.x, s.pose.y, s.pose.yaw],
                "label": [s.label.linear, s.label.angular, s.label.stop],
                "executed": [s.executed.linear, s.executed.angular],
                "disturbed": s.disturbed,
            }
            for s in episode.steps
        ],
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def _with_instruction(obs: Observation, text: str) -> Observation:
    return replace(obs, instruction=text)
