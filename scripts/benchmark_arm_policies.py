"""Compare arm policies open loop on recorded SO-101 episodes, through their servers.

    # one server per model, e.g. ports 8000 and 8001 (scripts/policy_server.py)
    python scripts/benchmark_arm_policies.py \
        --dataset assets/so101_red_cube --episodes 0 60 120 180 \
        --server smolvla=http://127.0.0.1:8000/act --server xvla=http://127.0.0.1:8001/act \
        --out training/results/<date>_arm-benchmark

Each sampled frame of each episode becomes one ``ActRequest`` -- the frames,
the recorded joint state, the episode's task as the instruction, a fixed seed
-- posted to every server in turn. Requests go over the same wire as the live
pipeline, so what is measured is what would drive the arm. Each answer's chunk
is scored against the actions the demonstrator recorded over the same steps.

Every model sees identical requests; the summary refuses to compare models
whose rows cover different frames. Joint units are converted once, with the
units of ``--units-spec``, for the state sent and the truth scored, and the
test suite pins both specs to the same units.

Open loop only: this measures imitation of the recording, not task success
(``carma.metrics.arm``). And it is honest only on episodes the models were not
trained on. Pass ``--trained-on`` when they were, and the summary says so.

Runs in the ``carma-train`` environment (LeRobot decodes the dataset's video)
with ``PYTHONPATH=src``. Writes ``predictions.jsonl`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from carma.backbone.remote import HttpTransport
from carma.backbone.wire import ActRequest, ActResponse, encode_image
from carma.metrics.arm import latency_summary, mean_joint_error, rank_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_policy import load_spec  # a sibling script, found through the line above


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _bgr(chw_rgb: Any) -> np.ndarray:
    """A LeRobot image tensor (C, H, W, RGB, [0, 1]) as the wire's BGR uint8."""
    rgb = (chw_rgb.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def main() -> int:
    """Run every server over the sampled frames and write rows and a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="local LeRobot dataset")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=30, help="score every Nth frame")
    parser.add_argument("--horizon", type=int, default=25, help="steps scored per chunk")
    parser.add_argument("--server", action="append", required=True, help="name=url, once per model")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="dataset camera=request source; default top=world_left, gripper=image",
    )
    parser.add_argument(
        "--units-spec", type=Path, default=Path("configs/arm_policy/svla_so101_red_cube.yaml")
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--trained-on", action="store_true", help="the models saw these episodes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    servers = dict(item.split("=", 1) for item in args.server)
    cameras = dict(
        item.split("=", 1) for item in (args.camera or ["top=world_left", "gripper=image"])
    )
    spec = load_spec(args.units_spec)
    joints = spec.joints
    info = json.loads((args.dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    dataset = LeRobotDataset(
        repo_id=f"local/{args.dataset.name}",
        root=args.dataset,
        episodes=args.episodes,
        delta_timestamps={"action": [i / fps for i in range(args.horizon)]},
    )
    transports = {name: HttpTransport(url) for name, url in servers.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index in range(len(dataset)):
        item = dataset[index]
        frame = int(item["frame_index"])
        if frame % args.stride:
            continue
        episode = int(item["episode_index"])
        state_model = item["observation.state"].numpy().astype(np.float64)
        truth_model = item["action"].numpy().astype(np.float64)
        valid = ~item["action_is_pad"].numpy().astype(bool)
        primary_key = next(k for k, src in cameras.items() if src == "image")
        views = tuple(
            (src, encode_image(_bgr(item[f"observation.images.{key}"])))
            for key, src in cameras.items()
            if src != "image"
        )
        request = ActRequest(
            image=encode_image(_bgr(item[f"observation.images.{primary_key}"])),
            instruction=str(item["task"]),
            pose=(0.0, 0.0, 0.0),
            t=frame / fps,
            stage="VEGETATIVE",
            seed=args.seed + episode * 100_000 + frame,
            samples=args.samples,
            views=views,
            arm_state=tuple(
                (j.name, float(j.to_rad(np.float64(v))))
                for j, v in zip(joints, state_model, strict=True)
            ),
        )
        truth_rad = np.stack([j.to_rad(truth_model[:, i]) for i, j in enumerate(joints)], axis=1)
        for name, transport in transports.items():
            sent = time.monotonic()
            response = ActResponse.from_json(transport.post(request.to_json(), args.timeout_s))
            round_trip_s = time.monotonic() - sent
            if response.arm is None:
                msg = f"{name} answered without an arm chunk"
                raise SystemExit(msg)
            predicted = np.asarray(response.arm.positions_rad, dtype=np.float64)
            steps = min(len(predicted), args.horizon)
            rows.append(
                {
                    "model": name,
                    "model_id": response.model_id,
                    "model_revision": response.model_revision,
                    "episode": episode,
                    "frame": frame,
                    "seed": request.seed,
                    "steps": steps,
                    "error_rad": mean_joint_error(
                        predicted[:steps], truth_rad[:steps], valid[:steps]
                    ),
                    "first_step_error_rad": mean_joint_error(
                        predicted[:1], truth_rad[:1], valid[:1]
                    ),
                    "per_joint_error_rad": {
                        j.name: float(
                            np.abs(predicted[:steps, i] - truth_rad[:steps, i])[
                                valid[:steps]
                            ].mean()
                        )
                        for i, j in enumerate(joints)
                    },
                    "uncertainty": response.uncertainty,
                    "uncertainty_method": response.uncertainty_method,
                    "spread": response.info.get("spread"),
                    "inference_s": response.inference_s,
                    "round_trip_s": round_trip_s,
                }
            )
        print(f"episode {episode} frame {frame}: {len(rows)} rows", flush=True)

    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    frames_by_model = {
        name: sorted((r["episode"], r["frame"]) for r in rows if r["model"] == name)
        for name in servers
    }
    if len({tuple(v) for v in frames_by_model.values()}) != 1:
        msg = "models were scored on different frames; refusing to compare them"
        raise SystemExit(msg)

    models: dict[str, Any] = {}
    for name in servers:
        mine = [r for r in rows if r["model"] == name]
        errors = [r["error_rad"] for r in mine]
        models[name] = {
            "model_id": mine[0]["model_id"],
            "model_revision": mine[0]["model_revision"],
            "frames": len(mine),
            "mean_error_rad": float(np.nanmean(errors)),
            "mean_first_step_error_rad": float(
                np.nanmean([r["first_step_error_rad"] for r in mine])
            ),
            "per_joint_error_rad": {
                j.name: float(np.nanmean([r["per_joint_error_rad"][j.name] for r in mine]))
                for j in joints
            },
            "uncertainty_error_rank_correlation": rank_correlation(
                [r["uncertainty"] for r in mine], errors
            ),
            "uncertainty_method": mine[0]["uncertainty_method"],
            "inference_s": latency_summary([r["inference_s"] for r in mine]),
            "round_trip_s": latency_summary([r["round_trip_s"] for r in mine]),
        }
    summary = {
        "what": "open-loop joint error against recorded actions; not task success",
        "trained_on_these_episodes": args.trained_on,
        "dataset": str(args.dataset).replace("\\", "/"),
        "episodes": args.episodes,
        "stride": args.stride,
        "horizon": args.horizon,
        "samples": args.samples,
        "seed": args.seed,
        "cameras": cameras,
        "units_spec": str(args.units_spec).replace("\\", "/"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "wall_s": round(time.monotonic() - started, 1),
        "models": models,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["models"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
