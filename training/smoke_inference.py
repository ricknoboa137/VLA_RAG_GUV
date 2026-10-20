"""Inference smoke test: load SmolVLA on the GPU and time action-chunk prediction.

Arm-independent: the state and action dimensions are arguments, and the
normalisation statistics are identity, so the output is not a meaningful action
for any robot. The point is the latency and memory a real policy of this shape
costs. Runs offline from ``assets/`` (see ``training/README.md``).

    python training/smoke_inference.py --dtype bfloat16 --action-dim 6
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _features(cfg: SmolVLAConfig, cameras: int, state_dim: int, action_dim: int) -> None:
    """Replace the checkpoint's SO-100 feature shapes with the requested ones."""
    if max(state_dim, action_dim) > cfg.max_action_dim:
        raise SystemExit(f"SmolVLA pads to {cfg.max_action_dim} dims; got {state_dim}/{action_dim}")
    inputs: dict[str, PolicyFeature] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for i in range(cameras):
        inputs[f"observation.images.camera{i + 1}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        )
    cfg.input_features = inputs
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}


def main() -> int:
    """Run warm-up plus timed predictions and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("assets/smolvla_base"))
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON result file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this interpreter")
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cfg = SmolVLAConfig.from_pretrained(args.policy)
    _features(cfg, args.cameras, args.state_dim, args.action_dim)
    cfg.device = "cuda"

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.policy, config=cfg, strict=True)
    policy = policy.to(device=device).eval()
    # SmolVLA's own mixed-precision layout: the VLM and action expert in the
    # requested dtype, the small state/action/time projections kept in float32
    # (the model casts the expert output to float32 before action_out_proj).
    policy.model.vlm_with_expert.to(dtype=dtype)
    load_s = time.perf_counter() - t0

    # Identity statistics: the normaliser is exercised but changes nothing.
    stats = {
        "observation.state": {
            "mean": torch.zeros(args.state_dim),
            "std": torch.ones(args.state_dim),
        },
        "action": {"mean": torch.zeros(args.action_dim), "std": torch.ones(args.action_dim)},
    }
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    frame: dict[str, object] = {
        "observation.state": torch.from_numpy(
            rng.standard_normal(args.state_dim).astype(np.float32)
        ),
        "task": "pick up the block and place it in the container",
    }
    for i in range(args.cameras):
        image = rng.integers(0, 256, size=(3, 480, 640), dtype=np.uint8)
        frame[f"observation.images.camera{i + 1}"] = torch.from_numpy(image).float() / 255.0

    def predict() -> torch.Tensor:
        batch = preprocessor(dict(frame))
        noise = torch.from_numpy(
            rng.standard_normal((1, cfg.chunk_size, cfg.max_action_dim)).astype(np.float32)
        ).to(device=device)
        chunk = policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize()
        return chunk

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        predict()
    times_ms: list[float] = []
    for _ in range(args.runs):
        t = time.perf_counter()
        chunk = predict()
        times_ms.append((time.perf_counter() - t) * 1e3)
    actions = postprocessor(chunk.float())

    result = {
        "policy": str(args.policy.as_posix()),
        "dtype": args.dtype,
        "cameras": args.cameras,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "chunk_shape": list(actions.shape),
        "finite": bool(torch.isfinite(actions).all()),
        "params_m": round(sum(p.numel() for p in policy.parameters()) / 1e6, 1),
        "load_s": round(load_s, 1),
        "latency_ms_median": round(statistics.median(times_ms), 1),
        "latency_ms_min": round(min(times_ms), 1),
        "latency_ms_max": round(max(times_ms), 1),
        "denoising_steps": cfg.num_steps,
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
