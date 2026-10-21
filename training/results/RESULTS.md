# Training results

The record of every fine-tuning measurement, for reports. Newest first. Each
entry names its folder here, which holds the raw JSON, loss curves (CSV) and
`env.json` (commit, package versions, GPU, driver, asset revisions). Add
entries with `training/record_results.py`; never edit an old entry's numbers.
Correct a mistake in a new entry that says which one it supersedes.

Conventions: memory is `torch.cuda.max_memory_allocated()` (reserved in
brackets where measured), GiB. Throughput is the median over LeRobot's
per-step timers after the first logged window, so model loading and the
dataloader warm-up are excluded. "Fits" means it ran under a 7.4 GiB
allocator cap (see `training/README.md`, the Windows spill trap).

---

## 2026-10-20: SmolVLA and X-VLA serving the SO-101 arm (plumbing smoke)

Folder: [`2026-10-20_arm-policies-smoke/`](2026-10-20_arm-policies-smoke/)

**Not a benchmark result.** Both checkpoints are community fine-tunes trained
on every episode scored here, so the open-loop numbers measure how well each
reproduces its own training data, through this project's serving path. They
show the comparison runs end to end; the fair benchmark fine-tunes both bases
on a held-out split (`docs/arm-policies.md`, "What is still open").

**Setup.** `whosricky/svla-so101-pick-red-cube-2cam` @`5302eec` (SmolVLA, bf16)
and `whosricky/xvla-so101-pick-red-cube-2cam` @`7a007f6` (X-VLA, bf16), both
fine-tuned from their model's base on `whosricky/so101_pick_mixed_cubes_2cams`
@`f77f316`. Specs `configs/arm_policy/*.yaml`. Each served by
`scripts/policy_server.py`, both at once, queried over HTTP by
`scripts/benchmark_arm_policies.py`. LeRobot 0.6.1, torch 2.10.0+cu128, same
GPU and driver as below. Code: commit `df60cc4`, clean tree.

**Conditions.** The GPU was shared with another CUDA process and SOLIDWORKS
throughout, so latencies are not idle-GPU numbers. The first requests to a
cold server took 15–99 s while its allocations settled near the 8 GB limit
(the Windows spill trap); everything below is after warm-up.

### Open loop: episodes 0, 45, 90, 135, 181, every 60th frame, 25 steps, 4 samples, seed 0

31 frames, identical requests to both models.

| Model | Mean joint error | First step | Uncertainty–error rank corr. | Inference p50 / p95 |
|---|---|---|---|---|
| SmolVLA | 0.049 rad | 0.023 rad | 0.53 | 0.64 / 0.81 s |
| X-VLA | 0.082 rad | 0.049 rad | 0.35 | 0.52 / 0.53 s |

Per joint (rad), SmolVLA / X-VLA: shoulder pan 0.034 / 0.048, shoulder lift
0.084 / 0.168, elbow 0.079 / 0.153, wrist flex 0.027 / 0.050, wrist roll
0.026 / 0.021, gripper 0.042 / 0.050. Uncertainty is
`chunk_spread/uncalibrated` for both; its values are not comparable between
the models (X-VLA's samples spread almost 30 times less).

### Closed loop in Gazebo: the arm moved by a spoken instruction

`gazebo_runs.json`, with the joint traces (`gazebo_*_joints.csv`, ~10 Hz) and every action record (`gazebo_*_actions.jsonl`); `gazebo_runs.json` says which run each file is from. `row_crop.launch.py arm:=true`, `carma_vla` with
`arm:=true`, the clip "Pick up the red cube in front of the robot." synthesised
offline and sent through the speech node. With each model the arm moved
(shoulder lift range SmolVLA 1.59 rad, X-VLA 1.08 rad), "stop" held every joint
within 0.005 rad, and "go" resumed (checked with SmolVLA). Round trip from the
container: SmolVLA about 1.1 s, X-VLA about 0.55 s per 4-sample chunk. The
world has no red cube and the arm is primitive geometry, so this shows the
path works, not that either model can do the task.

Found on the way: chunks timed from the observation all expired before they
arrived (1.8 s round trip against a 0.83 s chunk), so the loop was made
synchronous; and a socket timeout escaped `HttpTransport` and froze the loop,
fixed in `2484fb6`.

---

## 2026-10-19: SmolVLA environment smoke tests

Folder: [`2026-10-19_smolvla-smoke/`](2026-10-19_smolvla-smoke/)

**Setup.** `lerobot/smolvla_base` (450.0M parameters) fine-tuned on
`lerobot/svla_so101_pickplace` (50 episodes, 11 939 frames, 2 cameras, 6-dim
state and action), seed 0. LeRobot 0.6.1, torch 2.10.0+cu128, Windows 11,
NVIDIA RTX 2000 Ada Laptop GPU (8 GB, 35 W), driver 595.95. Code: commit
`ff31a10` (`env.json` says dirty because these results and the README were
not committed yet; the measuring scripts were unchanged except for the
failure handling in `measure_train.py`, added after the 300-step run).

**Conditions.** GPU idle for every number below, except where marked. Two
earlier attempts were discarded, not averaged in: one shared the GPU with
another CUDA process (2.4x slower at batch 8), and one spilled into shared
system memory at batch 32 before the allocator cap existed.

### Inference: one 50-action chunk, 10 denoising steps

| Policy | Precision | Cameras | Runs | Median latency | Min–max | Peak VRAM |
|---|---|---|---|---|---|---|
| smolvla_base | bf16 (native) | 3 | 30 | 537 ms | 429–597 ms | 0.91 GiB |
| smolvla_base | fp32 (upcast) | 3 | 30 | 576 ms | 478–628 ms | 1.76 GiB |
| smolvla_base | bf16 (native) | 2 | 30 | 487 ms | 370–568 ms | 0.90 GiB |
| fine-tuned, step 300 | bf16 (native) | 3 | 10 | 542 ms* | 413–726 ms | 0.91 GiB |

\*Measured while another CUDA process held 0.9 GB of the GPU.

The latency is launch-bound, not compute-bound: the image/language prefix
costs about 48 ms and each of the 10 denoising steps about 70 ms of small
kernels, with GPU utilisation near 42%.

### Training: batch-size sweep (25 steps each)

Trainable parameters: expert-only 99.9M, full 306.5M (vision encoder frozen
in both).

| Variant | Batch | Peak VRAM (reserved) | Steps/s | Samples/s | Result |
|---|---|---|---|---|---|
| expert | 8 | 3.22 (3.28) | 1.47 | 11.8 | fits |
| expert | 16 | 4.87 (4.92) | 0.76 | 12.2 | fits |
| expert | 20 | 5.68 (5.81) | 0.61 | 12.2 | fits |
| expert | **24** | 6.49 (6.66) | 0.50 | 12.1 | **fits, largest** |
| expert | 28 | — | — | — | OOM at the cap |
| full | 1 | 3.59 (3.85) | 2.96 | 3.0 | fits |
| full | 2 | 3.61 (3.82) | 2.48 | 5.0 | fits |
| full | 4 | 3.65 (3.86) | 2.08 | 8.3 | fits |
| full | 8 | 4.41 (4.70) | 1.30 | 10.4 | fits |
| full | 12 | 5.39 (5.68) | 0.92 | 11.0 | fits† |
| full | **16** | 6.38 (6.54) | 0.70 | 11.2 | **fits, largest** |
| full | 20 | — | — | — | OOM at the cap |
| expert, `use_amp` | 8 | — | — | — | crashes on step 1 |
| full, `use_amp` | 1 | — | — | — | crashes on step 1 |

†Ran as another application was starting on the GPU; consistent with its
neighbours, kept and marked.

`use_amp` crash: LeRobot autocasts SmolVLA to fp16 with a GradScaler, which
cannot unscale its bf16 gradients
(`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
'BFloat16'`). Gradient checkpointing: not available for SmolVLA in 0.6.1
(`unrecognized arguments`).

### Training: 300 steps, expert-only, batch 24

| Steps | Wall (training loop) | Steps/s | Samples/s | Peak VRAM | Loss, step 25 → 300 |
|---|---|---|---|---|---|
| 300 (7 200 samples, 0.60 epoch) | 639 s | 0.49 | 11.7 | 6.49 GiB | 0.571 → 0.207 |

Loss curve: `smoke300.curve.csv`. The checkpoint was written
(`pretrained_model` 1.2 GB, `training_state` 0.4 GB) and reloads strictly and
predicts; the process then exited 1 on LeRobot's `checkpoints/last` symlink
(WinError 1314, Windows Developer Mode off).

**Run-to-run.** The same full/batch-16 configuration, seed 0, run twice
gave losses 1.501 / 1.502 at step 5 and 0.603 / 0.602 at step 25: close, not
bit-identical (`cudnn_deterministic` was off).

**Verdict.** 8 GB is enough for real SmolVLA fine-tuning, both the upstream
expert-only recipe (batch 24) and with the VLM language layers trained
(batch 16). Throughput, not memory, is the limit: about 12 samples/s, so
20 000 steps at batch 24 take about 11 hours.
