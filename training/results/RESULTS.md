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

## 2026-10-22: SmolVLA against X-VLA on held-out SO-101 episodes (fair benchmark)

Folder: [`2026-10-22_arm-benchmark/`](2026-10-22_arm-benchmark/)

**What this is.** Both bases fine-tuned here with the same recipe on the same
training episodes, then scored open loop on episodes neither model saw. Open
loop measures imitation of the recording, **not task success**: no cube was
picked up, by a real arm or a simulated one, in this measurement.

**Split** (`split.json`, the same file as `training/splits/so101_red_cube_seed0.json`):
`whosricky/so101_pick_mixed_cubes_2cams` @`f77f316`, 182 episodes,
`np.random.default_rng(0).permutation(182)`: 146 train, 18 calibration (used
only for the uncertainty fit), 18 test (used only for scoring). Whole
episodes are held out. Normalisation statistics for state and action come
from the 146 training episodes only (`measure_train.py --split`); the
dataset's own `stats.json` covers all 182. The dataset's metadata carries 35
stale duplicate rows for episodes 132-166, which are skipped.

**Training** (`*_train_config.json`, `*_train.measure.json`, `*_train.curve.csv`):

| | SmolVLA | X-VLA |
|---|---|---|
| Base | `lerobot/smolvla_base` @`5e8d12a` | `lerobot/xvla-base` @`cdb7964`, `action_mode=auto` |
| Trained | action expert (VLM frozen), 99.9M | policy transformer + soft prompts (Florence-2 encoders frozen), 311.0M |
| Steps, batch, seed | 20 000, 8, 0 | 20 000, 8, 0 |
| Cameras | top -> camera1, gripper -> camera2 | top -> image, gripper -> image2, 1 empty slot |
| Optimiser, schedule | the policy's LeRobot preset: lr 1e-4, cosine with 1 000 warm-up steps, auto-scaled to 20 000 | same |
| Peak VRAM | 3.22 (3.27) GiB | 5.70 (5.76) GiB |
| Final loss (step 20 000) | 0.080 | 0.017 (the two losses are different quantities; not comparable) |
| Wall time | 16.4 h | 21.4 h |

Why frozen: a full X-VLA fine-tune (879M trainable, bf16) runs out of memory
at batch 8 under a 6.8 GiB cap, and frozen-encoder X-VLA runs out at batch 16.
So both run at batch 8 (also the community pair's recipe), and each trains its
action model with its VLM frozen. Wall times are **contended**: the GPU was
shared with another CUDA process throughout, and both runs were about 1.5-2x
slower than their uncontended 25-step probes (0.69 and 0.58 s/step).

**Uncertainty** (`uncertainty_fit.json`, `calibration_*`): the logistic map
`u = sigmoid(a ln(spread) + b)`, fitted per model by
`scripts/fit_arm_uncertainty.py` on the 18 calibration episodes (stride 15,
413 frames each). It predicts the event "chunk mean joint error > 0.0678 rad";
that tolerance is the pooled median calibration error, the same for both
models. SmolVLA a = 2.967, b = 14.338; X-VLA a = 0.807, b = 5.874.

**Result, 18 test episodes** (`test_summary.json`, `test_predictions.jsonl`;
stride 15, 426 frames per model, 4 samples, horizon 25, served in bf16):

| | SmolVLA | X-VLA |
|---|---|---|
| Mean joint error over the chunk | **0.0805 rad** | 0.0941 rad |
| First-step error | **0.0327 rad** | 0.0565 rad |
| Shoulder lift / elbow flex | 0.157 / 0.140 | 0.189 / 0.165 |
| Shoulder pan / wrist flex / wrist roll / gripper | 0.058 / 0.033 / 0.037 / 0.058 | 0.057 / 0.046 / 0.039 / 0.070 |
| Uncertainty-error rank correlation | 0.553 | 0.439 |
| Share above tolerance (test) / mean predicted u | 0.493 / 0.466 | 0.568 / 0.530 |
| Brier score / constant-rate baseline | 0.199 / 0.250 | 0.224 / 0.245 |
| Inference p50 / p95 (contended) | 0.92 / 1.36 s | 0.63 / 1.01 s |

Per-episode paired difference (SmolVLA minus X-VLA, mean of the episode
means): -0.0129 rad, 95% bootstrap interval over episodes [-0.0201, -0.0057]
(10 000 resamples of the 18 episodes with replacement,
`np.random.default_rng(0)`, from `test_predictions.jsonl`). X-VLA was better
on 5 of the 18 episodes.

**What it shows.** Under one recipe that fits an 8 GB card, fine-tuned
SmolVLA imitates held-out demonstrations more closely than fine-tuned X-VLA:
on every joint except shoulder pan (0.058 against 0.057, effectively level),
with a larger gap on the first step. Both uncertainty maps carry
information. On test data they were not fitted on, both score below the
constant-rate Brier baseline and their mean predictions sit close to the
observed rates. SmolVLA's map is a little better.

**What it does not show.** Task success, or closed-loop or real-arm
behaviour. A general ranking of the two architectures either: X-VLA trained
with its VLM frozen and at batch 8 because of the card, on one seed, one
dataset and one task. Absolute latency: the GPU was shared.

**Commands** (repository root, `carma-train`, `PYTHONPATH=src`, offline
environment variables from `training/README.md`):

```
python training/split_episodes.py --dataset assets/so101_red_cube --seed 0 --out training/splits/so101_red_cube_seed0.json
COMMON="--dataset.repo_id=local/so101_red_cube --dataset.root=assets/so101_red_cube --batch_size=8 --steps=20000 --log_freq=100 --save_checkpoint=true --save_freq=5000 --num_workers=2 --seed=0 --wandb.enable=false --vram-limit-gib=6.8 --split=training/splits/so101_red_cube_seed0.json"
python training/measure_train.py --policy.path=assets/smolvla_base --policy.push_to_hub=false --policy.train_expert_only=true '--rename_map={"observation.images.top": "observation.images.camera1", "observation.images.gripper": "observation.images.camera2"}' $COMMON --output_dir=runs/training/arm_benchmark/smolvla
python training/measure_train.py --policy.path=assets/xvla_base --policy.push_to_hub=false --policy.action_mode=auto --policy.dtype=bfloat16 --policy.empty_cameras=1 --policy.freeze_vision_encoder=true --policy.freeze_language_encoder=true '--rename_map={"observation.images.top": "observation.images.image", "observation.images.gripper": "observation.images.image2"}' $COMMON --output_dir=runs/training/arm_benchmark/xvla
# copy checkpoints/020000/pretrained_model to assets/smolvla_so101_red_cube_seed0 and assets/xvla_so101_red_cube_seed0
python scripts/policy_server.py --policy smolvla --arm-spec configs/arm_policy/smolvla_so101_red_cube_seed0.yaml --self-check --port 8010
python scripts/policy_server.py --policy xvla --arm-spec configs/arm_policy/xvla_so101_red_cube_seed0.yaml --self-check --port 8011
python scripts/benchmark_arm_policies.py --dataset assets/so101_red_cube --episodes <calibration episodes> --stride 15 --server smolvla=http://127.0.0.1:8010/act --server xvla=http://127.0.0.1:8011/act --units-spec configs/arm_policy/smolvla_so101_red_cube_seed0.yaml --out runs/bench/calibration
python scripts/fit_arm_uncertainty.py --predictions runs/bench/calibration/predictions.jsonl --out runs/bench/uncertainty_fit.json
# write a, b and tolerance_rad into both specs, restart both servers, then:
python scripts/benchmark_arm_policies.py --dataset assets/so101_red_cube --episodes <test episodes> --stride 15 --timeout-s 300 --server smolvla=http://127.0.0.1:8010/act --server xvla=http://127.0.0.1:8011/act --units-spec configs/arm_policy/smolvla_so101_red_cube_seed0.yaml --out runs/bench/test
python training/record_results.py --label arm-benchmark --date 2026-10-22 --json <the JSON files> --log <both training logs>
```

**Notes.**
- The calibration pass ran with the uncalibrated map, since the fit uses only
  the spread. The test pass ran with the fitted maps, so the uncertainties in
  `test_predictions.jsonl` are the ones the servers sent.
- The first test attempt stopped on a 60 s request timeout (a stall on the
  shared GPU) and was rerun in full with `--timeout-s 300`. The seeds are
  fixed, so the rerun sent the same requests.
- `dirty: true` in the summaries comes from this entry's specs and docs, not
  yet committed when they ran. The code that trained and scored is at
  `d549069` plus the commit that adds this entry.
- The plumbing baseline with the community checkpoints is the 2026-10-20 entry
  below. Those checkpoints were trained on these episodes, so their numbers
  are not comparable.

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

`gazebo_runs.json`. `row_crop.launch.py arm:=true`, `carma_vla` with
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
