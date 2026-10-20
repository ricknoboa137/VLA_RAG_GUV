# Fine-tuning a VLA on the development machine

This folder sets up and measures fine-tuning of
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) (Apache-2.0, 450M
parameters) with [LeRobot](https://github.com/huggingface/lerobot) on one
8 GB laptop GPU, for the pick-and-place arm that will be mounted on the UGV.

It is deliberately **arm-independent**. The arm (candidate: Hiwonder ArmPi
Ultra, 5 joints + gripper) is not bought yet and has no sim model, so nothing
here assumes its joint count, cameras or kinematics. SmolVLA pads state and
action to 32 dimensions internally; any arm with at most 32 of each fits
without code changes, and the dimensions come from the dataset.

Training is **not part of the `carma` package**. torch and LeRobot live only
in a separate conda environment, `carma-train`, so the library's pinned
dependencies in `pyproject.toml` are untouched.

Measured results, and the dated log of every run, are in
[`results/RESULTS.md`](results/RESULTS.md).

---

## 1. Why a native Windows conda environment

Checked against the LeRobot installation guide and the PyPI metadata of
`lerobot==0.6.1` (released 2026-08-03), not assumed:

- LeRobot needs **Python >= 3.12** and **torch >= 2.7, < 2.12**.
- Windows is supported natively: PyPI's torch on Windows is a CUDA wheel, and
  the guide gives the PowerShell activation path. WSL is only mentioned for an
  extra `evdev` package, which robot teleoperation needs and training does not.
- Video decoding uses **torchcodec**, which on Windows needs torch >= 2.8,
  torchcodec >= 0.7, and an `ffmpeg` from conda-forge (the guide's required
  route for torch < 2.10; it works for all versions).

So WSL2 or Docker is not needed. Both smoke tests below ran natively.

Chosen versions, all exact:

| Component | Version | Source |
|---|---|---|
| Python | 3.12.12 | conda-forge |
| ffmpeg | 7.1.1 | conda-forge |
| torch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 | download.pytorch.org/whl/cu128 |
| torchcodec | 0.10.0 (the release matched to torch 2.10) | PyPI |
| lerobot | 0.6.1, extras `smolvla,training` | PyPI |
| transformers | 5.5.4 (pulled in by the `smolvla` extra) | PyPI |

CUDA 12.8 wheels need NVIDIA driver >= 570; this machine has 595.95.

## 2. Install, from a clean Windows machine

Needs: Anaconda or Miniforge, an NVIDIA driver >= 570, and a checkout of this
repository. Python is not assumed to be on `PATH`; call the environment's
interpreter by its full path.

```powershell
# 1. The conda part (Python, ffmpeg and their libraries), exact builds:
conda create -n carma-train --file training/conda-win64.lock

# 2. The pip part, every package pinned (torch comes from the PyTorch index
#    named inside the lock file):
& "$env:USERPROFILE\anaconda3\envs\carma-train\python.exe" -m pip install -r training/requirements-win64.lock

# 3. Check:
& "$env:USERPROFILE\anaconda3\envs\carma-train\python.exe" -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), lerobot.__version__)"
```

`requirements.in` holds the four top-level pins the lock was resolved from.
On another platform, resolve a new lock from it rather than editing this one.

## 3. Fetch the assets (the only step that uses the network)

```powershell
$py = "$env:USERPROFILE\anaconda3\envs\carma-train\python.exe"
& $py scripts/fetch_training_assets.py smolvla_base
& $py scripts/fetch_training_assets.py smolvlm2-500m-video-instruct
& $py scripts/fetch_training_assets.py svla_so101_pickplace
```

All three are Apache-2.0, ungated, and pinned to a commit in the script:

| Name | Repository | Size | What it is |
|---|---|---|---|
| `smolvla_base` | `lerobot/smolvla_base` | 865 MB | The pretrained policy |
| `smolvlm2-500m-video-instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 5 MB | Config and tokenizer only |
| `svla_so101_pickplace` | `lerobot/svla_so101_pickplace` | 83 MB | 50 episodes, 11 939 frames, SO-101 pick-and-place, 2 cameras, LeRobot dataset v3.0 |

**One deliberate rewrite.** Upstream `smolvla_base` names its VLM and tokenizer
by Hub id and sets `load_vlm_weights=true`, which downloads the full 1.9 GB VLM checkpoint
at load time only to overwrite it with the policy's own weights. The fetch
script points `vlm_model_name` and the tokenizer at
`assets/smolvlm2-500m-video-instruct` and sets `load_vlm_weights=false`. The
policy weights are untouched, and loading them with `strict=True` succeeds:
every parameter comes from `smolvla_base`. The relative path means commands
are run from the repository root.

After this, set these for every command below, so nothing can reach the
network:

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:WANDB_MODE = "disabled"
$env:PYTHONUTF8 = "1"   # LeRobot's progress output is UTF-8
```

## 4. Smoke test: inference

```powershell
& $py training/smoke_inference.py --dtype bfloat16 --out runs/training/inference_bf16.json
```

Loads the policy on the GPU, feeds random images from three cameras plus a
random state, and times `predict_action_chunk` (one 50-step action chunk, 10
flow-matching denoising steps). `--state-dim`/`--action-dim`/`--cameras` set
the shape, so the arm's real numbers can be tried before the arm exists.
`bfloat16` is the checkpoint's own layout, unchanged: VLM and action expert
in bf16, the small state/action/time projections and the LM head in fp32.
`float32` upcasts the VLM and expert, for comparison only.

Measured on an idle GPU (full table and ranges in
[`results/RESULTS.md`](results/RESULTS.md)):

| Precision | Cameras | Median latency per 50-action chunk | Peak VRAM |
|---|---|---|---|
| bf16 (native) | 3 | 537 ms | 0.91 GiB |
| fp32 (upcast) | 3 | 576 ms | 1.76 GiB |
| bf16 (native) | 2 | 487 ms | 0.90 GiB |

The checkpoint from the 300-step run below loads the same way
(`--policy runs/training/smoke300/checkpoints/000300/pretrained_model`) and
predicts finite actions.

Latency is bound by kernel launches, not arithmetic: the prefix (images and
language) costs about 48 ms and each of the 10 denoising steps about 70 ms of
small kernels, with the GPU near 42% busy. A chunk is 1.7 s of motion at
30 Hz, so half a second per chunk is workable with chunked execution.

## 5. Smoke test: fine-tuning

```powershell
& $py training/measure_train.py `
  --policy.path=assets/smolvla_base --policy.push_to_hub=false `
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/svla_so101_pickplace `
  '--rename_map={"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' `
  --batch_size=24 --steps=300 --log_freq=25 --save_checkpoint=true --save_freq=300 `
  --num_workers=2 --seed=0 --wandb.enable=false `
  --output_dir=runs/training/smoke300 --vram-limit-gib=7.4
```

`measure_train.py` is `lerobot-train` plus a peak-memory record; every
other argument is LeRobot's. `--rename_map` maps the dataset's two cameras
onto the policy's camera slots. The batch sweep is
`training/sweep_batch.py --variant <name> --batches ...`.

**Windows trap, worth knowing before reading any memory number.** When a
batch needs more than the card has, the Windows driver does not raise
out-of-memory: it silently spills into shared system memory and the run
continues several times slower. The first sweep hit exactly this at batch 32
(7.95 GB dedicated plus 1.9 GB shared). `--vram-limit-gib` caps PyTorch's
allocator below the card's size so an oversized batch fails fast and is
recorded as not fitting.

**Precision is already bf16.** `smolvla_base` stores its VLM and action
expert in bf16 (474 of its 500 tensors), and they load and train in bf16;
only the projections, LM head and some norms are fp32. That is upstream's
layout too, so there is no fp32 baseline to save memory from.

**Two memory levers asked about do not work for SmolVLA in LeRobot 0.6.1:**

- *Gradient checkpointing does not exist.* `--policy.gradient_checkpointing=true`
  is rejected (`unrecognized arguments`); only the diffusion, EO-1 and EVO-1
  policies have it.
- *`--policy.use_amp=true` crashes on the first step.* SmolVLA has no `dtype`
  field, so LeRobot falls back to torch's default CUDA autocast type, fp16,
  with a GradScaler, and the scaler cannot unscale bf16 gradients:
  `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.

The lever that does exist is how much of the model is trained:
`--policy.train_expert_only=true` (the default and upstream recipe, 100M
trainable) or `false` (the VLM's language layers too, 306M trainable; the
vision encoder stays frozen either way).

Largest batch that fits, idle GPU, 25-step runs (every row is in
[`results/RESULTS.md`](results/RESULTS.md)):

| Variant | Trainable | Largest batch | Peak VRAM (reserved) | Steps/s | Samples/s | Next batch |
|---|---|---|---|---|---|---|
| expert-only (default) | 99.9M | 24 | 6.49 (6.66) GiB | 0.50 | 12.1 | 28: OOM |
| full VLM language layers | 306.5M | 16 | 6.38 (6.54) GiB | 0.70 | 11.2 | 20: OOM |
| either, `use_amp=true` | — | none | — | — | — | crashes on step 1 |

Throughput is flat at about 12 samples/s from batch 8 upwards: the GPU is
saturated, and a bigger batch only means fewer, slower steps.

The 300-step run (expert-only, batch 24, seed 0): 639 s, 0.49 steps/s,
11.7 samples/s, 6.49 GiB peak, loss 0.571 at step 25 down to 0.207 at step
300. The checkpoint was written and reloads; the process then exited 1 on the
Windows symlink issue in section 7.

## 6. Is 8 GB enough?

**Yes, for memory.** The upstream SmolVLA fine-tuning recipe (expert only)
runs at batch 24 with 0.7 GiB to spare, and training the VLM's language layers
as well still fits at batch 16. LoRA or freezing more is not needed.

**Throughput is the limit.** At about 12 samples/s, 20 000 steps at batch 24
(480 000 samples) take about 11 hours; the upstream recipe's batch 64 x 20 000
steps would be about 30 hours. Overnight runs on this laptop are realistic;
anything larger wants a bigger GPU. LeRobot has no gradient accumulation, so
batch 64 itself cannot be reproduced here (see section 7).

## 7. What is still open

1. **Checkpoint symlink on Windows.** LeRobot links `checkpoints/last` after
   each save, which needs Windows Developer Mode (Settings > System > For
   developers) or an elevated shell. Without it the run exits 1 right after
   the first checkpoint is written. The checkpoint is complete; resuming needs
   its explicit path instead of `last`.
2. **Batch 24 instead of 64.** No gradient accumulation in LeRobot 0.6.1, so
   the upstream recipe's batch cannot be matched here. Whether the learning
   rate or step count should change for batch 24 is untested: the smoke run
   shows the loss falling, not that the policy is good.
3. **No evaluation yet.** Only the training loss was measured. Success rate
   needs the arm, or its sim model, and recorded episodes.
4. **Determinism.** Two seed-0 runs agreed to the third decimal of the loss,
   not bit for bit; `--cudnn_deterministic=true` was not tried. Required
   before any number goes in a paper (AGENTS.md section 2).
5. **bf16 weights without an fp32 master copy.** Upstream SmolVLA trains its
   bf16 parameters directly. Late in the cosine schedule (learning rate down to
   2.5e-6) small updates may round away. Effect not measured.
6. **Arm-specific work, deliberately not started.** A LeRobot robot class for
   the chosen arm's bus servos, a recorded dataset of its own episodes, and the
   mapping of its cameras (e.g. the depth camera's RGB stream) onto the policy's
   camera slots. State and action sizes up to 32 need no code change.
7. **Latency on the robot.** Measured here only; the on-robot computer is not
   chosen. `torch.compile` (`--policy.compile_model`) needs Triton, which this
   Windows environment does not have, so it was not tried.
8. **Other platforms.** The lock files are Windows x64. Linux or a Jetson
   needs its own lock resolved from `requirements.in`.
9. **X-VLA fallback** (0.9B, LoRA) is not set up or measured.
