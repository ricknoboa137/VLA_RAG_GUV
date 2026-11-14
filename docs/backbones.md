# Backbones

A backbone is a `NavigationBackbone`: given an observation, the instruction and
any retrieved context, it returns a velocity command and an uncertainty in
`[0, 1]`. It never imports `carma.memory`; context arrives as text.

| Key | Module | What it is |
|---|---|---|
| `heuristic` | `backbone/heuristic.py` | Model-free scaffolding for the harness. |
| `xvla` | `backbone/xvla.py` | X-VLA 0.9B behind the chunk-playback backbone. |

## Chunked policies

Flow-matching and diffusion policies denoise a whole sequence of actions from
random noise. `backbone/chunked.py` adapts any such model. The model implements
`ChunkSampler` — noise in, chunks out — and `ChunkedBackbone` supplies
everything the arbiter sees, so two models under one config differ only in the
sampler. Bringing your own chunked model means writing a sampler and a
registered subclass, as `xvla.py` does.

**Inputs.** The left image is the primary view; the right image of a stereo
pair is passed as a second view when `Observation.rgb_right` is present. The
state vector carries the previous step's commanded linear and angular velocity,
zero-padded. Depth is not passed.

**Prompt.** `"<instruction> Operator notes: <context 1>; <context 2>"`. The
instruction comes first so a short text window truncates context, never the
instruction.

**Action layout.** Channel 0 is forward velocity in metres per second, channel
1 is yaw rate in radians per second, positive counter-clockwise, and channel 2,
if present, is a stop indicator in `[0, 1]`. Velocities are clipped to
`[0, max_linear]` and `[-max_angular, max_angular]`. Chunk step `i` is control
step `i`, so the model must be trained at the control period it is run at.

**Playback.** Each plan samples `num_samples` chunks and executes the first; it
does not average them, because the mean of "left" and "right" is "straight". A
chunk plays for `replan_every` control steps. A control step is identified by
`Observation.t`, so the loop's second call in a step re-proposes rather than
advancing. Non-empty context that differs from what the current chunk was
conditioned on forces a new plan, and that chunk then keeps steering until it
expires: one RETRIEVE or ASK shapes `replan_every` steps, not one.

**Uncertainty.** Over the `replan_every` steps that will be executed: the
standard deviation across samples of each capped velocity, divided by its cap,
the larger of the two channels, averaged over the steps, divided by
`dispersion_full` and clipped to `[0, 1]`. Two samples at opposite yaw caps
score 1.0 with `dispersion_full: 1.0`. It is constant while a chunk plays.

`dispersion_full` is **not calibrated**. The default maps only total
disagreement to 1.0, so a trained policy will read low and the arbiter will
rarely ask. Fit it on held-out episodes so that uncertainty tracks the
probability of needing a recovery, which is what `docs/cost-model.md` assumes,
before any number from this backbone is reported.

**Determinism.** All noise is drawn from the caller's generator; a sampler
draws none of its own. On CPU the result is a function of `(config, seed)`. On
GPU, floating-point results can differ between devices and driver versions, so
runs pooled in one report must come from one machine, or be run on CPU.

## X-VLA

Zheng et al., *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment
Vision-Language-Action Model*, arXiv 2510.10274. A Florence-2 encoder for
images and language, and a transformer that denoises an action chunk by flow
matching, conditioned on a learned soft prompt per embodiment selected by
`domain_id`.

### Licence audit

Checked 2026-10-02.

| Component | Source | Licence |
|---|---|---|
| X-VLA code | `github.com/2toinf/X-VLA` | Apache-2.0 |
| Weights, 879,738,545 parameters | `2toINF/X-VLA-Pt`, not gated | Apache-2.0 |
| Vision-language encoder | `microsoft/Florence-2-large` | MIT |
| `torch`, `torchvision`, `uvicorn` | PyPI | BSD-3-Clause |
| `transformers`, `safetensors`, `timm` | PyPI | Apache-2.0 |
| `einops`, `fastapi`, `json-numpy` | PyPI | MIT |
| `pillow` | PyPI | MIT-CMU |

The checkpoint ships its own model code, which `transformers` executes on load
(`trust_remote_code`). `scripts/fetch_models.py` pins the checkpoint to commit
`c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`; review the diff before moving it.
That code imports `fastapi`, `uvicorn` and `json-numpy` at module scope for an
HTTP server this project never starts, which is the only reason they are
dependencies. Loading is from disk with `local_files_only`.

### Install and fetch

```bash
pip install -e ".[vla]"
python scripts/fetch_models.py xvla-base
```

The `torch` wheel PyPI serves for Windows is CPU-only. For a GPU, install the
same version from the PyTorch index first, for example
`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`.

### The released weights do not drive

`xvla-base` was pretrained on manipulator data: its outputs are end-effector
poses and gripper commands. Loaded here it exercises the pipeline and measures
latency, and the motion it produces is meaningless. A usable checkpoint needs
fine-tuning on this robot's own driving data, with:

- **A new `domain_id`.** The published checkpoints use 0-10 and 18; `20` is
  the one reserved for this robot in `configs/backbone/xvla.yaml`.
- **The action layout above**, three channels, through X-VLA's `auto` action
  mode with `real_action_dim: 3`. That mode trains every channel with a
  squared-error loss, so the stop channel is a regressed 0/1 target and is
  thresholded directly, not passed through a sigmoid.
- **Left image as view 0, right as view 1**, and the previous command as the
  state vector, matching what the adapter feeds at run time.
- **Context in the prompt.** The pretrained model has never seen
  `Operator notes:`. For RETRIEVE and ASK to change behaviour, training
  episodes must include correction text in that format, paired with the
  driving that followed it. Without this the `carma` and `no_memory`
  conditions would differ in cost and not in behaviour.

The processor truncates the prompt to 50 tokens, the length the model was
pretrained with. Long instructions plus several retrieved entries will lose the
tail of the context.

### Verified and not yet verified

Verified: the licence and gating above; that the denoising loop in
`XvlaSampler.sample` matches the checkpoint's own `generate_actions` line for
line, apart from taking noise as an argument and encoding the images once;
that the pinned `vla` extra resolves alongside the core pins; and, with a
model-free sampler, everything in `tests/test_backbone_chunked.py`.

Not yet verified: `XvlaSampler` has not been run against the real weights,
because the checkpoint and `torch` are not installed in the development
environment. Its first load, its latency on the target hardware and its memory
footprint are open.
