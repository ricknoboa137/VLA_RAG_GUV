# Integrating a VLA backbone

The policy runs in its own process behind an HTTP endpoint, and
`carma.backbone.remote.RemoteBackbone` makes it satisfy `NavigationBackbone`.
A condition still differs from another by one line of config.

```bash
python scripts/policy_server.py --policy echo --self-check   # terminal 1
carma run --config configs/experiment/remote_smoke.yaml      # terminal 2
```

## Why a separate process

A 0.9B-parameter policy wants a GPU the robot may not carry, and the model can
then be restarted, swapped or profiled without touching the episode loop. The
cost is that the robot stops thinking when the link drops, which the arbiter
must treat as the same condition as low confidence: stop and ask, never guess.

## The contract

`src/carma/backbone/wire.py` holds both ends, so client and server cannot
drift. Images cross as base64 PNG, which is lossless, so the server sees the
pixels the robot saw. Depth crosses as 16-bit millimetres, the RealSense and
ROS `16UC1` convention, where 0 means no reading.

| Request | | Response | |
|---|---|---|---|
| `image` | base64 PNG, BGR | `linear` | m/s |
| `image_right` | optional stereo | `angular` | rad/s, CCW positive |
| `depth_mm` | optional 16-bit PNG | `stop` | goal reached |
| `instruction` | the instruction in force | `uncertainty` | in [0, 1] |
| `context` | retrieved corrections | `uncertainty_method` | how it was obtained |
| `pose`, `t`, `stage` | where and when | `model_id`, `model_revision` | what answered |
| `seed` | the only randomness | `inference_s` | measured |
| `samples` | chunks to draw | `info` | free-form, never read by the arbiter |

## Determinism across a process boundary

Section 2.2 says a run is `(config, seed, commit)`. A server that drew its own
noise would break that silently: the numbers would simply differ next time.

- The **seed travels with every request** and the server samples from it alone.
- `--self-check` asserts the policy answers a repeated request identically,
  before it serves anything.
- Every reply **names the weights**, and `RemoteBackbone.last_model_id`
  exposes them for the run manifest. A result whose weights are unstated is
  not a result.

Verified: two full runs of `remote_smoke` at seed 0, through a real socket,
produced byte-identical decision logs.

## Why no test opens a socket

Section 2.3 forbids network access at inference and test time. The rule exists
to stop results depending on a service nobody can reproduce — a proprietary
API, a gated checkpoint, an account. A server the experimenter starts from
openly licensed weights on their own machine is a different thing, and the
honest response is to make the dependency explicit rather than pretend it is
absent. So the transport is injected, `tests/test_remote_backbone.py` uses an
in-process fake, and `configs/experiment/remote_smoke.yaml` is excluded from
CI.

## X-VLA

[X-VLA](https://github.com/2toinf/X-VLA) (Apache 2.0, 0.9B parameters, ICLR
2026) is soft-prompted per embodiment and already ships a server with the same
shape: `model.run(processor, host, port=8000)` serving `POST /act`.

**It cannot drive this robot yet, and the reason is the action space.** Every
released checkpoint — WidowX, LIBERO, CALVIN, RoboTwin2, VLABench, SoftFold,
Google-Robot — is a *manipulator* policy. The output is a chunk of 30 poses in
EE6D: 3 position, 6 rotation, 1 gripper, 10 padding. This project commands a
wheeled robot's linear and angular velocity. There is no honest mapping
between an end-effector pose and a wheel velocity, so `XvlaPolicy` raises
rather than inventing one.

What it needs, in order:

1. **A navigation action space the checkpoint predicts.** That means
   fine-tuning from `2toINF/X-VLA-Pt` on navigation data with a soft prompt
   for this embodiment. This is the real work, and it needs a dataset.
2. **An uncertainty estimator.** Flow matching yields no logits. The
   defensible route is to draw `samples` chunks from different noise and use
   their disagreement, then calibrate it and say so in `uncertainty_method`,
   as section 8 requires. An uncalibrated confidence would corrupt every
   arbitration result, because the cost model reads it.
3. **A recorded revision**, so the manifest names the weights.

Hardware is not the obstacle: 0.9B in bf16 fits the development machine's
8 GB of VRAM comfortably.
