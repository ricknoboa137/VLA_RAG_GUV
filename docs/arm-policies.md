# Arm policies: SmolVLA and X-VLA on the SO-101

Two vision-language-action models drive the SO-101 arm from its stereo camera
plus the operator's spoken instruction, both served by
`scripts/policy_server.py` over the act contract, so the robot side never
knows which one answered. They are compared as a benchmark, which is why
everything that could differ between them except the model is pinned to be
the same.

The arm is outside the arbitration and the five experimental conditions
(`docs/arm.md`). Nothing here enters an experiment config, a config hash or
the decision log.

## The models

| | SmolVLA | X-VLA |
|---|---|---|
| Architecture | SmolVLM2 + flow-matching action expert, 450M | Florence-2 + soft-prompted flow-matching transformer, 0.9B |
| Base checkpoint | `lerobot/smolvla_base` @`5e8d12a`, Apache-2.0 | `lerobot/xvla-base` @`cdb7964`, Apache-2.0 |
| Base action space | 6 joints, SO-100/101 community data | 20-dim end-effector (EE6D): **not** the SO-101's joints |
| Checkpoint served now | `whosricky/svla-so101-pick-red-cube-2cam` @`5302eec` | `whosricky/xvla-so101-pick-red-cube-2cam` @`7a007f6` |
| Chunk | 50 steps at 30 Hz; 25 executed | 30 steps at 30 Hz; 30 executed |
| Stored precision | fp32, served in bf16 | bf16 |

Every licence, revision and gating flag above was read from the model's own
Hub page (API, 2026-10-20). All are Apache-2.0 and ungated, and so is the
dataset both served checkpoints were trained on,
`whosricky/so101_pick_mixed_cubes_2cams` @`f77f316` (182 episodes, top and
gripper cameras, "Pick the red cube"). The BART tokenizer X-VLA uses
(`facebook/bart-large` @`cb48c13`) is Apache-2.0 too.

**Why these two checkpoints.** They are the only openly licensed pair found
that was fine-tuned by the same person, on the same SO-101 dataset, from each
model's own base, with the same cameras and the same joint units. That makes
them a like-for-like *plumbing* comparison now. They are community checkpoints
with no published evaluation, and both were trained on every episode of their
dataset, so **no number measured with them is a benchmark result**. The fair
benchmark fine-tunes both bases on one dataset split here; see "What is still
open".

`smolvla_base` itself is not served: its pretraining mixes community datasets
recorded in the old SO-100 calibration (shoulder-lift mean 125°, outside the
SO-101's ±100° range), with no single unit map to this arm's radians.
`xvla-base` predicts end-effector poses, not joints, so it needs fine-tuning
(or an IK layer) before it can drive the SO-101 at all.

## Running one

They run in the `carma-train` environment (`training/README.md`), never in the
`carma` library's, and offline: the spec names a local checkpoint and the
adapter switches the Hub off before LeRobot is imported.

```powershell
$py = "$env:USERPROFILE\anaconda3\envs\carma-train\python.exe"
& $py scripts/fetch_training_assets.py svla_so101_red_cube     # 1.2 GB, once
& $py scripts/fetch_training_assets.py xvla_so101_red_cube     # 1.8 GB, once
& $py scripts/fetch_training_assets.py smolvlm2-500m-video-instruct
& $py scripts/fetch_training_assets.py bart-large-tokenizer

$env:PYTHONPATH = "src"
& $py scripts/policy_server.py --policy smolvla --self-check --host 0.0.0.0 --port 8000
& $py scripts/policy_server.py --policy xvla    --self-check --host 0.0.0.0 --port 8001
```

`--self-check` sends one request twice and refuses to serve unless both arm
chunks are identical. Both models pass on the development GPU.

## The spec: what defines a policy server

`configs/arm_policy/<name>.yaml`, one per model, read by `scripts/arm_policy.py`:

- **checkpoint, model_id, model_revision** — the weights, and what every reply
  names in `model_id` / `model_revision`.
- **dtype** — the precision it runs in.
- **cameras** — which robot camera fills which of the model's camera slots.
  The arm stereo camera (the request's primary `image`) plays the dataset's
  gripper camera; the world camera's left frame (`world_left`) stands in for
  its fixed top camera. A camera the request lacks is left out, and both
  models mask a missing slot rather than see a black frame.
- **joints** — a two-point linear map per joint between the model's units and
  the URDF's radians. The dataset's values saturate at ±100 and 0–100,
  LeRobot's `range_m100_100` body joints and `range_0_100` gripper, and are
  mapped onto the joint limits of `so101_new_calib.urdf`, which the Gazebo arm
  shares. That rests on the calibrated range spanning the URDF limits with
  zero at the middle, as LeRobot's new calibration intends: approximate, the
  same for both models, and to be checked joint by joint on a calibrated real
  arm.
- **uncertainty** — how sample spread becomes an uncertainty (below).

`tests/test_arm_policy.py` holds the two specs to identical joint maps, frame
rate, uncertainty map and camera sources, and the joint maps to the Gazebo
arm's limits, so a comparison cannot quietly differ by more than the model.

## Determinism and uncertainty

The request's seed seeds torch before each call, and all `samples` chunks
(4 by default) are drawn in one batch. X-VLA's LeRobot port ignores an
explicit noise tensor and draws from torch's global generator, so seeding that
generator is the one mechanism that works for both.

The spread is the standard deviation across samples at each step and joint,
divided by that joint's range, averaged. The reply sends the first sample's
chunk and the spread's uncertainty.

**The uncertainty is not calibrated, and its method key says so:**
`chunk_spread/uncalibrated`, `u = s / (s + 0.02)`. It orders observations, but
it is not a probability, and it is not comparable between the two models.
X-VLA's samples spread almost 30 times less than SmolVLA's for the same request
(0.0006 against 0.017 of the joint range on the self-check frame), so the same
map gives X-VLA a far lower number without it being any more right.
Calibrating needs held-out episodes, which these checkpoints do not have. The
pair fine-tuned here does, and its specs use a fitted logistic map (below).

## On the robot

`ros2_ws/src/carma_vla` sends the arm's joint angles from
`/carma/arm/joint_states` with every request. With `arm:=true` it plays each
returned chunk out on `/carma/arm/<joint>/cmd_pos` at 30 Hz:

- synchronously, as LeRobot drives the real arm: it observes only while the
  arm is still, and plays the whole chunk from the moment the answer arrives.
  Timing rows from the observation instead was tried first and moved nothing:
  with a 1.8 s round trip, a 0.83 s chunk had expired before it arrived;
- clamped to the joint limits, and to 1.5 rad/s per joint (the servos manage
  4.7), so a chunk that disagrees with the arm's pose cannot slam it across
  its range;
- when a chunk runs out, the arm holds its last target until the next answer;
- a spoken "stop", a policy `stop`, or a server that does not answer drops the
  chunk and commands each joint to hold where it is measured to be.

No request is sent with `arm:=true` until fresh joint states arrive.

## The benchmark harness

`scripts/benchmark_arm_policies.py` feeds recorded SO-101 episodes to every
server, as identical `ActRequest`s over the same wire, and scores each chunk
against the actions the demonstrator recorded over the same steps. The metrics
are in `carma.metrics.arm`: mean absolute joint error over the chunk, per joint
and at the first step; the rank correlation between uncertainty and error;
inference and round-trip latency. The summary refuses to compare models scored
on different frames, and records the commit, the dataset and whether the
models were trained on the episodes.

Open-loop error measures imitation of the recording, not task success.

## The held-out benchmark (2026-10-22)

Both bases were fine-tuned here on the same 146 of the dataset's 182 episodes
(seed-0 split, `training/splits/so101_red_cube_seed0.json`), with 20 000
steps, batch 8 and seed 0 each, the same two cameras, and normalisation
statistics taken from the training episodes only. Each trains its action
model with its VLM frozen: SmolVLA its action expert (100M), X-VLA its policy
transformer and soft prompts (311M). A full X-VLA fine-tune does not fit in
8 GB, and frozen X-VLA does not fit at batch 16. Specs:
`configs/arm_policy/{smolvla,xvla}_so101_red_cube_seed0.yaml`; both pass
`--self-check`.

On 18 test episodes neither model saw, scored open loop:

| | SmolVLA | X-VLA |
|---|---|---|
| Mean joint error over the chunk | **0.081 rad** | 0.094 rad |
| First-step error | **0.033 rad** | 0.057 rad |
| Uncertainty-error rank correlation | 0.55 | 0.44 |
| Brier (constant baseline 0.25) | 0.199 | 0.224 |

SmolVLA's lead is 0.013 rad per episode, 95% bootstrap interval
[0.006, 0.020]; X-VLA was closer on 5 of the 18 episodes. Full numbers,
commands and caveats: `training/results/RESULTS.md`, 2026-10-22.

**What the numbers do and do not show.** They show which fine-tune
reproduces held-out demonstrations more closely, under one recipe that fits
an 8 GB card. They do not show task success: open-loop error scores each
chunk from the demonstrator's own observations, so a policy can match
closely and still miss the cube, or drift and still pick it up. They do not
rank the architectures in general either: X-VLA ran with its VLM frozen and
a small batch, and only one seed, dataset and task were run. Latencies were
measured on a shared GPU.

**Calibrated uncertainty.** The fine-tuned specs use `method: logistic`,
fitted per model on the 18 calibration episodes for one event, "chunk mean
joint error > 0.0678 rad" (the pooled median, the same tolerance for both
models). The resulting `u` is a probability of that event for that model
only. On the test episodes both maps beat a constant rate and predict close
to the observed rate (SmolVLA 0.47 predicted against 0.49 observed, X-VLA
0.53 against 0.57). The community specs stay `uncalibrated`.

## What is still open

1. **More than one seed and a full X-VLA fine-tune.** The benchmark is one
   seed. X-VLA's full fine-tune (or LoRA, which would need the `peft`
   package) needs more than 8 GB.
2. **Closed-loop success**, which open-loop error does not measure. The
   Gazebo arm uses primitive geometry and the row-crop world has no red cube,
   so the models' behaviour there says nothing about task skill; recording
   the robot's own episodes, real or simulated, is the route.
3. **Real-arm units.** Check the joint map against a calibrated SO-101 before
   trusting an angle on hardware.
