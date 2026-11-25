# The SO-101 arm

The platform carries one manipulator: the open-source **SO-101 follower arm**
(TheRobotStudio / Hugging Face LeRobot, Apache-2.0), printed in PLA+, with
12 V Feetech STS3215 servos. Its job is to pick an object up from the ground
ahead of the robot while it drives along a row. The leader arm, if one is
built, stays with the operator.

## Where it sits, and why there

On a 5 mm aluminium plate over the front bumper, 140 mm above the ground. The
arm's reach is fixed by its links — 0.410 m from the shoulder-lift axis to the
point between the jaws, from the official URDF — so the only way to get the
gripper onto the ground is to mount it low. The three mounts that were checked:

| Mount | Deck height | Top-down grasp on the ground ahead |
|---|---|---|
| **Front plate over the bumper** | 0.140 m | out to 0.31 m past the bumper |
| Sloped hood | 0.237 m | out to 0.12 m past the bumper |
| Rear top cover | 0.308 m | none, and it collides with the GNSS antenna |

From the front plate it also reaches a plant row 0.35 m off the centreline at
0.2–0.4 m above ground, on both sides. Nothing behind the front axle is in
reach.

What it does to the vehicle: 0.86 kg with its mount, overall length 558 →
638 mm, front axle load 36 → 40 %, backward tip angle 37 → 40° (it helps), and
39° at worst with the arm straight out at any pan angle holding 0.2 kg. It has
its own 36 V → 12 V 20 A converter, because six stalled servos draw 16 A and
must not brown out the steering servo on the shared 12 V supply.

The checks, the CAD and the bill of materials live in the platform CAD project
(`CARMA-platform`, kept outside this repository): `cad/arm_check.py` and
section 15 of `BOM-dimensions.md`.

## The camera cannot see what the arm picks up

The pick zone is directly ahead of the bumper, and the hood's front edge hides
exactly that strip from the head camera — on the real platform and in the
simulation alike. The head camera finds the object on the approach; the
**stereo camera on the arm** — the project's 3D USB camera — has to guide the
grasp. Drive with the
arm folded flat: upright, it shadows more of the ground ahead.

## The camera on the arm

The project's 3D USB stereo camera rides on the arm and is the VLA pipeline's
primary view (`docs/vla-integration.md`); the world camera on the platform is
the headset's. It moves the arm's centre of mass outward, so re-check that the
arm holds its poses once the camera is mounted.

## The gripper

The follower is built with the **standard SO-101 hinged gripper**: the fixed
jaw is the printed `wrist_roll_follower` part, which holds the gripper servo,
and the moving jaw swings about that servo's horn. That is the gripper the
official URDF and LeRobot describe, so there is one revolute joint, `gripper`,
everywhere: on the real arm, in the VLA configs under `configs/arm_policy/`
(which map a policy's 0..100 onto it), in `carma_vla`, and in Gazebo.

| | |
|---|---|
| Joint | `gripper`, revolute about the servo horn, the URDF's origin and axis |
| Range | -0.174533 rad (closed) to 1.74533 rad (fully open), -10° to +100° |
| Moving jaw mass | 12 g (URDF) |
| Gripper body mass | 87 g, fixed jaw with its servo (URDF) |
| Grasp frame | `gripper_frame`, the URDF's `gripper_frame_link`, between the jaw tips |
| Reach, shoulder-lift axis to the grasp frame | 0.411 m |

`tests/test_arm_kinematics.py` pins the joint's origin, axis, limits and the
jaw's mass to the official URDF with the five arm joints, and
`tests/test_arm_gripper.py` checks the meshes, the grasp frame, the controller
and the topics.

**Meshes.** The three gripper parts are drawn from the URDF's own meshes of
the printed parts and the servo, vendored unmodified in
`carma_sim/models/so101_follower/meshes/` (Apache-2.0; provenance and checksums
in the README there). At full detail they are 76 146 triangles, which cost
the row-crop world about a third of its speed, so the simulation uses two
light derivatives that `carma_sim/tools/build_gripper_geometry.py` makes from
them: the meshes simplified on a 1.5 mm grid for drawing (about 7 000
triangles, indistinguishable at camera distance), and boxes for collision, one
per slab along each jaw (8 per jaw) and one for the servo, which are also what
grasping wants: box contacts are cheap and stable, triangle meshes are neither.
The tests rebuild both and fail on any difference.

**Controller.** The jaw is light, 1.6e-5 kg m^2 about its hinge, about a
hundredth of what the arm joints carry, so it has its own gains (P 4, I 1,
D 0.005, with the joint's own damping doing most of the settling). With the arm
joints' gains the controller's D term alone would be unstable at the world's
1 ms step; `tests/test_arm_gripper.py` checks the margin.

**Earlier model.** Until 2026-11-25 the simulation carried a rack-and-pinion
parallel gripper from GrabCAD, as two prismatic fingers drawn as boxes, with a
relay node translating the `gripper` command. The assembled follower has the
standard jaw, so that model, its relay and its CAD tools were removed; they are
in the git history before that date.

## In simulation

Off by default, so every recorded world expands exactly as before:

```bash
ros2 launch carma_sim row_crop.launch.py arm:=true
```

That spawns `models/carma_ugv_arm`, which is `carma_ugv` merged in unchanged
plus `models/so101_follower` on a plate over the front guard. Topics:

| ROS topic | Type | Direction |
|---|---|---|
| `/carma/arm/<joint>/cmd_pos` | `std_msgs/Float64`, radians, for the five arm joints and `gripper` | to the arm |
| `/carma/arm/joint_states` | `sensor_msgs/JointState`, six joints ending in `gripper` | from the arm |
| `/carma/arm/stereo/{left,right}/image_raw` | `sensor_msgs/Image`, `bgr8` | arm stereo camera |
| `/carma/arm/stereo/{left,right}/camera_info` | `sensor_msgs/CameraInfo` | arm stereo camera |

Joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. For all six, frames, limits and masses are the
URDF's, so joint angles mean the same thing in a LeRobot recording and in
Gazebo. The arm links are drawn as cylinders rather than the printed-part
meshes; the gripper is drawn from light copies of the URDF's meshes and collides
as boxes (see above).

```bash
ros2 topic pub --once /carma/arm/shoulder_lift/cmd_pos std_msgs/msg/Float64 '{data: -0.8}'
ros2 topic pub --once /carma/arm/gripper/cmd_pos std_msgs/msg/Float64 '{data: 1.2}'
ros2 topic echo /carma/arm/joint_states --field position
```

The simulated body is the larger photographed platform, not the CAD, so the
arm sits higher there and reaches about 0.13 m past the guard. Use the
simulation for control and for what the cameras see, not for reach numbers.
Checked running in Gazebo Harmonic (headless, software rendering). On
2026-10-23: both arm cameras stream, and the vehicle drives at 0.292 m/s per
simulated second against 0.300 commanded. On 2026-11-25, with the hinged
gripper: from a step command the jaw comes within 0.01 rad of fully open or
closed in about 0.3 s (the servo's 4.7 rad/s over the 1.92 rad stroke) and
then holds there; a half-open command lands within 0.003 rad; over-range
commands clamp at the joint limit; with nothing commanded the arm drifts under
0.005 rad in 1.8 s; and through the launch file the ROS command reaches the jaw
and `/carma/arm/joint_states` reports it as `gripper`.

**It slows the simulator.** Measured on this laptop on 2026-11-25 as simulated
time over wall-clock time across a minute of running: 0.79 without the arm and
0.45 with it. The same day the earlier box gripper gave 0.39, and the gripper
drawn and collided with the full-detail meshes 0.29, which is why the simulation
uses the light copies. Earlier figures (0.57, 0.33 and 0.28, from 2026-10-23)
were read from Gazebo's instantaneous `real_time_factor`, which swings by a
factor of fifty from one reading to the next here, so they are not comparable.
Distances covered in wall-clock time are therefore not comparable between runs
with and without the arm, only per simulated second.

**The arm camera sees the gripper.** The camera pair sits above the gripper body,
clear of it (the earlier mount would have been inside the new plate), and the
right camera sees the jaw tips at the edge of its frame (both jaws when closed,
the fixed jaw when open) while the left, 60 mm further out, sees neither. The mount position is still an estimate. A stereo pair in
which only one eye sees a jaw will mismatch there, so mask that edge before
computing depth near it.

## On the real arm

Driven with LeRobot over the bus-servo adapter's USB serial port. Two steps
need a person at the arm: plugging in one servo at a time while its ID is set,
and moving each joint through its range during calibration.

```bash
lerobot-find-port
lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>
lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=carma_arm
```

LeRobot is not a dependency of the `carma` library, and nothing in `src/`
imports it.

## Driven by a VLA

`scripts/policy_server.py --policy smolvla` or `--policy xvla` serves a
vision-language-action model that answers with joint-target chunks, and
`carma_vla` with `arm:=true` plays them out on `/carma/arm/<joint>/cmd_pos`,
from the arm camera, the world camera and the operator's spoken instruction.
Checked in Gazebo with both models. The arm's angles cross the wire in this
URDF's radians; each model's own units are converted by its spec in
`configs/arm_policy/`. Details, and how the two models are compared:
`docs/arm-policies.md`.

## What the arm is not, yet

It is not part of the arbitration. No `Protocol` in `carma.types` describes a
manipulator, no arbiter prices a grasp, and none of the five experimental
conditions in AGENTS.md section 5 uses it. Bringing the arm into the loop —
for example, pricing "ask the operator to confirm the grasp" against "grasp" —
would be a new claim, and needs a Protocol change, a cost-model entry in
`docs/cost-model.md` and its own condition before any number from it is
reported.
