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

The arm carries the GrabCAD **parallel gripper**, not the hinged jaw of the
standard SO-101 that LeRobot's URDF describes. A micro servo turns a pinion that
drives two racks, so the two fingers slide in opposite directions along a rail.
That changes the joints: the standard arm has one revolute `gripper`, this one
has two prismatic fingers.

The simulation models it that way, from the project's own CAD
(`SO101 Assembly new2.stp`, assembly `gripper v3`):

| | |
|---|---|
| Joints | `gripper_plus`, `gripper_minus`, prismatic, opening in opposite directions along `gripper_link` x |
| Gap between the fingertips, closed | 15.4 mm (CAD: 15.5 mm) |
| Stroke per finger | 14 mm, so 43.4 mm fully open |
| Fingertips reach | 111 mm past the wrist-roll plane |
| Grasp frame | 101 mm out, on the roll axis, in the distal 20 mm of the fingertips |
| Reach, shoulder-lift axis to the grasp frame | 0.4135 m, against 0.4108 m for the standard jaw |
| Finger mass | 20.7 g each: CAD volume times an assumed PLA density of 1240 kg/m^3 |

The five arm joints are unchanged and are pinned to the official URDF by
`tests/test_arm_kinematics.py`; the gripper is pinned to the CAD data by
`tests/test_arm_gripper.py`.

**What this means for the pick.** The CAD pose is the *closed* limit: sliding
either finger inward by 1 mm makes it interfere with the other, so the fingertips
never get closer than 15.4 mm and nothing thinner can be held. Fully open they are
43.4 mm apart, which clears the 40 mm block in the platform plan by only 1.7 mm a
side. That is tight for a gripper positioned by a camera, and it rests on the
stroke below, which is an estimate.

**The old interface survives.** Everything outside the simulator was written
against one joint named `gripper`, commanded in radians from -0.174533 (closed)
to 1.74533 (open): the VLA configs under `configs/arm_policy/` map a policy's
0..100 gripper onto exactly that range, and `carma_vla` addresses the joint by
name. So `carma_sim/scripts/gripper_relay.py` translates at the boundary. A
command on `/carma/arm/gripper/cmd_pos` becomes a finger travel for both fingers,
and the two finger joints are folded back into one `gripper` entry on
`/carma/arm/joint_states`. The mapping is linear, so "50 % open" is a gripper half
open. The angle is a scale, not a physical rotation.

**What was assumed, and what to check on the printed part.** The CAD says how the
gripper is built, not how it is bolted to this arm, so three things are
assumptions (recorded in `gripper_boxes.json`):

1. The wrist-roll axis runs along the CAD y axis through x = 30 mm, z = 20 mm,
   the centre of the wrist block and base plate.
2. The back face of the wrist block is the wrist-roll horn plane, which the URDF
   puts at `gripper_link` z = 0. A sanity check: the fingertips then end 111 mm out
   and the grasp frame 101 mm, against the standard gripper's 98 mm, which is
   what a drop-in replacement should give.
3. The fingers open along `gripper_link` x, the plane the URDF's jaw swings in.

The stroke is also an estimate: no interference with any fixed part up to 12 mm
outward, and 14 mm keeps each rack engaged with the pinion, but the real limit is
set by the printed rack's teeth. Measure the closed gap, the open gap and the
distance from the horn plane to the fingertips on the print, and update
`mechanism` and `frame` in the JSON.

**Where the shape comes from.** `models/so101_follower/gripper_boxes.json` holds
272 axis-aligned boxes sampled from the CAD: 3 mm cells for the wrist block and
base plate, 2 mm for the rail and racks, 1 mm for the fingertips, because their
inner faces set the grip gap. Only the parts that show are drawn; the servo,
pinion and horn sit inside the body. `tools/gripper_from_step.py` rebuilds the
JSON from the STEP file (it needs `pip install cadquery-ocp`, which is not a
project dependency, and reproduces the committed file exactly), and
`tools/build_gripper_sdf.py` generates `model.sdf` from the JSON.

**Licence.** The SO-101 is Apache-2.0. This community gripper's licence was not
checked, and the boxes are a coarse resampling of its shape. Confirm the
author's licence before this repository is shared outside the project; AGENTS.md
section 2.1 asks that everything in it be redistributable.

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
`wrist_roll`, `gripper`. For the first five, frames, limits and masses are the
URDF's, so joint angles mean the same thing in a LeRobot recording and in
Gazebo. `gripper` is the one joint clients see; underneath it are two prismatic
fingers (see above). The arm links are drawn as cylinders rather than the
printed-part meshes, and the gripper as boxes.

```bash
ros2 topic pub --once /carma/arm/shoulder_lift/cmd_pos std_msgs/msg/Float64 '{data: -0.8}'
ros2 topic echo /carma/arm/joint_states --field position
```

The simulated body is the larger photographed platform, not the CAD, so the
arm sits higher there and reaches about 0.13 m past the guard. Use the
simulation for control and for what the cameras see, not for reach numbers.
Checked running in Gazebo Harmonic (2026-10-23, headless, software rendering):
the arm holds its pose with nothing commanded (under 0.002 rad of drift in 3 s);
the gripper reaches a command within 0.12 s and lands within 0.015 rad of it, at
both ends and in the middle, with the fingers 0.2 mm off their targets in
opposite directions, which is gravity along the finger axis; over-range commands
clamp; both cameras stream; and the vehicle drives at 0.292 m/s per simulated
second against 0.300 commanded.

**It slows the simulator.** On this laptop the real-time factor is 0.57 without
the arm, 0.33 with the arm and its earlier cylinder-and-hinge gripper, and 0.28
with the box gripper. The arm's extra camera pair accounts for most of that; the
boxes cost about 15 % more. Distances covered in wall-clock time are therefore
not comparable between runs with and without the arm, only per simulated second.

**The arm camera sees the gripper.** The camera pair sits above the gripper body,
clear of it (the earlier mount would have been inside the new plate), and the
right camera sees a fingertip at the edge of its frame while the left, 60 mm
further out, does not. The mount position is still an estimate. A stereo pair in
which only one eye sees a finger will mismatch there, so mask that edge before
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
