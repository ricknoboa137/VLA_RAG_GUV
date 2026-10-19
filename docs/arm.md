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
**wrist camera** that ships with the arm has to guide the grasp. Drive with the
arm folded flat: upright, it shadows more of the ground ahead.

## In simulation

Off by default, so every recorded world expands exactly as before:

```bash
ros2 launch carma_sim row_crop.launch.py arm:=true
```

That spawns `models/carma_ugv_arm`, which is `carma_ugv` merged in unchanged
plus `models/so101_follower` on a plate over the front guard. Topics:

| ROS topic | Type | Direction |
|---|---|---|
| `/carma/arm/<joint>/cmd_pos` | `std_msgs/Float64`, radians | to the arm |
| `/carma/arm/joint_states` | `sensor_msgs/JointState` | from the arm |
| `/carma/arm/wrist/image_rgb` | `sensor_msgs/Image`, `rgb8` | wrist camera |
| `/carma/arm/wrist/camera_info` | `sensor_msgs/CameraInfo` | wrist camera |

Joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. Frames, limits and masses are the URDF's, so joint
angles mean the same thing in a LeRobot recording and in Gazebo. The links are
drawn as cylinders rather than the printed-part meshes.

```bash
ros2 topic pub --once /carma/arm/shoulder_lift/cmd_pos std_msgs/msg/Float64 '{data: -0.8}'
ros2 topic echo /carma/arm/joint_states --field position
```

The simulated body is the larger photographed platform, not the CAD, so the
arm sits higher there and reaches about 0.13 m past the guard. Use the
simulation for control and for what the cameras see, not for reach numbers.
The joint PID gains have not yet been tuned against a running simulation.

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

## What the arm is not, yet

It is not part of the arbitration. No `Protocol` in `carma.types` describes a
manipulator, no arbiter prices a grasp, and none of the five experimental
conditions in AGENTS.md section 5 uses it. Bringing the arm into the loop —
for example, pricing "ask the operator to confirm the grasp" against "grasp" —
would be a new claim, and needs a Protocol change, a cost-model entry in
`docs/cost-model.md` and its own condition before any number from it is
reported.
