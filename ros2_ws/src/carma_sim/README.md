# carma_sim

Gazebo Harmonic worlds and the UGV description.

## Worlds

| File | Purpose |
|---|---|
| `worlds/row_crop.sdf` | Row-crop field, the default experimental world |

A world must be parameterised by crop stage, because the drift study depends on
running the same routes at two phenological stages. Encode the stage as a world
argument rather than forking the file, so the two runs differ by one recorded
parameter instead of by an untracked edit.

### The stage argument

`worlds/row_crop.sdf` is a **xacro template**, not a world you hand straight to
`gz sim`. `launch/row_crop.launch.py` expands it before starting the server:

```bash
ros2 launch carma_sim row_crop.launch.py stage:=VEGETATIVE
ros2 launch carma_sim row_crop.launch.py stage:=CANOPY_CLOSURE
```

`stage` takes the names of `carma.types.PhenologyStage`. The launch file
validates it against that list, so a typo fails the launch rather than quietly
rendering the default stage — a run recorded as `CANOPY_CLOSURE` that actually
rendered `VEGETATIVE` is an unreproducible number.

The expansion is written to `~/.ros/log/carma_sim/row_crop_<stage>.sdf` and
kept, so the exact world an episode ran against is recoverable afterwards. To
expand one by hand:

```bash
xacro worlds/row_crop.sdf stage:=CANOPY_CLOSURE > /tmp/row_crop.sdf
```

What the stage changes, and nothing else does:

| Stage | Canopy height (m) | Canopy width (m) | Alley width (m) |
|---|---|---|---|
| `BARE_SOIL` | — (no crop) | — | 1.80 |
| `EMERGENCE` | 0.10 | 0.12 | 1.68 |
| `VEGETATIVE` | 0.60 | 0.45 | 1.35 |
| `CANOPY_CLOSURE` | 1.20 | 0.80 | 1.00 |
| `SENESCENCE` | 1.10 | 0.70 | 1.10 |
| `POST_HARVEST` | 0.18 | 0.35 | 1.45 |

Row spacing (1.8 m), row count, row length, obstacle placement and lighting are
identical across stages. The tractor is 0.72 m wide over its rear tyres, so
every stage including `CANOPY_CLOSURE` leaves the same routes drivable, with
0.14 m of clearance each side at the tightest; what changes is how much of the
alley is left and how much of the scene is canopy.

Spacing was 1.5 m while the robot was a narrow differential drive. The tractor
does not fit the 0.70 m alley that produced, so the default moved to 1.8 m.
Any numbers recorded against the old spacing are not comparable with numbers
recorded against this one.

`arm:=true` spawns the vehicle with the SO-101 arm on its front (see
[The arm](#the-arm) below). It defaults to `false`, and with it off the world
expands to the same robot as before the arm existed.

Other world arguments — `rows`, `row_spacing`, `row_length`, `obstacles` — exist
for debugging. Changing them changes the world the numbers came from, so leave
them at their defaults for anything that reaches a paper.

## The robot

`models/carma_ugv/` is a tractor-style UGV: rear-wheel drive on two large
wheels, front wheels steered through Ackermann linkage, matching the physical
platform. It carries a stereo pair and a co-located depth camera.

### Body shape

The hull is modelled on the physical machine: a flat aluminium deck, a raised
rear housing carrying the emergency stop, a sloped nose down to the bonnet, a
tubular front guard, mudguards over the driven rear wheels, a GNSS puck on a
short mast, a radio whip, and the stereo pair on a pan housing above the
bonnet. Wheels have inset rims so they read as tyres on wheels.

All of that is **visual only**. Collision is still the two boxes it always was,
and mass and inertia are untouched, so reshaping the body changed no dynamics:
drive and odometry numbers stay comparable with runs recorded before it. The
alley clearance is set by the rear tyres at 0.72 m, which is wider than any
part of the body, so nothing on the hull decides whether the machine fits.

The GNSS mast and antenna carry no sensor — there is no GNSS topic in the
bridge contract. They exist so the machine is recognisable in the viewport and
in screenshots, nothing more.

**No part of the robot appears in its own camera images** — with the arm off;
see below for the arm. The mast and whip
sit 0.58 m behind the camera, and the front guard, at 0.28 m ahead and 0.19 m
off the ground, falls below the bottom of the frame — the view at that range
starts 0.57 m up. So the body shape does not enter the perception path, and a
simulated frame is not directly comparable with a field photograph taken from
a camera that does see its own bonnet. If self-occlusion matters later, it has
to be arranged deliberately by moving the camera back or down.

**It cannot turn on the spot.** That is the single most important consequence
of the running gear for route design:

```
minimum turning radius = wheel_base / tan(steering_limit)
                       = 0.70 / tan(0.70) = 0.83 m
```

The narrowest alley is 1.00 m at `CANOPY_CLOSURE`, so a U-turn between the rows
is not available. The vehicle turns at the headland, beyond the end of the
rows, which is how the real machine is driven. A route that assumes it can
pivot in place will not execute.

| Quantity | Value |
|---|---|
| Wheel base (rear axle to front axle) | 0.700 m |
| Wheel separation (rear track) | 0.600 m |
| Kingpin width (front track) | 0.520 m |
| Rear wheel radius, driven | 0.200 m |
| Front wheel radius, steered | 0.140 m |
| Steering limit | 0.700 rad (40°) |
| Minimum turning radius | 0.83 m |
| Overall width over rear tyres | 0.720 m |
| **Stereo baseline** | **0.120 m** (left `y = +0.060`, right `y = -0.060`) |
| Camera height above ground | 0.750 m |
| Camera pitch | 0.150 rad down |
| Resolution | 640 × 480 |
| Horizontal FOV | 1.0472 rad (60°) |
| `fx` = `fy` | 554.26 px = (width / 2) / tan(hfov / 2) |
| `cx`, `cy` | 320.0, 240.0 px |
| Depth range | 0.10 m … 20.0 m, `32FC1`, metres |
| Camera rate | 15 Hz |
| Odometry rate | 30 Hz |

These numbers are stated in `models/carma_ugv/model.sdf` and are published on
`/carma/stereo/{left,right}/camera_info`, so nothing downstream has to hard-code
them. The stereo baseline gives a depth resolution of roughly
`z² / (fx · b) = z² / 66.5` metres per pixel of disparity — about 1.5 cm at 1 m
and 0.54 m at 6 m, which is the useful range for row following.

The depth camera sits at exactly the left camera's pose and shares its
intrinsics, so depth is registered to `/carma/stereo/left/image_raw` with no
reprojection step.

`wheel_base`, `wheel_separation`, `kingpin_width` and `wheel_radius` in the
`AckermannSteering` plugin must match the link poses they describe. A mismatch
is a systematic scale or heading error in `/carma/odom` that corrupts every
pose-conditioned memory entry without ever looking like a bug.

The running gear changed from differential drive to Ackermann, but the bridge
contract did not: `AckermannSteering` consumes the same
`geometry_msgs/Twist` on `/carma/cmd_vel` and publishes the same
`nav_msgs/Odometry` on `/carma/odom`. What changed is which commands are
*achievable* — `angular.z` with `linear.x` near zero no longer turns the
vehicle, because there is no steering authority without forward motion.

### The arm

`models/so101_follower/` is the SO-101 follower arm, and `models/carma_ugv_arm/`
is `carma_ugv` merged in unchanged with that arm nested as `arm` on a plate over
the front guard. `arm:=true` spawns the second one, still named `carma_ugv`.

| Quantity | Value |
|---|---|
| Arm base origin in `base_link` | x 0.771, z 0.013 (0.213 m above ground) |
| Reach, shoulder-lift axis to the jaws | 0.410 m |
| Arm mass | 0.632 kg, link by link from the URDF |
| Joint effort / speed limit | 2.94 N·m / 4.7 rad/s (12 V STS3215) |
| Arm stereo camera | 60 mm baseline, 640 × 480 per eye, 58.9° horizontal, 15 Hz, looking down the jaws |

Kinematics, joint limits and link masses are copied from the official URDF
(`Simulation/SO101/so101_new_calib.urdf` in TheRobotStudio/SO-ARM100,
Apache-2.0); the links are drawn as cylinders, not the printed-part meshes, so
nothing is fetched. Checked running: each joint settles within 0.05 rad of
its command, and the vehicle still drives on `/carma/cmd_vel` with the arm on.

The head camera sees the ground from about 1.8 m ahead, and the pick zone ends
about 1.0 m ahead of `base_link`, so the head camera never sees a grasp; the
arm's stereo camera does. The arm enters the head camera's frame only when lifted
above about 0.5 m at the front. The chassis inertia is not adjusted for the
arm, which is why `arm:=true` stays off for recorded runs.

## Topics

The model publishes on the Gazebo topics named in the bridge contract
(`ros2_ws/src/carma_bridge/README.md`), and `config/bridge.yaml` maps them
one-to-one onto ROS. Nothing is renamed, so a missing topic is a typo in one of
the three files rather than a remap to chase.

| ROS topic | Type | Source |
|---|---|---|
| `/carma/cmd_vel` | `geometry_msgs/Twist` | subscribed by `DiffDrive` |
| `/carma/stereo/left/image_raw` | `sensor_msgs/Image`, `bgr8` | `stereo_left` via `bgr_relay` |
| `/carma/stereo/right/image_raw` | `sensor_msgs/Image`, `bgr8` | `stereo_right` via `bgr_relay` |
| `/carma/camera/depth` | `sensor_msgs/Image`, `32FC1` m | `depth` |
| `/carma/odom` | `nav_msgs/Odometry` | `DiffDrive` |

Also bridged, outside the contract: the three `camera_info` topics, `/clock`
and `/carma/joint_states`; and, published only with `arm:=true`,
`/carma/arm/<joint>/cmd_pos`, `/carma/arm/joint_states` and
`/carma/arm/stereo/{left,right}/{image_rgb,camera_info}`, relayed to `bgr8` on
`/carma/arm/stereo/{left,right}/image_raw`.

### Why the stereo pair takes a detour

The contract says `bgr8`, and AGENTS.md section 7 fixes BGR as the project's
image convention because that is what OpenCV expects and what the real camera
driver publishes. Gazebo cannot produce it: `<format>B8G8R8</format>` is
accepted by SDFormat but the ogre2 render path rejects it at run time with
`Unsupported pixel format [8]`, and the sensor falls back to RGB without
failing. `ros_gz_bridge` passes the pixel format straight through, so the topic
would carry `rgb8` under a name the contract declares as `bgr8` — a channel
swap that looks like nothing at all until a colour-sensitive analyzer quietly
reports the wrong thing.

So the cameras publish `rgb8` on `/carma/stereo/{left,right}/image_rgb`, and
`scripts/bgr_relay.py` swaps the channels onto the contract's `.../image_raw`.
It costs one copy per frame and exists only in simulation; on the robot the
driver publishes `bgr8` directly and the relay does not run. Depth and odometry
need no such treatment and go straight through.

### Measured rates

Nominal is 15 Hz for the cameras and 30 Hz for odometry. On a CPU-only host —
Docker Desktop with Mesa's software EGL — the observed rates are about 10 Hz
and 20 Hz, because the render loop cannot hold real-time factor 1. The topics
are steady at those rates, not intermittent. On a machine with a GPU the
nominal rates are reached. Record the rate you actually got: a run whose
cameras ran at two thirds of nominal saw two thirds as many decision points.

## Bring-up

Headless, in the container that already has Gazebo:

```bash
docker compose -f docker/compose.yaml --profile sim up --build sim
```

Keep the `--build`. The `sim` service runs the workspace compiled into the
image, not `ros2_ws/src` on the host, so an edit to the world, the model or the
relay has no effect until the image is rebuilt — and the service starts and
publishes happily on the old code, which makes the mistake hard to notice.

Or in any sourced ROS 2 Jazzy environment with `ros_gz` and `xacro` installed:

```bash
ros2 launch carma_sim row_crop.launch.py
```

`headless:=false` opens the GUI instead, which needs a display.

Check it is alive:

```bash
ros2 topic hz /carma/stereo/left/image_raw /carma/stereo/right/image_raw \
              /carma/camera/depth /carma/odom
ros2 topic pub -r 10 /carma/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.4}, angular: {z: 0.1}}'
ros2 topic echo /carma/odom --field pose.pose.position
```

`linear.x` must be non-zero for `angular.z` to do anything: the steered wheels
change the vehicle's path, they do not rotate it in place. A command of
`{linear: {x: 0.0}, angular: {z: 0.5}}` moves nothing, which is correct
behaviour for this platform and not a fault in the bridge.

## No Fuel, ever

Nothing in this package includes a model from `fuel.gazebosim.org`. Ground,
lighting, crop and obstacles are all inline. AGENTS.md section 2.3 forbids
network access at run time, and a world that silently downloads a mesh is a
world that produces different numbers on a machine that is offline.

## Status

The world, the UGV description and the launch file are in place: WP1 is done
for this package. `ros2 launch carma_sim row_crop.launch.py` brings the field
up headless, publishes the four robot→bridge topics, and drives on
`/carma/cmd_vel`.

Still open: `GazeboAdapter` in `src/carma/sim/gazebo.py` and the synchroniser
in `carma_bridge` are stubs, so the library cannot yet run an episode against
this world. Until they land, run experiments with `sim.kind=synthetic`.
