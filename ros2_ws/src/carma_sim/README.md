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
| `BARE_SOIL` | — (no crop) | — | 1.50 |
| `EMERGENCE` | 0.10 | 0.12 | 1.38 |
| `VEGETATIVE` | 0.60 | 0.45 | 1.05 |
| `CANOPY_CLOSURE` | 1.20 | 0.80 | 0.70 |
| `SENESCENCE` | 1.10 | 0.70 | 0.80 |
| `POST_HARVEST` | 0.18 | 0.35 | 1.15 |

Row spacing (1.5 m), row count, row length, obstacle placement and lighting are
identical across stages. The robot is 0.50 m wide over the wheels, so every
stage including `CANOPY_CLOSURE` leaves the same routes drivable; what changes
is how much of the alley is left and how much of the scene is canopy.

Other world arguments — `rows`, `row_spacing`, `row_length`, `obstacles` — exist
for debugging. Changing them changes the world the numbers came from, so leave
them at their defaults for anything that reaches a paper.

## The robot

`models/carma_ugv/` is a differential-drive UGV with a stereo pair and a
co-located depth camera.

| Quantity | Value |
|---|---|
| Wheel radius | 0.100 m |
| Wheel separation | 0.440 m |
| Overall width over wheels | 0.500 m |
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

`wheel_separation` and `wheel_radius` in the `DiffDrive` plugin must match the
wheel link poses. A mismatch is a systematic scale error in `/carma/odom` that
corrupts every pose-conditioned memory entry without ever looking like a bug.

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
and `/carma/joint_states`.

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
