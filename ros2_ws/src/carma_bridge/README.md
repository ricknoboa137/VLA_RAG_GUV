# carma_bridge

Mediates between the `carma` library and ROS 2. The library's
`GazeboAdapter` talks to this node; the node owns every ROS import.

## Topic contract

This contract is the interface `GazeboAdapter` is written against. Changing a
name or a type here is a breaking change and must be reflected in
`src/carma/sim/gazebo.py` in the same commit.

| Direction | Topic | Type | Meaning |
|---|---|---|---|
| bridge -> robot | `/carma/cmd_vel` | `geometry_msgs/Twist` | Velocity command |
| robot -> bridge | `/carma/stereo/left/image_raw` | `sensor_msgs/Image` | Left camera, `bgr8`; the monocular image when not stereo |
| robot -> bridge | `/carma/stereo/right/image_raw` | `sensor_msgs/Image` | Right camera, `bgr8` |
| robot -> bridge | `/carma/camera/depth` | `sensor_msgs/Image` | Depth, `32FC1`, metres |
| robot -> bridge | `/carma/odom` | `nav_msgs/Odometry` | Pose in the map frame |
| bridge -> library | `/carma/observation` | serialised `Observation` | Synchronised bundle |
| library -> bridge | `/carma/reset` | `std_srvs/Trigger` | Episode reset |

## Synchronisation

Image, depth and odometry are synchronised with an approximate-time policy
before an `Observation` is emitted. The slop is a parameter; record it in the
run manifest when the adapter lands, because a loose slop silently degrades
every pose-conditioned memory entry.

## Who publishes the robot side

In simulation, `carma_sim` does. Its UGV model publishes on the Gazebo topics
named above and `carma_sim/config/bridge.yaml` maps them one-to-one onto these
ROS names — nothing is renamed, so a missing topic is a typo rather than a
remap to chase. `ros2 launch carma_sim row_crop.launch.py` brings that side up
headless and it is live now: all four robot→bridge topics publish at their
nominal rates (stereo and depth 15 Hz, odometry 30 Hz) and `/carma/cmd_vel`
moves the robot. See `ros2_ws/src/carma_sim/README.md` for the sensor geometry
the images and depth come from.

One wrinkle belongs to simulation alone: Gazebo's renderer cannot produce BGR,
so the stereo pair arrives as `rgb8` and `carma_sim`'s `bgr_relay` swaps the
channels onto the `bgr8` topics named above. The contract is unchanged — the
topics here carry `bgr8` either way — and nothing on this side needs to know.

On the robot, the camera driver and the base controller publish the same
topics. That is the point of naming them here rather than in either package.

## Status

Node skeleton only: the robot→bridge half of the contract is satisfied by
`carma_sim`, but this node does not yet synchronise those topics into an
`Observation`, and `GazeboAdapter` in `src/carma/sim/gazebo.py` does not yet
consume one. Implement both against the contract above; until then run
experiments with `sim.kind=synthetic`.
