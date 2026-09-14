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

## Status

Node skeleton only. Implement against the contract above; until then run
experiments with `sim.kind=synthetic`.
