# ROS 2 workspace

Three packages, kept out of `src/carma/` on purpose.

- `carma_bridge` — the ROS 2 node that mediates between the `carma` library and
  a running simulator or robot.
- `carma_vision` — live scene analysis (`analysis_node`) and camera streaming
  over MQTT (`stream_node`). See `docs/analyzers.md` and `docs/streaming.md`.
- `carma_sim` — Gazebo Harmonic worlds and the UGV description.

The easiest way to get a working ROS 2 environment on any machine is the
container in `docker/`; see `docker/README.md`.

ROS packages may import `carma`; `carma` never imports ROS.

## Why the library has no ROS dependency

`rclpy` is not a dependency of `carma`. The library speaks to `carma_bridge`
over a local transport, and the bridge owns every ROS import. This is what lets
the unit suite run in a plain virtualenv with no ROS installation, which keeps
CI cheap and lets a reviewer run the tests without a robotics stack.

Do not add `rclpy` to `pyproject.toml`. If a library module seems to need it,
the logic belongs in the bridge.

## Build

```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```
