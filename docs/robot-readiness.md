# Robot readiness

What has to be true before CARMA drives the physical platform, what exists,
and how each item is tested before the hardware does. Gazebo stands in for the
robot wherever it can; each row says where it cannot.

| Item | Status | Tested in sim by | Needs the hardware for |
|---|---|---|---|
| Safety gate: e-stop, command watchdog, speed limits | Built (`carma_bridge/safety.py`, `safety_node`) | `tests/test_safety.py`; the gate is in the sim launch | Wiring `/carma/estop` to the hardware stop's status; confirming the motor driver also stops on loss of `/carma/cmd_vel` |
| Pose without ground truth | Built: wheel speed + gyro (`motion.PlanarEstimator`) | `pose_source: estimate`; scored on the true pose | Real gyro noise and bias (the sim IMU is noiseless, see below); GNSS/RTK in the field |
| Stuck/crash detection without world geometry | Built: camera-based stall (`motion.StallDetector`) | Runs beside the geometric collision check; `stall_rate` | Threshold check on real images (dust, wind-blown canopy) |
| Robot interface | Not started | — | A bridge for the robot's own topics: same reply format, no world-control calls |
| Operator can be asked a question | Not started | — | Quest prompt and spoken yes/no; the MQTT operator raises `OperatorUnavailableError` on `ask()` until then |
| Policy server on the robot computer | Not started | — | Latency on the NVIDIA board |
| CARMA's choice changes the outcome | In progress | See `docs/cost-model.md` | — |
| Sim-to-real | Not started | — | Teleop demonstrations recorded on the robot with `scripts/record_nav_demos.py` |

## Known gaps

- **The simulated gyro is noiseless.** Gazebo draws sensor noise from an RNG
  this project does not seed, which would break the replay check. The estimate
  is therefore better in simulation than it will be on the robot. Measure the
  real gyro's drift standing still before trusting an alley run.
- **The estimate does not fix slip.** Distance comes from wheel speed. The stall
  detector catches the stuck case; partial slip in mud is not caught.
- **Stall thresholds are untested on real images.** A canopy moving in the wind
  can make a stalled robot's view change; a featureless view (bare soil, dusk)
  can make a moving robot's view look static.
