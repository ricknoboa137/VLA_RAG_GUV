# SO-101 gripper meshes

The three parts of the standard SO-101 hinged gripper, as the follower arm is
built: the fixed jaw, the moving jaw and the gripper servo, as the official URDF
draws them. They are the source; the simulation does not load them directly.

`../../../tools/build_gripper_geometry.py` makes two light derivatives from them,
because at full detail (76 146 triangles) they cost the row-crop world about a
third of its speed:

- `visual/`: the same meshes simplified on a 1.5 mm grid, about 7 000 triangles
  in all, which `../model.sdf` draws at the URDF's poses.
- Collision boxes, 8 slabs per jaw and one for the servo, written into
  `../model.sdf` between `BEGIN generated` markers.

Regenerate both with that script after changing anything here; the tests
rebuild them and fail on any difference.

| File | Part | Triangles | SHA-256 |
|---|---|---|---|
| `wrist_roll_follower_so101_v1.stl` | Fixed jaw and wrist-roll housing (printed) | 28 796 | `4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4` |
| `moving_jaw_so101_v1.stl` | Moving jaw (printed) | 28 270 | `785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d` |
| `sts3215_03a_v1.stl` | Feetech STS3215 servo that drives the jaw | 19 080 | `a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0` |

Source: TheRobotStudio/SO-ARM100, `Simulation/SO101/assets/`, from the `main`
branch archive downloaded on 2026-11-20 (latest changelog entry v0.1.13).
Copied unmodified. They are binary STL in metres, in the frames the URDF's
`<visual>` origins assume.

Licence: Apache-2.0, the upstream repository's licence, reproduced in
`LICENSE` beside them.

`tests/test_arm_gripper.py` checks the checksums above, so a changed mesh is a
deliberate change.
