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

## Status

Placeholder. The world and the robot description land with WP1.
