# Hidden hazards

The scenario in which CARMA's choice is supposed to matter: a spot on the
route that the robot must not enter and that its camera cannot see, such as a
wet patch or a buried stone. The backbone alone cannot avoid it. Operator
knowledge can, whether asked for now (ASK) or remembered from an earlier pass
(RETRIEVE).

## Pieces

| Piece | Where | What it does |
|---|---|---|
| `Hazard` | `carma.types.core` | Position, keep-out radius, and the operator's advice text |
| `EpisodeSpec.hazards` | `carma.types.core` | When present, `goal` is the stop point before the hazard |
| Route options | `row_alley_episodes(hazard_probability, hazard_x_m, ...)` | Place hazards; at probability 0 the routes are exactly as before |
| Hazard check | `carma.runtime.episode` | Ends the episode as a failure when the true pose enters a keep-out (`entered_hazard`, `hazard_rate`) |
| No arrival oracle | `carma.runtime.episode` | With hazards, reaching the goal radius does not end the episode; only the robot's own stop does |
| Operator knowledge | `ScriptedOperator(advice_range_m, hazard_warning_rate)` | Gives the advice when asked within range; volunteers it in a chosen fraction of episodes |
| Answers become memory | `carma.runtime.episode` | An operator's answer is ingested like a volunteered correction |
| Spatial memory | `SpatialRetriever(radius_m)` | Scores entries by distance from where they were given |
| Context to the policy | `carma.backbone.context.task_text` | Folds context into the model's one text input; the recorder, dataset and server share it |
| Demonstrations | `record_nav_demos.py --hazard-probability --distractor-probability` | The expert obeys the advice from within range; distractors teach it to ignore unrelated remarks |

The hazard does not exist in Gazebo. It is a property of the episode, checked
on the true pose, so it behaves the same in every simulator and needs no world
changes. On the robot a real hazard has a real effect, which the stall check
(`docs/simulation.md`) is there to catch.

## First result: the cost model ignores memory when the policy is confident

Synthetic simulator, the scripted expert as backbone (it obeys advice
perfectly), `configs/experiment/hazard_study/` with only the backbone and sim
swapped, 8 episodes, 2026-11-14:

| Condition | Success | Hazard entered | Choices |
|---|---|---|---|
| no_memory | 0/8 | 8/8 | act only |
| unconditional_retrieval | 7/8 | 1/8 | retrieve only; the first episode had no memory yet |
| ask_only | 0/8 | 8/8 | never asked |
| carma | 0/8 | 8/8 | never retrieved |
| carma_no_hygiene | 0/8 | 8/8 | never retrieved |

The mechanics work: once a warning is in spatial memory, retrieving it stops
the robot at the right place every time. CARMA never retrieves because the
cost model (`docs/cost-model.md`) prices acting at `u * R`, and the expert's
uncertainty is 0. A hidden hazard is exactly where a policy is confident and
wrong, so `u` cannot signal it; the probe's gain `g` only discounts a risk
that `u` has already set to zero. This is a property of the cost model, not of
the scenario, and changing it is a decision about the method itself.

## With memory counted as risk

`docs/cost-model.md` now prices risk as `r = max(u, rho * g)`. Same setup,
`memory_risk: 0.5`, 2026-11-14:

| Condition | Success | Hazard entered | Choices |
|---|---|---|---|
| no_memory | 0/8 | 8/8 | act only |
| unconditional_retrieval | 7/8 | 1/8 | 954 retrievals |
| ask_only | 0/8 | 8/8 | never asked |
| carma | 7/8 | 1/8 | 771 retrievals, 183 acts |
| carma_no_hygiene | 7/8 | 1/8 | 771 retrievals, 183 acts |

carma matches unconditional retrieval at 19% fewer retrievals: it acts freely
where memory knows nothing. The saving is small because the 8 m radius covers
most of the route before the patch. Every condition loses the first episode:
memory is empty and a confident backbone never asks. Hygiene makes no
difference here because the hazard never moves; a patch that dries up between
passes is the case that would separate them.

## With a trained policy in Gazebo

`nav_hz/003000`, a navigation SmolVLA trained on demonstrations that carry the
advice in the task text, same configs, 2026-11-15
(`training/results/2026-11-15_nav-hazard/`):

| Condition | Success | Hazard entered | Choices |
|---|---|---|---|
| no_memory | 0/8 | 8/8 | 1159 acts |
| unconditional_retrieval | 7/8 | 1/8 | 1090 retrievals |
| ask_only | 0/8 | 8/8 | never asked |
| carma | 7/8 | 1/8 | 902 retrievals, 188 acts |
| carma_no_hygiene | 7/8 | 1/8 | 902 retrievals, 188 acts |

The learned policy obeys retrieved advice as the expert did, and the
synthetic pattern holds: carma matches unconditional retrieval at 17% fewer
retrievals. Its uncertainty stays near 0.03 to 0.06, so asking is never priced
below acting and the first episode is lost everywhere.
