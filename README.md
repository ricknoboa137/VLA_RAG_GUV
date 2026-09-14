# CARMA

**Cost-Aware Retrieval Memory Arbitration for field robots.**

CARMA wraps a vision-language-action navigation backbone with a memory of past
human corrections and an arbitration policy that decides, at each step, whether
to act, to retrieve from that memory, or to ask the operator. Operator attention
is treated as a measured cost rather than a free oracle.

The system targets a wheeled differential-drive UGV in outdoor agricultural
environments, with Gazebo Harmonic and ROS 2 for simulation.

## Status

Skeleton. Interfaces and the experiment harness are in place; backbone,
retrieval and cost-model implementations are stubs that run end to end and
produce valid manifests, so the plumbing can be validated before the science
lands.

Expect a success rate of zero on the synthetic simulator. The placeholder
backbone is not goal-directed — the goal pose is used only for scoring and is
deliberately never given to the policy — so a blind heuristic reaching it would
be an accident. Zero here means the harness is behaving correctly, not that it
is broken. What the smoke run does validate is that all three arbitration paths
execute, that the ask budget is enforced, that every step records all three
prices, and that the manifest is complete.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Run an episode

```bash
carma run --config configs/experiment/smoke.yaml --seed 0
carma report runs/
```

Every run writes `runs/<run_id>/manifest.yaml` recording config hash, git
commit, seed, platform and dependency versions. `carma report` refuses to
aggregate runs whose manifests disagree.

## Conditions

| Key | What it isolates |
|---|---|
| `no_memory` | Backbone alone — the floor |
| `unconditional_retrieval` | Retrieval on every step |
| `ask_only` | Uncertainty-gated asking, no memory |
| `carma` | Full arbitration |
| `carma_no_hygiene` | Arbitration without staleness handling |

## Repository layout

```
src/carma/        library — see AGENTS.md for module boundaries
ros2_ws/          ROS 2 packages for the physical and simulated robot
docker/           container image and compose file for the ROS 2 side
configs/          every experiment is a config file, not a CLI flag
docs/             architecture, interfaces, cost model, data schema
tests/            unit suite, plus architecture and determinism gates
scripts/          asset fetching and one-off tooling
```

## On the robot

The ROS 2 side runs in Docker on any machine (`docker/README.md`):

```bash
docker compose -f docker/compose.yaml up --build broker vision
```

`carma_vision` runs scene analyzers on the live camera — your own models plug in
through `configs/analyzers/` (`docs/analyzers.md`) — and streams the stereo
camera over MQTT. VR viewing is covered in `docs/streaming.md`. `carma_voice`
turns operator speech into commands, answers and corrections with offline
speech-to-text (`docs/speech.md`).

## Contributing

Read `AGENTS.md` first. It is the development contract and it is enforced in CI.

## Licence

Apache-2.0. See `LICENSE`.
