# CARMA — development contract

This file is the single source of truth for how code in this repository is
written, reviewed and extended. Every contributor reads this before touching
`src/`. Where this file and a habit disagree, this file
wins. Where this file and `docs/architecture.md` disagree, that is a bug in one
of them — fix it in the same change, do not work around it.

The project is research code that produces numbers which go into papers. That
single fact drives every rule below: a result that cannot be reproduced from a
clean checkout is not a result.

---

## 1. What this system is

CARMA (Cost-Aware Retrieval Memory Arbitration) is a layer around a
vision-language-action navigation backbone running on a wheeled field robot. At
each decision point it chooses between three options:

- **ACT** — emit the backbone's action directly.
- **RETRIEVE** — query a memory of past operator corrections, then act on the
  retrieved context.
- **ASK** — put a question to the human operator and wait.

The contribution is the arbitration between those three under a shared cost
model, and the measurement of what each choice costs the operator. Keep that
framing in mind: **operator attention is a measured resource, not a free
oracle.** Any change that makes the system quietly ask more often without
accounting for it defeats the purpose of the project.

---

## 2. Non-negotiables

1. **Open components only.** Every model, dataset and dependency must be
   redistributable under an OSI-approved or equivalent open licence. No
   proprietary API calls in the runtime path, no gated checkpoints, no
   credential-bearing config. If a component cannot be obtained by a stranger
   with a clean machine and no accounts, it does not go in.
2. **Determinism.** A run is defined by `(config, seed, commit)`. The same
   triple must produce the same numbers. Anything that breaks this — an unseeded
   RNG, a set iteration order that reaches output, a wall-clock timestamp used
   in a decision — is a defect of the same severity as a crash.
3. **No network access at inference or test time.** Models and datasets are
   fetched by explicit `scripts/fetch_*.py` invocations into `assets/`, which is
   gitignored. Tests that need a model use the fixtures in `tests/fixtures/`.
4. **Config as code, never as argument soup.** Every experiment is fully
   described by a YAML file under `configs/`. If you find yourself adding a CLI
   flag that changes scientific behaviour, add a config field instead.
5. **Metrics are computed in one place.** `carma.metrics` owns every number that
   appears in a paper. No module computes its own success rate.
6. **Authorship.** Commits, file headers and documentation carry the project
   authors' names and nothing else. No tooling attribution lines anywhere in
   this repository.

---

## 3. Module boundaries

The dependency graph is acyclic and enforced in CI by
`tests/test_architecture.py`. Arrows point in the only direction imports are
allowed to travel.

```
              types  (depends on nothing but stdlib + numpy)
                ^
    +-----------+-----------+-----------+-----------+
    |           |           |           |           |
perception   memory      backbone    operator     sim
    ^           ^           ^           ^           ^
    +-----------+-----+-----+-----------+-----------+
                      |
                   arbiter
                      ^
                   runtime
                      ^
                metrics / cli
```

| Package | Owns | Must not import |
|---|---|---|
| `carma.types` | Dataclasses and Protocols shared by everything | any other `carma.*` |
| `carma.perception` | OpenCV feature extraction, traversability cues, scene analyzers | `arbiter`, `runtime` |
| `carma.memory` | Entry store, embedding, retrieval, hygiene | `arbiter`, `runtime`, `backbone` |
| `carma.backbone` | VLA navigation policies and their uncertainty | `arbiter`, `runtime`, `memory` |
| `carma.operator` | Correction capture, query channel, response model | `arbiter`, `runtime` |
| `carma.sim` | Simulator adapters (Gazebo first) | everything except `types` |
| `carma.arbiter` | Cost model and the act/retrieve/ask decision | `runtime`, `cli` |
| `carma.runtime` | Episode loop, condition wiring, recording | `cli` |
| `carma.metrics` | Every reported number | `runtime`, `cli` |
| `carma.cli` | Entry points, manifest writing | — |

**The rule that matters most:** `memory` must never import `backbone`, and
`backbone` must never import `memory`. They are joined only by `arbiter`. This
is what keeps the baseline conditions honest — a no-memory baseline is obtained
by handing `runtime` a different arbiter, not by adding an `if` inside the
backbone.

---

## 4. Interfaces before implementations

Every replaceable component is a `typing.Protocol` in `carma.types`. When you
add a new backbone, store, embedder or cost model:

1. Confirm the existing Protocol already fits. If it does not, change the
   Protocol deliberately and update every implementation in the same commit.
   Never widen a Protocol with optional keyword arguments to smuggle in one
   implementation's needs.
2. Register the implementation in that package's `registry.py` under a short
   string key.
3. Reference it from config by that key. Construction happens only through
   `carma.config.build()`. No module instantiates a concrete class from another
   package by name.

This is what makes the ablation table cheap: a condition is a config file, not a
code path.

---

## 5. The five experimental conditions

These are fixed by the research design and live in `configs/condition/`.
Do not add a sixth without it corresponding to a claim in the paper.

| Key | Arbiter | Purpose |
|---|---|---|
| `no_memory` | `always_act` | Floor — backbone alone |
| `unconditional_retrieval` | `always_retrieve` | The published RAG-VLA pattern |
| `ask_only` | `uncertainty_ask` | The ask-for-help pattern, no memory |
| `carma` | `cost_aware` | The proposed condition |
| `carma_no_hygiene` | `cost_aware` + hygiene disabled | Isolates the drift contribution |

Every condition runs against the same backbone, the same routes and the same
seeds. If a change makes that untrue, it is not ready.

---

## 6. Reproducibility gates

A run is only valid if it produced a manifest. `carma.cli.run` writes
`runs/<run_id>/manifest.yaml` containing the config hash, the resolved config,
the git commit, whether the tree was dirty, the seed, the platform, and the
versions of every installed dependency. `carma.cli.report` refuses to aggregate
runs whose manifests disagree on config hash.

Enforced in CI:

- `ruff check` and `ruff format --check` — no unformatted code lands.
- `mypy --strict` over `src/` — no untyped public function.
- `pytest` including `test_determinism.py`, which runs a short episode twice
  with the same seed and asserts byte-identical step records.
- `test_architecture.py`, which parses imports and fails on a boundary
  violation.
- `test_manifest.py`, which fails if a manifest is missing any required field.
- A dependency check that fails on an unpinned requirement.

If you need to skip a gate to land a change, the change is wrong. The one
exception is a genuinely flaky external dependency, which must be quarantined
with a linked issue, never with a bare `# noqa` or `@pytest.mark.skip`.

---

## 7. Conventions

**Language and style.** Python 3.11+. `ruff` for lint and format, line length
100. Type hints on every public function; `from __future__ import annotations`
at the top of every module. Prefer dataclasses over dicts for anything that
crosses a module boundary.

**Naming.** Distances in metres, angles in radians, time in seconds, all as
`float`. Suffix any variable whose unit is not the default: `timeout_ms`,
`heading_deg`. Image arrays are `np.ndarray` in HWC, BGR, `uint8` — the OpenCV
convention — and any conversion to RGB happens at the boundary of the module
that needs it, named explicitly.

**Errors.** Raise `carma.types.CarmaError` subclasses. Never return `None` to
signal failure from a public function. Never catch bare `Exception` outside
`carma.cli`.

**Logging.** `structlog` to JSONL. One event per decision, carrying `run_id`,
`episode_id`, `step`, the arbitration choice and the costs that produced it.
The decision log is a research artefact, not debug output — treat its schema as
public and version it in `docs/data-schema.md`.

**Randomness.** Every stochastic component takes an explicit
`rng: np.random.Generator`. There are no module-level RNGs and no calls to
`random` or `np.random.*` free functions anywhere in `src/`.

**Tests.** `pytest`, no network, no GPU, under sixty seconds for the whole unit
suite. Simulator-dependent tests are marked `@pytest.mark.sim` and excluded from
the default run. A bug fix arrives with the test that would have caught it.

**Commits.** Conventional Commits (`feat:`, `fix:`, `refactor:`, `exp:`,
`docs:`). One logical change per commit. An experiment-affecting change says so
in the body and names the config hashes it invalidates.

---

## 8. How to add things

**A new backbone.** Implement `NavigationBackbone` in
`carma/backbone/<name>.py`, register it, add a config under `configs/backbone/`,
add a fixture-based test asserting output shape and uncertainty range. The
backbone must return calibrated-ish uncertainty in `[0, 1]`; if the underlying
model gives you logits, calibrate in the adapter and document how.

**A new metric.** Add it to the right family module in `carma/metrics/`, add it
to `MetricBundle`, extend `test_metrics.py` with a hand-computed expected value.
A metric with no hand-computed test is not trusted.

**A new scene analyzer.** Plant health, weeds, disease: anything that describes
the scene without choosing an action. A model exported to ONNX needs only an
`onnx_classifier` entry in `configs/analyzers/`; custom logic implements
`SceneAnalyzer` in `carma/perception/analyzers/<name>.py`. Analyzer configs sit
outside experiment configs, so an analyzer never changes a config hash. Full
guide in `docs/analyzers.md`.

**A new memory backend.** Implement `MemoryStore` in `carma/memory/stores/`.
It must support `invalidate` and must preserve insertion order for equal
scores, or determinism breaks.

**A cost model change.** This is the scientific core. It requires a docs update
in `docs/cost-model.md` explaining the units, a determinism test, and a note in
the commit body listing the config hashes whose results are now stale.

---

## 9. Definition of done

A change is done when all of the following hold:

- `make check` passes locally from a clean virtualenv.
- New public functions are typed and docstringed with their units.
- Anything that changes numbers names the invalidated config hashes.
- `docs/` reflects the new behaviour in the same commit.
- No new dependency without a licence check and a pin.
- No credential, no absolute path, no personal directory in the diff.
