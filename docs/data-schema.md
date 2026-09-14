# Data schema

The decision log is a research artefact, not debug output. Its schema is public
and versioned here; changing a field name is a breaking change and needs a
version bump.

## `runs/<run_id>/manifest.yaml`

| Field | Meaning |
|---|---|
| `run_id` | Unique identifier for this run |
| `created_at` | ISO-8601 UTC timestamp |
| `config_hash` | Hash over the resolved config, **excluding the seed** |
| `config` | The fully resolved config, defaults applied |
| `seed` | Master seed for the run |
| `git_commit` | Commit the code was at |
| `git_dirty` | True when the working tree had uncommitted changes |
| `python` | Interpreter version |
| `platform` | Host platform string |
| `dependencies` | Name to version for every installed distribution |

The seed is excluded from the hash deliberately: repeated seeds of one
configuration are exactly what should be pooled, while any other difference must
prevent pooling.

A run whose `git_dirty` is true is not reproducible and must not appear in a
paper.

## `runs/<run_id>/decisions.jsonl`

One JSON object per control step.

| Field | Type | Meaning |
|---|---|---|
| `run_id` | string | Matches the manifest |
| `episode_id` | string | Episode within the run |
| `step` | int | Zero-based step index |
| `choice` | `act` \| `retrieve` \| `ask` | What the arbiter selected |
| `costs` | object | Price of each option in seconds |
| `uncertainty` | float | Backbone uncertainty in `[0, 1]` |
| `timestamp` | string | ISO-8601 |

## `runs/<run_id>/metrics.yaml`

The serialised `MetricBundle`. Field names match the dataclass exactly; see
`carma/metrics/bundle.py`.
