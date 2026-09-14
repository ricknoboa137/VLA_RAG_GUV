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

## Operator utterance (`carma/operator/utterance`, `/carma/operator/utterance`)

One JSON object per transcribed operator utterance, published by
`carma_voice/stt_node`.

| Field | Type | Meaning |
|---|---|---|
| `stamp` | float | Unix time the clip was received, seconds |
| `operator_id` | string | Who spoke; becomes memory provenance |
| `text` | string | Transcription, whitespace-trimmed; kept even when rejected |
| `intent` | string | `stop` \| `go` \| `turn_left` \| `turn_right` \| `yes` \| `no` \| `correction` \| `none` |
| `rejected` | string \| null | `no_speech` \| `low_confidence` when the transcription was not trusted (intent is then `none`); null otherwise |
| `language` | string | ISO 639-1 code, chosen among the configured operator languages or forced |
| `language_probability` | float | Whisper's probability for that language in `[0, 1]`; 1.0 when forced |
| `audio_s` | float | Duration of the operator's clip, seconds |
| `stt_latency_s` | float | Clip receipt to published text, seconds |
| `avg_logprob` | float \| null | Mean segment log-probability; null when nothing was recognised |
| `no_speech_prob` | float \| null | Highest segment no-speech probability |

## `runs/<run_id>/metrics.yaml`

The serialised `MetricBundle`. Field names match the dataclass exactly; see
`carma/metrics/bundle.py`.
