# The cost model

## The currency

All three options are priced in **expected additional traverse time, in
seconds**. Choosing one unit is what makes operator attention, retrieval
latency and the risk of acting wrongly comparable at all.

The alternative — an abstract utility with tuned weights — would let the method
be tuned into looking good, and would make the operator-burden claim
unfalsifiable. Seconds are measurable on the platform and arguable with an
agronomist.

## The three prices

Let `u` be the backbone's uncertainty in `[0, 1]` and `g` the retriever's
expected gain, normally its top similarity score. The risk that acting now
goes wrong is

```
r = max(u, rho * g)
```

and the three prices are

```
cost_act      = r * R
cost_retrieve = L + r * (1 - e_r * g) * R
cost_ask      = A + r * (1 - e_a) * R
```

| Symbol | Meaning | How it is obtained |
|---|---|---|
| `R` | Seconds lost to a recovery traverse after a wrong action | Measured on the platform |
| `L` | Wall-clock cost of one retrieval, embedding included | Measured, WP5 |
| `A` | Expected seconds to obtain an operator answer | Measured in the field campaign |
| `e_r` | Fraction of acting risk a good retrieval removes | Estimated on held-out data |
| `e_a` | Fraction of acting risk an operator answer removes | Estimated on held-out data |
| `rho` | Chance that a remembered warning at this place still marks a real risk (`memory_risk`) | Not yet estimated: 0.5 in `configs/experiment/hazard_study/` is a stated placeholder. 0 gives `r = u`, the original model |

## Three decisions worth defending

**`A` is larger than raw response latency.** Interrupting a person costs more
than the seconds they spend speaking: they must re-orient to the robot's
situation first. Pricing `A` at the stopwatch value would systematically
under-price asking and would flatter the method.

**The operator term is paid in full regardless of `u`.** A person is
interrupted whether or not the robot felt uncertain. Scaling `A` by uncertainty
would let a confident-but-wrong policy ask for free.

**`e_r` is never assumed to be 1.** Retrieval that returns something is not
retrieval that returns something useful. Tying the retrieve price to `g` is
what stops unconditional retrieval from looking free, and it is the reason the
arbiter runs a cheap probe before committing to a full retrieval.

**Memory is evidence of risk, not only a remedy for it (`rho`, 2026-11-14).**
The first version priced risk as `u` alone. A hidden hazard (`docs/hazards.md`)
is where a policy is confident and wrong, so `u` is near 0 there; acting was
then free and the arbiter never consulted memory, however strongly the probe
matched an operator's warning given at that exact place. With a perfect
context-following backbone, carma scored 0/8 against 7/8 for unconditional
retrieval. Letting the probe's gain raise the risk, `r = max(u, rho * g)`, gave
carma 7/8 with 19% fewer retrievals. `max` rather than a sum, so neither source
double-counts the other; scaled by `rho`, because an old warning can be wrong
(the patch dried up), which is what memory hygiene is for. With `rho = 0`
every earlier configuration prices exactly as before.

## Hygiene time base

`stage_decay` staleness is
`0.5 * min(stage_distance / 3, 1) + 0.2 * (1 - 0.5 ** (age_days / 30)) + 0.3 * d / (d + 25)`
(default weights; `d` in metres). Age is measured on the **run clock**:
simulated seconds since the start of the run, accumulated across its episodes.
Episode `i` starts at the sum of the durations (last step's `t`) of episodes
`0..i-1`, and the run clock at an observation is that offset plus `obs.t`.
`MemoryEntry.created_at` is the run clock at ingest; `age_days` is
`(run_t - created_at) / 86400`. The runtime passes `run_t` explicitly to
`Retriever.ingest` and to `HygienePolicy.staleness` / `select_for_review`;
nothing reads the wall clock, so `(config, seed, commit)` fixes every entry.

Stage distance stays the dominant term by design: crop stage, not elapsed
seconds, is what makes a correction wrong. A study run lasts minutes of
simulated time, so against a 30-day half-life the age term is near zero within
one run; it matters once a store persists over a season. Before this, ingest
stamped Unix epoch seconds while staleness subtracted from `obs.t`, so age was
always clamped to 0 and the age term never contributed.

## Changing this file

A change here invalidates every result produced under the old model. The commit
must say so and list the affected config hashes, per `AGENTS.md` section 8.
