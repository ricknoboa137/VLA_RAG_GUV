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
expected gain, normally its top similarity score.

```
cost_act      = u * R
cost_retrieve = L + u * (1 - e_r * g) * R
cost_ask      = A + u * (1 - e_a) * R
```

| Symbol | Meaning | How it is obtained |
|---|---|---|
| `R` | Seconds lost to a recovery traverse after a wrong action | Measured on the platform |
| `L` | Wall-clock cost of one retrieval, embedding included | Measured, WP5 |
| `A` | Expected seconds to obtain an operator answer | Measured in the field campaign |
| `e_r` | Fraction of acting risk a good retrieval removes | Estimated on held-out data |
| `e_a` | Fraction of acting risk an operator answer removes | Estimated on held-out data |

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

## Changing this file

A change here invalidates every result produced under the old model. The commit
must say so and list the affected config hashes, per `AGENTS.md` section 8.
