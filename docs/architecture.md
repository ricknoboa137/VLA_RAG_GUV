# Architecture

The system is a layer around a vision-language-action navigation backbone. At
each control step it chooses between three options and records what the choice
cost.

## The loop

```
observe
  -> poll operator for volunteered corrections   (ingested into memory)
  -> backbone proposes an action + uncertainty
  -> arbiter prices ACT / RETRIEVE / ASK and picks the cheapest
       ACT      : use the proposal as-is
       RETRIEVE : query memory, drop stale hits, re-propose with context
       ASK      : query the operator, re-propose with their answer,
                  and spend the interaction on retiring one stale entry
  -> record the step
  -> actuate
```

The loop lives in `carma/runtime/episode.py` and is **identical across all five
experimental conditions**. Conditions differ only in which arbiter and which
hygiene policy `carma.config.build` constructed. If a change to the loop seems
to need an `if condition == ...` branch, the change belongs in an arbiter.

## Why memory and backbone never meet

`carma.memory` may not import `carma.backbone`, and vice versa. They are joined
only in `carma.arbiter`. This is not fastidiousness: it is what makes the
no-memory baseline honest. A baseline obtained by disabling a branch inside the
policy is a different system from the policy; a baseline obtained by handing the
runtime a different arbiter is the same system with one component swapped.

## Composition root

`carma.config.build` is the only place a concrete class is instantiated from a
registry key. Components that need collaborators expose a `bind` method that the
composition root calls — the retriever binds an embedder and a store, the
arbiter binds a cost model and a retriever. This keeps config files free of
wiring detail while keeping the wiring in one readable place.

## Where the science is

Almost all of it is in two files:

- `carma/arbiter/cost_model.py` — how the three options are priced.
- `carma/arbiter/policies.py` — how that pricing becomes a decision.

Everything else is infrastructure whose job is to make the numbers those two
files produce trustworthy.
