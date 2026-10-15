# Configs

An experiment is a config file. Nothing that changes scientific behaviour is a
command-line flag.

`condition/` holds the five experimental conditions fixed by the research
design. `experiment/` holds complete experiments: a condition plus every
component block (backbone, embedder, store, retriever, hygiene, cost model,
arbiter), the episode count and a seed. `analyzers/` lists the scene analyzers
the live vision pipeline runs; it is not part of any experiment, so editing it
never changes a config hash.

Run `carma components` to list every registered key you may name here.
