# Configs

An experiment is a config file. Nothing that changes scientific behaviour is a
command-line flag.

`condition/` holds the five experimental conditions fixed by the research
design. `experiment/` composes a condition with an episode source and a seed.
`backbone/`, `embedder/` and the rest hold reusable component blocks you can
copy into an experiment file.

Run `carma components` to list every registered key you may name here.
