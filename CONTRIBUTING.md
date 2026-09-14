# Contributing

`AGENTS.md` is the development contract. Read it before your first change; it
describes module boundaries, the reproducibility gates and the definition of
done, and CI enforces most of it.

## Workflow

1. Branch from `main`. Name it `<type>/<short-slug>`, matching the Conventional
   Commit type you intend to use.
2. Make one logical change.
3. `make check` must pass from a clean virtualenv before you open a PR.
4. If your change alters any reported number, say so in the PR description and
   list the config hashes it invalidates.

## Review

A reviewer checks, in this order: does it violate a module boundary, does it
break determinism, does it change numbers without saying so, is it typed and
tested. Style is handled by `ruff` and is not review material.

## Reporting a bug in a result

Open an issue with the `run_id`, the manifest, and the seed. A result bug
without a manifest cannot be investigated and will be closed asking for one.
