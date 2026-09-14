# Running experiments

## The rule

A run without a manifest is not a result. `carma run` always writes one; if you
produce numbers another way, they do not go in a paper.

## One condition, many seeds

```bash
for seed in 0 1 2 3 4; do
  carma run --config configs/experiment/sim_study.yaml --seed "$seed"
done
carma report runs/
```

`carma report` refuses to aggregate runs whose config hashes disagree. Since the
seed is excluded from the hash, seeds pool and configuration changes do not.

## All five conditions

Copy `configs/experiment/sim_study.yaml` once per condition, changing only
`condition`, `arbiter` and `hygiene` to match `configs/condition/<key>.yaml`.
Everything else must stay byte-identical, or the comparison is not controlled.

## Before you trust a number

- `git_dirty` false in every manifest.
- The same episode count and the same seeds across conditions.
- Success rate reported alongside every interaction metric. Queries per 100 m
  falling while success also falls is not an improvement, and reporting it
  alone would be misleading.
