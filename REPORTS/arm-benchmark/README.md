# SmolVLA against X-VLA on the SO-101 arm

Open-loop benchmark of two open vision-language-action models, fine-tuned with
one recipe on the same SO-101 episodes and scored on 18 episodes neither saw.
The report is `CARMA-Arm-Benchmark-Report-RevA.docx`.

| File | What it is |
|---|---|
| `CARMA-Arm-Benchmark-Report-RevA.docx` | The report: method, results, figures, limits |
| `docx-source.txt` | The report's text source, one line-prefixed paragraph per line |
| `make_figures.py` | Draws `figures/` from `data/` |
| `figures/` | The five figures in the report |
| `data/` | Every input the figures and numbers come from, with `SHA256SUMS` |

## Result in one paragraph

On the 18 test episodes SmolVLA's mean joint error over a 25-step chunk is
0.0805 rad against X-VLA's 0.0941 rad. The paired per-episode difference is
-0.0129 rad (95% bootstrap interval -0.0201 to -0.0057). This is imitation of
recorded demonstrations, not task success, and X-VLA trained with its VLM
frozen because a full fine-tune does not fit in 8 GB, so it is not a general
ranking of the two architectures. The report lists what the numbers do and
do not show.

## Redo the figures

```bash
python make_figures.py        # needs matplotlib and numpy; reads ./data
```

Check the data is the recorded data:

```bash
cd data && sha256sum -c SHA256SUMS
```

`data/` holds the same files as `training/results/2026-10-22_arm-benchmark/`
(the split also as `training/splits/so101_red_cube_seed0.json`); that folder
and the dated entry in `training/results/RESULTS.md` are the record, and this
copy exists so the report can be rebuilt on its own.

| File in `data/` | Contents |
|---|---|
| `split.json` | 146 train, 18 calibration, 18 test episodes, from `np.random.default_rng(0).permutation(182)` |
| `test_predictions.jsonl` | One row per model and test frame: chunk error, per-joint error, uncertainty, spread, timings |
| `test_summary.json` | The numbers in the report's results table |
| `calibration_predictions.jsonl`, `calibration_summary.json` | The same on the 18 calibration episodes, used only to fit the uncertainty map |
| `uncertainty_fit.json` | The fitted logistic maps and the error tolerance they predict |
| `*_train_config.json`, `*_train.measure.json`, `*_train.curve.csv` | Each fine-tune's configuration, measured memory and speed, and loss every 100 steps |
| `env.json` | Package versions, GPU and driver, and the pinned revision of every asset |

## Reproduce the whole benchmark

Needs an NVIDIA GPU with 8 GB, the `carma-train` environment
(`training/README.md`) and about 38 hours of GPU time. Every asset is
Apache-2.0, ungated and pinned to a commit in
`scripts/fetch_training_assets.py`; after the fetch nothing uses the network.

| Asset (name for the fetch script) | Source and revision |
|---|---|
| `smolvla_base` | `lerobot/smolvla_base` @`5e8d12a` |
| `xvla_base` | `lerobot/xvla-base` @`cdb7964` |
| `smolvlm2-500m-video-instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @`7b375e1` |
| `bart-large-tokenizer` | `facebook/bart-large` @`cb48c13` |
| `so101_red_cube` | `whosricky/so101_pick_mixed_cubes_2cams` @`f77f316` |

From the repository root, in the `carma-train` environment with
`PYTHONPATH=src` and the offline variables of `training/README.md`:

```bash
for a in smolvla_base xvla_base smolvlm2-500m-video-instruct bart-large-tokenizer so101_red_cube; do
  python scripts/fetch_training_assets.py $a
done
python training/split_episodes.py --dataset assets/so101_red_cube --seed 0 \
  --out training/splits/so101_red_cube_seed0.json
```

The two fine-tunes (seed 0, 20 000 steps, batch 8, 146 training episodes) take
about 16 and 21 hours. The exact commands, with every flag, are in the dated
entry of `training/results/RESULTS.md`; the configurations they produced are
`data/*_train_config.json`. Then, per that entry: serve both checkpoints with
`scripts/policy_server.py --policy smolvla|xvla --arm-spec
configs/arm_policy/<model>_so101_red_cube_seed0.yaml --self-check`, score the
calibration episodes with `scripts/benchmark_arm_policies.py`, fit the
uncertainty maps with `scripts/fit_arm_uncertainty.py`, write the fit into the
two specs, and score the test episodes.

Latencies depend on the GPU being shared and will differ. The errors should
match closely but not bit for bit: training on a GPU is not exactly
repeatable.

## Rebuild the Word file

`docx-source.txt` is plain text: `T` title, `S` subtitle, `#` headings, `B`
bullets, `!` a warning box, `I path | caption` an image, and `|#` / `|` table
header and body rows. The `.docx` here was generated from it by a separate
OOXML builder that is not part of this repository; any tool that reads that
markup, or copying the text into a word processor with the figures from
`figures/`, gives the same report.
