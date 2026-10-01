# Which parts of the reverberation carry distance from single-channel recordings?

A single-channel distance estimator sees a reverberant signal. Does it use the
reverberation, or something simpler? This study removes one room-impulse-response
component at a time and measures what the estimator loses.

Published at **IWAENC 2026** · preprint [arXiv:2605.07694](https://arxiv.org/abs/2605.07694)

## The design

Four RIR variants, trained and evaluated independently under five-fold
cross-validation. Every variant shares the same rooms, talkers and distances, so the
only thing that changes is which part of the impulse response survives:

| variant | contents |
|---|---|
| `full` | the complete impulse response |
| `direct` | direct path only (2 ms window, 5 ms half-cosine fade) |
| `no_early` | direct path plus the late tail; early reflections removed |
| `no_late` | direct path plus early reflections; late tail removed |

The early/late boundary is the **mixing time**, derived per room from the mean free
path rather than fixed globally, so the split follows each room's geometry.

The corpus also crosses these with four shortcut configurations that remove the
propagation-delay and received-level cues — see the
[dataset section of the root README](../../README.md#datasets). The headline results use
`synthetic_both`, where neither cue is available, which is the realistic uncalibrated
case.

## Results

`results/variant_results.csv`, mean absolute error over five folds with 95 % intervals:

| variant | MAE (m) | 95 % CI | relative MAE |
|---|---|---|---|
| `full` | **1.29** | 1.17 – 1.41 | 0.29 |
| `no_late` | 1.39 | 1.28 – 1.50 | 0.31 |
| `direct` | 1.63 | 1.52 – 1.73 | 0.41 |
| `no_early` | 1.79 | 1.65 – 1.94 | 0.45 |

**Early reflections are the informative component.** Removing them (`no_early`, 1.79 m) is worse than keeping *only* the direct path (1.63 m) — the late tail alone is not merely uninformative, it is misleading. Keeping
early reflections without the tail (`no_late`, 1.39 m) nearly matches the full
response (1.29 m). `results/sweep_summary.csv` holds the per-fold rows (4 variants × 5 folds) behind that
table.

## Running it

```bash
pip install -e .
python -m speaker_distance.paths            # confirm the corpus resolved
python papers/rir-analysis/train_val_test.py
```

Configuration is the `CONFIG` dict at the top of `train_val_test.py` — data paths,
variants, folds, architecture and trainer settings in one place. Defaults reproduce the
published setting: `synthetic_both`, all four variants, five folds, `att_conf="onAll"`,
2 GRU layers, 50 epochs.

To train on a different configuration, change `configuration` and the two
paths beside it — `synthetic_baseline` gives the time-calibrated condition used in
[`../estimation-2023-2024/`](../estimation-2023-2024/).

W&B logging is off by default (`offline: True`).

### Regenerating the corpus

`generate_all_datasets.py` builds all four configurations in one pass, reusing one RIR
per scene so the variants stay exactly paired:

```bash
python papers/rir-analysis/generate_all_datasets.py /path/to/EARS --out /path/to/output
```

Needs the [EARS](https://sp-uhh.github.io/ears_dataset/) anechoic corpus, which is not
redistributable; pass it as an argument or set `SPEAKER_DISTANCE_EARS`. Takes 1–2 days
and about 26 GB. It refuses to overwrite an existing corpus unless `--overwrite` is
passed.


## This study feeds the next one

[`../calibration/`](../calibration/) takes the checkpoints trained here and asks what
minimal labelling makes them work on real recordings.

## Citation

```bibtex
@INPROCEEDINGS{Neri_IWAENC_2026,
  author={Neri, Michael and Politis, Archontis and Virtanen, Tuomas},
  booktitle={18th International Workshop on Acoustic Signal Enhancement (IWAENC)},
  title={{Dependence on Early and Late Reverberation of Single-Channel Speaker Distance Estimation}},
  year={2026},
  pages={1-5},
  doi={}
}
```
