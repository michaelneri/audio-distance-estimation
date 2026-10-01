# Few-shot calibration for sim-to-real transfer

A distance estimator trained on simulated acoustics transfers poorly to real
recordings. This study asks how few labelled real samples are enough to fix it by
**post-hoc calibration**, rescaling a frozen model's outputs without retraining.

Preprint: [arXiv:2609.29203](https://arxiv.org/abs/2609.29203) · under review

## Layout

```
*.py        scripts
results/    the published numbers (committed) and the per-sample dumps they
            are computed from (regenerable, gitignored)
```

## The pipeline

Three stages, in order. The CSVs in `results/` are the published numbers — run the
pipeline and diff against them.

```
          ┌─ eval_real_data.py ─────────┐
          │  (+ eval_qmul_0db.py,       │   zero-shot evaluation of frozen
          │     eval_real_train.py,     │   synthetic models on real corpora
          │     eval_qmul_train_0db.py) │
          └──────────────┬──────────────┘
                         │ writes
                         ▼
        real_data_results.csv, real_data_train_results.csv
              (per-sample dumps — regenerable, not committed)
                         │ read by
          ┌──────────────┴──────────────┐
          ▼                             ▼
  run_calibration_variants.py    calib_significance.py
          │                             │
          ▼                             ▼
  calib_variants_results.csv     calib_significance.csv
         ✓ committed                 ✓ committed
```

`eval_real_data.py` also writes `real_data_summary.csv` directly.

### 1. Evaluate frozen synthetic models on real data

| script | covers |
|---|---|
| `eval_real_data.py` | test splits of VoiceHome-2, STARSS23, QMULTIMIT → `real_data_results.csv`, `real_data_summary.csv` |
| `eval_real_train.py` | training splits, which form the calibration pool → `real_data_train_results.csv` |
| `eval_qmul_0db.py` | QMULTIMIT test at 0 dB SNR, appended to the test results |
| `eval_qmul_train_0db.py` | QMULTIMIT training split at 0 dB, appended to the train results |

The train-split evaluations matter as much as the test ones: calibration is fitted on
samples a deployed system could actually label, so the pool comes from training data,
and correlations are measured there rather than on the test split.

### 2. Fit and compare calibration maps

`run_calibration_variants.py` evaluates every map in the paper, over 500 draws per
configuration.

It also validates the decision rule `ρ² > 1/(N+2)` for choosing between the affine and
constant maps.

### 3. Significance

`calib_significance.py` → `calib_significance.csv`: paired tests across corpora,
models and `N`.

## Models

- `model.py` — the baseline network, kept so the frozen checkpoints load as trained
- `model_robust.py` — `SeldNetMultiTask` and `RobustTrainer`, the multi-task heads and
  augmentation used for the "full stack" condition
- `QMULTIMIT.py`, `STARS23.py`, `VoiceHome.py` — dataset loaders
- `VoiceHome2_splitted.npz` — the VoiceHome-2 split definition

## Running it

```bash
pip install -e .
export SPEAKER_DISTANCE_DATA=/path/to/real/corpora    # QMULTIMIT, VoiceHome2
export SPEAKER_DISTANCE_NOISE=/path/to/noise          # WHAM!-derived splits
python -m speaker_distance.paths                      # confirm what resolved

python papers/calibration/eval_real_data.py
python papers/calibration/eval_real_train.py
python papers/calibration/eval_qmul_0db.py
python papers/calibration/eval_qmul_train_0db.py
python papers/calibration/run_calibration_variants.py
python papers/calibration/calib_significance.py
```

STARSS23 ships with the repository's Zenodo record; QMULTIMIT and VoiceHome-2 must be
obtained separately. See the [root README](../../README.md#datasets).


## Noise-robustness tables

`aggregate_noisy_results.py` collapses the per-SNR sweep outputs into one row per
(model, regime, RIR variant, SNR, fold) — 200 rows per variant, from 5 models × 2
training regimes × 4 SNRs × 5 folds:

```bash
python papers/calibration/aggregate_noisy_results.py --runs /path/to/runs_noisy --rir full
python papers/calibration/aggregate_noisy_results.py --runs /path/to/runs_noisy --rir no_early
```

It reads `runs_noisy/model=…_regime=…_rir=…_test=…_val=…/test_results_snr_*.csv` and
writes `results/noisy_summary.csv` and `results/noisy_summary_no_early.csv`.

`n_params` is counted from each run's checkpoint, excluding the fixed STFT analysis
filters and batch-norm running statistics — those are not trainable parameters and
including them inflates the figure by about 263,000.

The sweep itself comes from [`../rir-analysis/`](../rir-analysis/) plus the noise
splits; `--snrs all` includes the 5 dB, 30 dB and 40 dB runs the tables leave out.

## What was removed, and why

Exploratory branches whose outcomes are not in the paper: adaptive ridge selection by
leave-one-out CV, stratified-versus-random sample selection, L1-versus-OLS fitting, and
an earlier variant comparison that included a quadratic map. 
