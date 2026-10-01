#!/usr/bin/env python
"""
train_val_test.py
=================
Single entry-point for training, validation, and testing the SeldNet
distance-estimation model across all RIR variants and cross-validation folds.

Workflow
--------
1. Edit CONFIG below – no command-line arguments needed.
2. Run:  python train_val_test.py

What it does
------------
* Auto-generates folds if the folds CSV is missing.
* Runs the "author schedule": for each fold i in [0, n_folds),
    test  = fold i
    val   = fold (i+1) % n_folds
    train = all remaining folds
  Each schedule is repeated for every variant, giving
  n_folds × n_variants independent training runs.
* Saves per-run test predictions and a sweep summary CSV.
* Prints a final table with mean MAE ± 95 % CI (t-distribution) per variant.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats as st
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data import LoaderConfig, SyntheticFoldDataModule, make_folds
from model import SeldTrainer

# Resolve paths from the repository root rather than the working directory, so this
# script runs from anywhere. See src/speaker_distance/paths.py.
import sys as _sys

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO / "src") not in _sys.path:
    _sys.path.insert(0, str(_REPO / "src"))
from speaker_distance.paths import SYNTHETIC_DIR  # noqa: E402


# ===========================================================================
# ██████  CONFIG  ─  edit this dict to configure your entire experiment
# ===========================================================================

CONFIG: Dict = {
    # ── Data ────────────────────────────────────────────────────────────────
    # Configuration to train on. "synthetic_both" removes both the timing
    # and the level cue; "synthetic_baseline" keeps both, as in WASPAA/TASLP.
    "configuration": "synthetic_both",
    "data_root":     str(SYNTHETIC_DIR / "synthetic_both"),   # holds the variant sub-folders
    "metadata_csv":  str(SYNTHETIC_DIR / "synthetic_both" / "metadata_variants.csv"),
    "folds_csv":     str(_HERE / "data" / "folds5.csv"),  # auto-generated if missing
    "auto_make_folds": True,                # create folds_csv from metadata_csv if absent

    # ── Variants ─────────────────────────────────────────────────────────────
    # RIR ablation variants: full RIR | direct path only |
    #                        no early reflections | no late reverberation
    "variants": ["full", "direct", "no_early", "no_late"],

    # ── Cross-validation ─────────────────────────────────────────────────────
    "n_folds": 5,                           # number of folds (0..n_folds-1)
    "fold_seed": 1337,                      # seed for fold assignment shuffle

    # ── Audio ────────────────────────────────────────────────────────────────
    "sample_rate":   16000,
    "clip_seconds":  10.0,

    # ── Gain augmentation (dB)  ───────────────────────────────────────────────
    # Set to 0.0 to disable. Applied as uniform random gain in [-gain_db, +gain_db].
    # test split always uses RMS normalisation regardless of this setting.
    "train_gain_db": 0.0,
    "test_gain_db":  0.0,
    "gain_prob":     1.0,                   # probability of applying gain augmentation

    # ── DataLoader ───────────────────────────────────────────────────────────
    "batch_size":   16,
    "num_workers":   4,

    # ── Model architecture ───────────────────────────────────────────────────
    "lr":           1e-3,
    "kernels":      "freq",                 # "freq" | "time" | "square"
    "n_grus":       2,                      # 0 | 1 | 2
    "features_set": "all",                  # "stft" | "sincos" | "all"
    "att_conf":     "onAll",              # "Nothing" | "onSpec" | "onAll"

    # ── Trainer ──────────────────────────────────────────────────────────────
    "max_epochs":  50,
    "accelerator": "auto",
    "devices":     "auto",
    "precision":   "32",
    "seed":        1337,

    # ── Weights & Biases ─────────────────────────────────────────────────────
    "project":  "Distance-Estimation-RIR-Analysis-NewVersion",
    "entity":   None,                       # W&B entity (username / team), or None
    "group":    "synthetic-rir-ablations",
    "tags":     "Clean,Synt,VaryingSoundSourceVolume",
    "offline":  True,                      # True = no W&B upload

    # ── Output paths ─────────────────────────────────────────────────────────
    "run_dir":      "runs",                 # base dir for per-run checkpoints & CSVs
    "summary_csv":  str(_HERE / "results" / "sweep_summary.csv"),    # one row per (variant × fold)
    "results_csv":  str(_HERE / "results" / "variant_results.csv"),  # aggregated CI per variant
}

# ===========================================================================


# ---------------------------------------------------------------------------
# Fold schedule helpers
# ---------------------------------------------------------------------------

def author_schedule(n_folds: int, i: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Rotating validation / test schedule used throughout the paper.

    For fold index *i*:
      test  = [i]
      val   = [(i+1) % n_folds]
      train = all other folds
    """
    val_folds  = [i]
    test_folds = [(i + 1) % n_folds]
    train_folds = [f for f in range(n_folds) if f not in val_folds + test_folds]
    return train_folds, val_folds, test_folds


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_one(
    cfg: Dict,
    variant: str,
    train_folds: List[int],
    val_folds: List[int],
    test_folds: List[int],
) -> Dict:
    """
    Train, validate, and test for one (variant, fold-split) combination.

    Returns a summary dict that is later collected into sweep_summary.csv.
    """
    pl.seed_everything(cfg["seed"], workers=True)

    gain_mismatch_db = abs(cfg["train_gain_db"] - cfg["test_gain_db"])
    gain_mismatch    = gain_mismatch_db > 1e-6

    run_name = (
        f"variant={variant}_"
        f"test={','.join(map(str, test_folds))}_"
        f"val={','.join(map(str, val_folds))}_"
        f"tgain={cfg['train_gain_db']}_"
        f"egain={cfg['test_gain_db']}"
    )
    save_dir = os.path.join(cfg["run_dir"], run_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── DataModule ───────────────────────────────────────────────────────────
    loader_cfg = LoaderConfig(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    dm = SyntheticFoldDataModule(
        data_root    = cfg["data_root"],
        metadata_csv = cfg["metadata_csv"],
        folds_csv    = cfg["folds_csv"],
        variant      = variant,
        train_folds  = train_folds,
        val_folds    = val_folds,
        test_folds   = test_folds,
        sample_rate  = cfg["sample_rate"],
        clip_seconds = cfg["clip_seconds"],
        loader       = loader_cfg,
        train_gain_db  = cfg["train_gain_db"],
        train_gain_prob= cfg["gain_prob"],
        test_gain_db   = cfg["test_gain_db"],
        test_gain_prob = cfg["gain_prob"],
        seed           = cfg["seed"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SeldTrainer(
        lr           = cfg["lr"],
        kernels      = cfg["kernels"],
        n_grus       = cfg["n_grus"],
        features_set = cfg["features_set"],
        att_conf     = cfg["att_conf"],
    )

    # ── W&B logger ───────────────────────────────────────────────────────────
    tags = [t for t in cfg["tags"].split(",") if t]
    tags += [
        f"variant:{variant}",
        f"test:{test_folds}",
        f"val:{val_folds}",
        "gain_mismatch" if gain_mismatch else "gain_matched",
        "RMS_normalized",
    ]
    logger = WandbLogger(
        project  = cfg["project"],
        entity   = cfg["entity"],
        group    = cfg["group"],
        name     = run_name,
        save_dir = save_dir,
        tags     = tags,
        offline  = cfg["offline"],
        log_model= False,
    )
    logger.experiment.config.update(
        {
            **{k: v for k, v in cfg.items() if not isinstance(v, dict)},
            "variant":          variant,
            "train_folds":      train_folds,
            "val_folds":        val_folds,
            "test_folds":       test_folds,
            "gain_mismatch":    gain_mismatch,
            "gain_mismatch_db": gain_mismatch_db,
            "loader":           asdict(loader_cfg),
        },
        allow_val_change=True,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt = ModelCheckpoint(
        dirpath   = os.path.join(save_dir, "checkpoints"),
        filename  = "epoch{epoch:03d}-val_mae{val/mae:.4f}",
        monitor   = "val/mae",
        mode      = "min",
        save_top_k= 1,
        save_last = True,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        default_root_dir = save_dir,
        max_epochs       = cfg["max_epochs"],
        accelerator      = cfg["accelerator"],
        devices          = cfg["devices"],
        precision        = cfg["precision"],
        logger           = logger,
        callbacks        = [ckpt, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps= 50,
        deterministic    = True,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)

    # ── Test (best checkpoint) ───────────────────────────────────────────────
    dm.setup("test")
    best_ckpt = ckpt.best_model_path or "last"
    trainer.test(model=None, datamodule=dm, ckpt_path=best_ckpt)

    # ── Save per-example predictions ─────────────────────────────────────────
    tested_model = trainer.lightning_module
    rows = getattr(tested_model, "all_test_results", [])
    df   = pd.DataFrame(rows)

    test_csv = os.path.join(save_dir, "test_results.csv")
    df.to_csv(test_csv, index=False)

    # ── Compute metrics from saved CSV ────────────────────────────────────────
    mae  = float(np.mean(np.abs(df["Pred"] - df["GT"]))) if len(df) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((df["Pred"] - df["GT"]) ** 2))) if len(df) > 0 else float("nan")
    rel_mae = float(np.mean(np.abs(df["Pred"] - df["GT"]) / df["GT"].clip(lower=1e-6))) if len(df) > 0 else float("nan")

    logger.log_metrics({"test/mae_csv": mae, "test/rmse_csv": rmse})
    logger.experiment.summary["test_csv"] = test_csv
    logger.experiment.finish()

    print(
        f"\n[run_one] {run_name}\n"
        f"  MAE={mae:.4f}  RMSE={rmse:.4f}  relMAE={rel_mae:.4f}  n={len(df)}\n"
        f"  Predictions → {test_csv}\n"
    )

    return {
        "variant":          variant,
        "train_folds":      ",".join(map(str, train_folds)),
        "val_folds":        ",".join(map(str, val_folds)),
        "test_folds":       ",".join(map(str, test_folds)),
        "train_gain_db":    cfg["train_gain_db"],
        "test_gain_db":     cfg["test_gain_db"],
        "gain_mismatch":    gain_mismatch,
        "gain_mismatch_db": gain_mismatch_db,
        "test_csv":         test_csv,
        "mae":              mae,
        "rmse":             rmse,
        "rel_mae":          rel_mae,
        "n":                len(df),
    }


# ---------------------------------------------------------------------------
# Confidence-interval reporting
# ---------------------------------------------------------------------------

def compute_variant_ci(
    summary: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each variant, aggregate per-fold MAE values and compute:
      - mean MAE
      - std MAE
      - 95 % confidence interval (t-distribution, two-tailed)

    With 5 folds the t critical value is t_{0.025, 4} ≈ 2.776.

    Parameters
    ----------
    summary : DataFrame with columns ``variant`` and ``mae`` (one row per fold)
    alpha   : significance level (default 0.05 → 95 % CI)

    Returns
    -------
    DataFrame with one row per variant containing aggregated statistics.
    """
    records = []
    for variant, grp in summary.groupby("variant"):
        values = grp["mae"].dropna().to_numpy()
        n      = len(values)
        mean   = values.mean()
        std    = values.std(ddof=1) if n > 1 else float("nan")

        if n > 1:
            # t-based confidence interval
            t_crit = st.t.ppf(1 - alpha / 2, df=n - 1)
            half_width = t_crit * std / np.sqrt(n)
        else:
            half_width = float("nan")

        records.append({
            "variant":    variant,
            "n_folds":    n,
            "mae_mean":   round(mean, 4),
            "mae_std":    round(std, 4) if not np.isnan(std) else float("nan"),
            "ci_lower":   round(mean - half_width, 4) if not np.isnan(half_width) else float("nan"),
            "ci_upper":   round(mean + half_width, 4) if not np.isnan(half_width) else float("nan"),
            "ci_half_width": round(half_width, 4) if not np.isnan(half_width) else float("nan"),
            "conf_level": f"{int((1-alpha)*100)}%",
            # Also report relative-MAE summary
            "rel_mae_mean": round(grp["rel_mae"].dropna().mean(), 4),
            "rmse_mean":    round(grp["rmse"].dropna().mean(), 4),
        })

    return pd.DataFrame(records).sort_values("mae_mean").reset_index(drop=True)


def print_ci_table(ci_df: pd.DataFrame) -> None:
    """Pretty-print the confidence interval table."""
    width = 78
    print("\n" + "═" * width)
    print("  Per-Variant Performance  (MAE, metres)")
    print("═" * width)
    print(
        f"  {'Variant':<12}  {'Folds':>5}  "
        f"{'MAE mean':>9}  {'MAE std':>8}  "
        f"{'CI low':>8}  {'CI high':>8}  "
        f"{'relMAE':>7}  {'RMSE':>7}"
    )
    print("─" * width)
    for _, row in ci_df.iterrows():
        print(
            f"  {row['variant']:<12}  {int(row['n_folds']):>5}  "
            f"{row['mae_mean']:>9.4f}  {row['mae_std']:>8.4f}  "
            f"{row['ci_lower']:>8.4f}  {row['ci_upper']:>8.4f}  "
            f"{row['rel_mae_mean']:>7.4f}  {row['rmse_mean']:>7.4f}"
        )
    print("═" * width)
    print(
        f"  CI computed with t-distribution at {ci_df['conf_level'].iloc[0]} confidence.\n"
        f"  Columns: MAE mean ± CI half-width = [ci_lower, ci_upper]\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG

    # ── Auto-generate folds if requested and file is missing ─────────────────
    folds_csv = Path(cfg["folds_csv"])
    if not folds_csv.exists():
        if cfg.get("auto_make_folds", False):
            print(f"[main] {folds_csv} not found – generating from {cfg['metadata_csv']} …")
            make_folds(
                metadata_csv = cfg["metadata_csv"],
                out_csv      = str(folds_csv),
                n_folds      = cfg["n_folds"],
                seed         = cfg["fold_seed"],
            )
        else:
            raise FileNotFoundError(
                f"{folds_csv} not found. Either create it or set auto_make_folds=True in CONFIG."
            )

    variants = cfg["variants"]
    n_folds  = cfg["n_folds"]

    # ── Run all (fold, variant) combinations ──────────────────────────────────
    results: List[Dict] = []
    total_runs = n_folds * len(variants)
    run_idx    = 0

    for i in range(n_folds):
        train_folds, val_folds, test_folds = author_schedule(n_folds, i)
        for variant in variants:
            run_idx += 1
            print(
                f"\n{'='*70}\n"
                f"  Run {run_idx}/{total_runs}  │  variant={variant}  │  "
                f"train={train_folds}  val={val_folds}  test={test_folds}\n"
                f"{'='*70}"
            )
            result = run_one(cfg, variant, train_folds, val_folds, test_folds)
            results.append(result)

            # Checkpoint: save intermediate summary after every run
            Path(cfg["summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(cfg["summary_csv"], index=False)

    # ── Final sweep summary ───────────────────────────────────────────────────
    summary = pd.DataFrame(results)
    summary.to_csv(cfg["summary_csv"], index=False)
    print(f"\n[main] Sweep summary written → {cfg['summary_csv']}")

    # ── Confidence-interval table per variant ─────────────────────────────────
    ci_df = compute_variant_ci(summary)
    ci_df.to_csv(cfg["results_csv"], index=False)
    print(f"[main] Per-variant CI table written → {cfg['results_csv']}\n")

    print_ci_table(ci_df)


if __name__ == "__main__":
    main()
