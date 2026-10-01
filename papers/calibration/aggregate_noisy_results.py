#!/usr/bin/env python
"""Aggregate the per-SNR sweep outputs into noisy_summary.csv.

The noise-robustness sweep writes one per-sample CSV per (run, test SNR):

    <runs>/model=<m>_regime=<r>_rir=<v>_test=<i>_val=<j>/test_results_snr_<snr>.csv

This collapses them into one row per (model, regime, rir_variant, test_snr, fold),
which is the table the paper reports from.

    python aggregate_noisy_results.py --runs /path/to/runs_noisy
    python aggregate_noisy_results.py --runs ... --rir no_early

Per-sample columns consumed: ``GT``, ``Pred``, ``rL1``, and, when the model has
auxiliary heads, ``log_{t60,mt,vol}_{gt,pred}``. Rows for models without those heads
leave the corresponding ``mae_log_*`` columns empty, matching the published tables.

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

N_FOLDS = 5

#: The SNRs the published tables report. The sweep also writes 5 dB, 30 dB and 40 dB.
DEFAULT_SNRS = ("clean", "0dB", "10dB", "20dB")

RUN_DIR_RE = re.compile(
    r"^model=(?P<model>.+?)"
    r"_regime=(?P<regime>.+?)"
    r"_rir=(?P<rir>.+?)"
    r"_test=(?P<test>\d+)"
    r"_val=(?P<val>\d+)$"
)
SNR_RE = re.compile(r"^test_results_snr_(?P<snr>.+)\.csv$")

AUX = ("t60", "mt", "vol")


#: Checkpoint entries that are not trainable parameters. The STFT front-end holds
#: fixed analysis filters (~263k values) and batch-norm carries running statistics;
#: counting either would inflate the figure well beyond the model's actual size.
_NON_PARAM = ("stft.", "STFT.", "running_mean", "running_var", "num_batches_tracked")


def count_params(run_dir: Path) -> int | None:
    """Trainable parameter count, read from any checkpoint in the run directory."""
    ckpts = sorted(run_dir.glob("checkpoints/*.ckpt")) or sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    try:
        import torch

        state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        return int(
            sum(
                v.numel()
                for k, v in state.items()
                if hasattr(v, "numel") and not any(marker in k for marker in _NON_PARAM)
            )
        )
    except Exception:  # noqa: BLE001 - a missing or unreadable checkpoint is not fatal
        return None


def summarise(csv_path: Path) -> dict | None:
    df = pd.read_csv(csv_path)
    if df.empty or not {"GT", "Pred"} <= set(df.columns):
        return None

    err = df["Pred"].to_numpy(float) - df["GT"].to_numpy(float)
    row: dict = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }

    for name in AUX:
        gt, pred = f"log_{name}_gt", f"log_{name}_pred"
        if gt in df.columns and pred in df.columns:
            diff = (df[pred].to_numpy(float) - df[gt].to_numpy(float))
            value = float(np.mean(np.abs(diff)))
            row[f"mae_log_{name}"] = "" if np.isnan(value) else value
        else:
            row[f"mae_log_{name}"] = ""

    if "rL1" in df.columns:
        row["rel_mae"] = float(df["rL1"].mean())
    else:
        gt = df["GT"].to_numpy(float)
        row["rel_mae"] = float(np.mean(np.abs(err) / np.maximum(np.abs(gt), 1e-6)))

    row["n"] = int(len(df))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", type=Path, required=True,
        help="directory holding the model=..._regime=... run folders",
    )
    parser.add_argument(
        "--rir", default="full",
        help="RIR variant to summarise (one file per variant)",
    )
    parser.add_argument(
        "--snrs", nargs="*", default=list(DEFAULT_SNRS),
        help="test SNRs to include; pass 'all' for every SNR found",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output CSV (default: results/noisy_summary[_<rir>].csv)",
    )
    args = parser.parse_args()

    if not args.runs.is_dir():
        raise SystemExit(f"--runs is not a directory: {args.runs}")

    take_all = len(args.snrs) == 1 and args.snrs[0] == "all"
    rows: list[dict] = []
    skipped_variants: set[str] = set()

    for run_dir in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue
        meta = match.groupdict()
        if meta["rir"] != args.rir:
            skipped_variants.add(meta["rir"])
            continue

        test_fold, val_fold = int(meta["test"]), int(meta["val"])
        train_folds = [f for f in range(N_FOLDS) if f not in (test_fold, val_fold)]
        n_params = count_params(run_dir)

        for csv_path in sorted(run_dir.glob("test_results_snr_*.csv")):
            snr_match = SNR_RE.match(csv_path.name)
            if not snr_match:
                continue
            snr = snr_match.group("snr")
            if not take_all and snr not in args.snrs:
                continue

            stats = summarise(csv_path)
            if stats is None:
                print(f"  [skip] empty or unexpected schema: {csv_path}")
                continue

            rows.append(
                {
                    "model": meta["model"],
                    "regime": meta["regime"],
                    "rir_variant": meta["rir"],
                    "test_snr": snr,
                    "n_params": "" if n_params is None else n_params,
                    "train_folds": ",".join(str(f) for f in train_folds),
                    "val_folds": val_fold,
                    "test_folds": test_fold,
                    # Recorded relative to the repository root, as in the published tables.
                    "test_csv": f"runs_noisy/{run_dir.name}/{csv_path.name}",
                    **stats,
                }
            )

    if not rows:
        raise SystemExit(
            f"No runs matched rir={args.rir!r} under {args.runs}."
            + (f" Variants present: {sorted(skipped_variants)}" if skipped_variants else "")
        )

    columns = [
        "model", "regime", "rir_variant", "test_snr", "n_params",
        "train_folds", "val_folds", "test_folds", "test_csv",
        "mae", "rmse", "mae_log_t60", "mae_log_mt", "mae_log_vol", "rel_mae", "n",
    ]
    # Order by condition, then by SNR as the tables present it.
    snr_order = {snr: i for i, snr in enumerate(DEFAULT_SNRS)}
    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values(
        by=["model", "regime", "test_folds", "test_snr"],
        key=lambda col: col.map(snr_order) if col.name == "test_snr" else col,
        kind="stable",
    ).reset_index(drop=True)

    out = args.out
    if out is None:
        suffix = "" if args.rir == "full" else f"_{args.rir}"
        out = RESULTS / f"noisy_summary{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{len(df)} rows -> {out}")
    print(
        f"  {df['model'].nunique()} models x {df['regime'].nunique()} regimes x "
        f"{df['test_snr'].nunique()} SNRs x {df['test_folds'].nunique()} folds"
    )
    if skipped_variants:
        print(f"  other RIR variants present: {sorted(skipped_variants)}")


if __name__ == "__main__":
    main()
