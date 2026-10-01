"""
eval_real_data.py — Zero-shot evaluation of synthetic-trained models on real/hybrid datasets.

Datasets
--------
  VoiceHome2  : real noisy recordings, distances from VoiceHome2_splitted.npz
  STARS23     : real recordings, filenames encode distance
  QMUL-TIMIT  : hybrid (real RIRs + TIMIT speech), tested clean and at 0 dB

Models evaluated (noise-trained only, zero-shot)
-------------------------------------------------
  iwaenc_repro : SeldNet baseline (no ablations)
  full_stack   : full multi-task + augmentation stack

Output
------
  real_data_results.csv   — per-sample predictions for all datasets/models/folds
  real_data_summary.csv   — mean MAE ± 95 % CI across the 5 fold checkpoints

Usage
-----
  conda run -n deeplearning python eval_real_data.py

Paths below follow the __main__ structure in the three dataloader scripts.
Edit DATASET_ROOT if your real datasets live elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats as sp_stats

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
# Resolve data locations from the repository root rather than this file's folder,
# so the script keeps working wherever it is moved to. See src/speaker_distance/paths.py.
import sys as _sys
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in _sys.path:
    _sys.path.insert(0, str(_REPO / "src"))
from speaker_distance.paths import DATASETS_DIR, NOISE_DIR, starss23_dir  # noqa: E402
sys.path.insert(0, str(BASE))

from model_robust import RobustTrainer
from VoiceHome  import VoiceHome2
from STARS23    import STARS23
from QMULTIMIT  import QMULLIMIT

# ---------------------------------------------------------------------------
# Paths — edit here if your layout differs
# ---------------------------------------------------------------------------
DATASET_ROOT = DATASETS_DIR
VH_NPZ        = BASE / "VoiceHome2_splitted.npz"
VH_AUDIO_DIR  = DATASET_ROOT / "VoiceHome2" / "audio" / "noisy"
VH_ANNOT_DIR  = DATASET_ROOT / "VoiceHome2" / "annotations" / "rooms"

STARS_DIR    = starss23_dir()

QMUL_DIR      = DATASET_ROOT / "QMULTIMIT"
WHAM_TEST_DIR = NOISE_DIR / "test_noise"   # optional; skipped if absent

RUNS_DIR      = BASE / "runs_noisy"
OUT_DIR       = BASE
BATCH_SIZE    = 16

# ---------------------------------------------------------------------------
# Which checkpoints to evaluate
# ---------------------------------------------------------------------------
MODELS   = ["iwaenc_repro", "full_stack"]
REGIMES  = ["clean_trained", "noise_trained"]
RIR      = "full"
N_FOLDS  = 5

def ckpt_path(model: str, regime: str, fold: int) -> Path:
    val = (fold + 1) % N_FOLDS
    d = RUNS_DIR / f"model={model}_regime={regime}_rir={RIR}_test={fold}_val={val}"
    return d / "checkpoints" / "last.ckpt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t_halfwidth(values: np.ndarray, alpha: float = 0.05) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return float(sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
                 * np.std(values, ddof=1) / np.sqrt(n))


@torch.no_grad()
def run_inference(model: RobustTrainer, loader: DataLoader,
                  device: torch.device) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        audio  = batch["audio"].to(device)
        labels = batch["label"].to(device)
        ids    = batch["id"]
        out    = model.model(audio)
        preds  = out["dist_pred"]
        for i in range(labels.shape[0]):
            gt   = float(labels[i].cpu())
            pred = float(preds[i].cpu())
            rows.append({
                "GT":   gt,
                "Pred": pred,
                "L1":   abs(pred - gt),
                "rL1":  abs(pred - gt) / max(gt, 1e-6),
                "ID":   ids[i] if isinstance(ids[i], str) else str(ids[i]),
            })
    return rows


def build_loaders() -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}

    # --- VoiceHome2 (test split only) ---
    if VH_NPZ.exists() and VH_AUDIO_DIR.exists():
        npz = np.load(str(VH_NPZ))
        ds  = VoiceHome2(
            path_annotations=str(VH_ANNOT_DIR),
            path_audios=str(VH_AUDIO_DIR),
            filenames=npz["arr_4"],
            distances=npz["arr_5"],
        )
        loaders["VoiceHome2"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    else:
        print(f"[SKIP] VoiceHome2 — {VH_NPZ} or {VH_AUDIO_DIR} not found")

    # --- STARS23 (test split) ---
    stars_test = STARS_DIR / "test"
    if stars_test.exists():
        ds = STARS23(path_audios=str(stars_test))
        loaders["STARS23"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    else:
        print(f"[SKIP] STARS23 — {stars_test} not found")

    # --- QMUL-TIMIT clean (test split) ---
    qmul_test = QMUL_DIR / "test"
    if qmul_test.exists():
        ds_clean = QMULLIMIT(path_audios=str(qmul_test))
        loaders["QMUL_clean"] = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)

        # QMUL_0dB is handled by eval_qmul_0db.py (pre-loads noise to avoid per-sample I/O)
        print("[SKIP] QMUL_0dB — use eval_qmul_0db.py for this condition")
    else:
        print(f"[SKIP] QMUL-TIMIT — {qmul_test} not found")

    return loaders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    loaders = build_loaders()
    if not loaders:
        print("No datasets found — check DATASET_ROOT paths.")
        return

    all_rows: list[dict] = []

    for model_name in MODELS:
        for regime in REGIMES:
            print(f"\n=== {model_name} / {regime} ===")
            for fold in range(N_FOLDS):
                ckpt = ckpt_path(model_name, regime, fold)
                if not ckpt.exists():
                    print(f"  [missing] {ckpt}")
                    continue
                model = RobustTrainer.load_from_checkpoint(str(ckpt), map_location=device)
                model.to(device).eval()
                print(f"  fold {fold} loaded", end="")

                for ds_name, loader in loaders.items():
                    rows = run_inference(model, loader, device)
                    for r in rows:
                        r.update({"model": model_name, "regime": regime,
                                  "fold": fold, "dataset": ds_name})
                    all_rows.extend(rows)
                    mae = np.mean([r["L1"] for r in rows])
                    print(f"  | {ds_name}: MAE={mae:.3f} m", end="")
                print()

    if not all_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS / "real_data_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-sample results → {out_csv}")

    # --- Summary: mean ± 95% CI across 5 folds ---
    summary_rows = []
    for (model_name, regime, ds_name), grp in df.groupby(["model", "regime", "dataset"]):
        fold_maes  = grp.groupby("fold")["L1"].mean().values
        fold_rmaes = grp.groupby("fold")["rL1"].mean().values
        summary_rows.append({
            "model":       model_name,
            "regime":      regime,
            "dataset":     ds_name,
            "n_folds":     len(fold_maes),
            "MAE_mean":    float(np.mean(fold_maes)),
            "MAE_ci":      t_halfwidth(fold_maes),
            "relMAE_mean": float(np.mean(fold_rmaes)),
            "relMAE_ci":   t_halfwidth(fold_rmaes),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "regime", "model"])
    out_sum = RESULTS / "real_data_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"Saved summary → {out_sum}\n")

    # Pretty-print
    print(f"{'Dataset':<20} {'Regime':<16} {'Model':<15} {'MAE (m)':<18} {'Rel. MAE'}")
    print("-" * 80)
    for _, r in summary.iterrows():
        mae_str  = f"{r['MAE_mean']:.2f} ± {r['MAE_ci']:.2f}"
        rmae_str = f"{r['relMAE_mean']:.3f} ± {r['relMAE_ci']:.3f}"
        print(f"{r['dataset']:<20} {r['regime']:<16} {r['model']:<15} {mae_str:<18} {rmae_str}")


if __name__ == "__main__":
    main()
