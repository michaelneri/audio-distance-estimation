"""
eval_real_train.py
Run inference on the TRAINING splits of each real dataset.
Output: real_data_train_results.csv  (same schema as real_data_results.csv)
Used as the calibration pool in few_shot_calibration.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
from VoiceHome   import VoiceHome2
from STARS23     import STARS23
from QMULTIMIT   import QMULLIMIT

DATASET_ROOT = DATASETS_DIR
VH_NPZ       = BASE / "VoiceHome2_splitted.npz"
VH_AUDIO_DIR = DATASET_ROOT / "VoiceHome2" / "audio" / "noisy"
VH_ANNOT_DIR = DATASET_ROOT / "VoiceHome2" / "annotations" / "rooms"
STARS_DIR    = starss23_dir()
QMUL_DIR     = DATASET_ROOT / "QMULTIMIT"
RUNS_DIR     = BASE / "runs_noisy"
BATCH_SIZE   = 16
MODELS       = ["iwaenc_repro", "full_stack"]
REGIMES      = ["clean_trained", "noise_trained"]
RIR          = "full"
N_FOLDS      = 5


def ckpt_path(model, regime, fold):
    val = (fold + 1) % N_FOLDS
    d = RUNS_DIR / f"model={model}_regime={regime}_rir={RIR}_test={fold}_val={val}"
    return d / "checkpoints" / "last.ckpt"


def build_train_loaders():
    loaders = {}
    # VoiceHome2 train (arr_0, arr_1)
    if VH_NPZ.exists() and VH_AUDIO_DIR.exists():
        npz = np.load(str(VH_NPZ))
        ds  = VoiceHome2(path_annotations=str(VH_ANNOT_DIR),
                         path_audios=str(VH_AUDIO_DIR),
                         filenames=npz["arr_0"], distances=npz["arr_1"])
        loaders["VoiceHome2"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"VoiceHome2 train: {len(ds)} samples")

    # STARS23 train
    stars_train = STARS_DIR / "train"
    if stars_train.exists():
        ds = STARS23(path_audios=str(stars_train))
        loaders["STARS23"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"STARS23 train: {len(ds)} samples")

    # QMUL train (clean only; 0dB calibration can reuse same predictions)
    qmul_train = QMUL_DIR / "train"
    if qmul_train.exists():
        ds = QMULLIMIT(path_audios=str(qmul_train))
        loaders["QMUL_train"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"QMUL train: {len(ds)} samples")

    return loaders


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        audio  = batch["audio"].to(device)
        labels = batch["label"].to(device)
        ids    = batch["id"]
        preds  = model.model(audio)["dist_pred"]
        for i in range(labels.shape[0]):
            gt   = float(labels[i].cpu())
            pred = float(preds[i].cpu())
            rows.append({"GT": gt, "Pred": pred, "L1": abs(pred - gt),
                         "ID": ids[i] if isinstance(ids[i], str) else str(ids[i])})
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    loaders = build_train_loaders()

    all_rows = []
    for model_name in MODELS:
        for regime in REGIMES:
            print(f"\n=== {model_name} / {regime} ===")
            for fold in range(N_FOLDS):
                ckpt = ckpt_path(model_name, regime, fold)
                if not ckpt.exists():
                    print(f"  [missing] {ckpt}"); continue
                m = RobustTrainer.load_from_checkpoint(str(ckpt), map_location=device)
                m.to(device).eval()
                print(f"  fold {fold}", end="")
                for ds_name, loader in loaders.items():
                    rows = run_inference(m, loader, device)
                    for r in rows:
                        r.update({"model": model_name, "regime": regime,
                                  "fold": fold, "dataset": ds_name})
                    all_rows.extend(rows)
                    print(f"  | {ds_name}: MAE={np.mean([r['L1'] for r in rows]):.3f} m", end="")
                print()

    df = pd.DataFrame(all_rows)
    out = RESULTS / "real_data_train_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
