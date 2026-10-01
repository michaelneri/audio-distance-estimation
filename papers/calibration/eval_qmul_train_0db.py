"""
eval_qmul_train_0db.py
Run inference on the QMUL-TIMIT TRAIN split at 0 dB SNR (WHAM! train noise).
Output appended to real_data_train_results.csv as dataset="QMUL_train_0dB".
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa as lb
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
# Resolve data locations from the repository root rather than this file's folder,
# so the script keeps working wherever it is moved to. See src/speaker_distance/paths.py.
import sys as _sys
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in _sys.path:
    _sys.path.insert(0, str(_REPO / "src"))
from speaker_distance.paths import DATASETS_DIR, NOISE_DIR, wham48_dir  # noqa: E402
sys.path.insert(0, str(BASE))

from model_robust import RobustTrainer

DATASET_ROOT = DATASETS_DIR
QMUL_TRAIN   = DATASET_ROOT / "QMULTIMIT" / "train"
WHAM_DIR     = wham48_dir()
WHAM_META    = WHAM_DIR / "high_res_metadata.csv"
RUNS_DIR     = BASE / "runs_noisy"
BATCH_SIZE   = 16
MODELS       = ["iwaenc_repro", "full_stack"]
REGIMES      = ["clean_trained", "noise_trained"]
RIR          = "full"
N_FOLDS      = 5
FS           = 16_000
CLIP         = 10 * FS  # 10 s


def ckpt_path(model, regime, fold):
    val = (fold + 1) % N_FOLDS
    d = RUNS_DIR / f"model={model}_regime={regime}_rir={RIR}_test={fold}_val={val}"
    return d / "checkpoints" / "last.ckpt"


def preload_noise(wham_dir, meta_path):
    meta = pd.read_csv(meta_path)
    train_files = meta[meta["WHAM! Split"] == "Train"]["Filename"].tolist()
    clips = []
    for fname in train_files:
        fpath = wham_dir / "audio" / fname
        if not fpath.exists():
            continue
        x, _ = lb.load(str(fpath), sr=FS, mono=True,
                        res_type="kaiser_fast", duration=10.0)
        x = x[:CLIP] if len(x) >= CLIP else np.pad(x, (0, CLIP - len(x)))
        clips.append(x)
    print(f"Pre-loaded {len(clips)} WHAM! train noise clips.")
    return clips


class QMULNoisy(Dataset):
    def __init__(self, audio_dir, noise_clips, db=0):
        self.files  = sorted(audio_dir.glob("*.wav"))
        self.noises = noise_clips
        self.db     = db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sound, _ = lb.load(str(self.files[idx]), sr=FS, mono=True,
                            res_type="kaiser_fast")
        sound = sound[:CLIP] if len(sound) >= CLIP else np.pad(sound, (0, CLIP - len(sound)))
        noise = self.noises[np.random.randint(len(self.noises))].copy()
        rms_s = np.sqrt(np.mean(sound ** 2)) + 1e-9
        rms_n_target = np.sqrt(rms_s ** 2 / 10 ** (self.db / 10))
        rms_n = np.sqrt(np.mean(noise ** 2)) + 1e-9
        mixed = sound + noise * (rms_n_target / rms_n)
        distance = float(self.files[idx].stem.split("_")[2][:-1])
        return {
            "audio": torch.tensor(mixed).float(),
            "label": torch.tensor(distance).float(),
            "id":    self.files[idx].name,
        }


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

    noise_clips = preload_noise(WHAM_DIR, WHAM_META)
    ds  = QMULNoisy(QMUL_TRAIN, noise_clips, db=0)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)
    print(f"QMUL train 0dB: {len(ds)} samples\n")

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
                rows = run_inference(m, loader, device)
                mae  = np.mean([r["L1"] for r in rows])
                print(f"  fold {fold}: MAE={mae:.3f} m")
                for r in rows:
                    r.update({"model": model_name, "regime": regime,
                               "fold": fold, "dataset": "QMUL_train_0dB"})
                all_rows.extend(rows)

    df_new = pd.DataFrame(all_rows)
    out = RESULTS / "real_data_train_results.csv"
    df_old = pd.read_csv(out)
    # remove any prior QMUL_train_0dB rows then append
    df_old = df_old[df_old["dataset"] != "QMUL_train_0dB"]
    pd.concat([df_old, df_new], ignore_index=True).to_csv(out, index=False)
    print(f"\nAppended {len(df_new)} rows (QMUL_train_0dB) -> {out}")


if __name__ == "__main__":
    main()
