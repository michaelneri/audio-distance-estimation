"""
eval_qmul_0db.py — Zero-shot evaluation on QMUL-TIMIT at 0 dB SNR.

Noise files are pre-loaded once at startup.
"""
import sys, numpy as np, pandas as pd, torch, librosa as lb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
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
from speaker_distance.paths import DATASETS_DIR, NOISE_DIR  # noqa: E402
sys.path.insert(0, str(BASE))
from model_robust import RobustTrainer

WHAM_DIR  = NOISE_DIR / "test_noise"
QMUL_TEST = DATASETS_DIR / "QMULTIMIT" / "test"
RUNS_DIR  = BASE / "runs_noisy"
N_FOLDS, RIR, BATCH = 5, "full", 16
MODELS  = ["iwaenc_repro", "full_stack"]
REGIMES = ["clean_trained", "noise_trained"]
DB_SNR  = 0
FS      = 16000
CLIP    = FS * 10   # 10 s
TRAIN_MAX = 17   # cap matching the training ceiling


class QMULNoisy(Dataset):
    """QMUL-TIMIT test set with pre-loaded WHAM noise mixed at fixed SNR."""

    def __init__(self, audio_dir: Path, noise_clips: list[np.ndarray], db: int):
        self.files = sorted(audio_dir.glob("*.wav"))
        self.noises = noise_clips
        self.db = db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sound, _ = lb.load(str(self.files[idx]), sr=FS, mono=True, res_type="kaiser_fast")
        sound = sound[:CLIP] if len(sound) >= CLIP else np.pad(sound, (0, CLIP - len(sound)))

        noise = self.noises[np.random.randint(len(self.noises))].copy()
        RMS_s = np.sqrt(np.mean(sound ** 2)) + 1e-9
        RMS_n_target = np.sqrt(RMS_s ** 2 / 10 ** (self.db / 10))
        RMS_n_cur    = np.sqrt(np.mean(noise ** 2)) + 1e-9
        mixed = sound + noise * (RMS_n_target / RMS_n_cur)

        distance = float(self.files[idx].stem.split("_")[2][:-1])
        return {"audio": torch.tensor(mixed).float(),
                "label": torch.tensor(distance).float(),
                "id":    self.files[idx].name}


def preload_noise(noise_dir: Path) -> list[np.ndarray]:
    clips = []
    files = sorted(noise_dir.glob("*.wav"))
    print(f"Pre-loading {len(files)} noise clips... ", end="", flush=True)
    for f in files:
        x, _ = lb.load(str(f), sr=FS, mono=True, res_type="kaiser_fast", duration=10.0)
        x = x[:CLIP] if len(x) >= CLIP else np.pad(x, (0, CLIP - len(x)))
        clips.append(x)
    print("done.")
    return clips


def ckpt_path(model, regime, fold):
    val = (fold + 1) % N_FOLDS
    d = RUNS_DIR / f"model={model}_regime={regime}_rir={RIR}_test={fold}_val={val}"
    return d / "checkpoints" / "last.ckpt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    noise_clips = preload_noise(WHAM_DIR)
    ds = QMULNoisy(QMUL_TEST, noise_clips, DB_SNR)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"QMUL_0dB samples: {len(ds)}")

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
                rows = []
                with torch.no_grad():
                    for batch in loader:
                        audio  = batch["audio"].to(device)
                        labels = batch["label"].to(device)
                        ids    = batch["id"]
                        preds  = m.model(audio)["dist_pred"]
                        for i in range(labels.shape[0]):
                            gt, pred = float(labels[i].cpu()), float(preds[i].cpu())
                            rows.append({"GT": gt, "Pred": pred,
                                         "L1": abs(pred-gt),
                                         "rL1": abs(pred-gt)/max(gt, 1e-6),
                                         "ID": ids[i], "model": model_name,
                                         "regime": regime, "fold": fold,
                                         "dataset": "QMUL_0dB"})
                mae = np.mean([r["L1"] for r in rows])
                print(f"  fold {fold}: MAE={mae:.3f} m (all)  "
                      f"MAE(≤{TRAIN_MAX}m)={np.mean([r['L1'] for r in rows if r['GT']<=TRAIN_MAX]):.3f} m")
                all_rows.extend(rows)

    new_df = pd.DataFrame(all_rows)

    existing = RESULTS / "real_data_results.csv"
    old = pd.read_csv(existing)
    old = old[old.dataset != "QMUL_0dB"]
    combined = pd.concat([old, new_df], ignore_index=True)
    combined.to_csv(existing, index=False)
    print(f"\nSaved {len(new_df)} rows -> {existing}")

    print(f"\nSummary capped at ≤{TRAIN_MAX} m (mean ± 95% CI across 5 folds):")
    for (model, regime), grp in new_df.groupby(["model", "regime"]):
        grp_cap = grp[grp["GT"] <= TRAIN_MAX]
        fold_m  = grp_cap.groupby("fold")["L1"].mean().values
        ci = sp_stats.t.ppf(0.975, df=len(fold_m)-1)*np.std(fold_m,ddof=1)/np.sqrt(len(fold_m))
        lo, hi = np.mean(fold_m)-ci, np.mean(fold_m)+ci
        fold_rm = grp_cap.groupby("fold")["rL1"].mean().values
        print(f"  {model:15s} {regime:15s}  "
              f"MAE={np.mean(fold_m):.2f} [{lo:.2f},{hi:.2f}]  "
              f"rMAE={np.mean(fold_rm)*100:.1f}%")


if __name__ == "__main__":
    main()
