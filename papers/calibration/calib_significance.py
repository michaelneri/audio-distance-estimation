"""
calib_significance.py
(1) Paired t-test of each calibration map against the constant predictor,
    across the 5 cross-validation folds (df=4) -- the same test used for
    Tables 1-2, so the paper stays internally consistent.
    Per fold we average the MAE over 500 random calibration draws.
(2) Pearson r (with p) for Table 3.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as st

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
N_REPEATS = 500
N_SHOTS   = [5, 10, 20, 50]
CAP       = 10.97
EPS       = 1e-3

df_test  = pd.read_csv(RESULTS / "real_data_results.csv")
df_train = pd.read_csv(RESULTS / "real_data_train_results.csv")

DATASETS = {"VoiceHome2": "VoiceHome2", "STARS23": "STARS23",
            "QMUL_clean": "QMUL_train", "QMUL_0dB": "QMUL_train_0dB"}
CONDS = [("iwaenc_repro", "clean_trained", "Baseline, clean-trained"),
         ("iwaenc_repro", "noise_trained", "Baseline, noise-trained"),
         ("full_stack",   "clean_trained", "Full stack, clean-trained"),
         ("full_stack",   "noise_trained", "Full stack, noise-trained")]
rng = np.random.default_rng(42)


# ---------------- calibration maps ----------------
def m_const(p, d, pt):  return np.full(len(pt), d.mean())
def m_affine(p, d, pt):
    c, *_ = np.linalg.lstsq(np.column_stack([p, np.ones(len(p))]), d, rcond=None)
    return pt * c[0] + c[1]
def m_offset(p, d, pt): return pt + np.mean(d - p)
def m_scale(p, d, pt):  return pt * ((p @ d) / (p @ p + 1e-12))
def m_logaff(p, d, pt):
    lp, ld = np.log(np.clip(p, EPS, None)), np.log(np.clip(d, EPS, None))
    c, *_ = np.linalg.lstsq(np.column_stack([lp, np.ones(len(lp))]), ld, rcond=None)
    return np.exp(c[1]) * np.clip(pt, EPS, None) ** c[0]
def m_shrink(p, d, pt):
    n = len(p)
    if n < 3 or np.std(p) < 1e-12: return np.full(len(pt), d.mean())
    r = np.corrcoef(p, d)[0, 1]
    if not np.isfinite(r): return np.full(len(pt), d.mean())
    c, *_ = np.linalg.lstsq(np.column_stack([p, np.ones(n)]), d, rcond=None)
    a = (r**2 / (r**2 + 1.0/n)) * c[0]
    return pt * a + (d.mean() - a * p.mean())

MAPS = {"constant": m_const, "offset": m_offset, "scale": m_scale,
        "log-affine": m_logaff, "affine": m_affine, "shrink": m_shrink}


def stars(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))


# ---------------- (1) per-fold MAE, then paired t-test ----------------
print("=" * 86)
print("Calibration maps vs constant predictor -- paired t-test across 5 folds (df=4)")
print("full-stack noise-trained; Delta<0 means the map beats the constant predictor")
print("=" * 86)

MODEL, REGIME = "full_stack", "noise_trained"
rows = []
for ds, train_ds in DATASETS.items():
    te = df_test[(df_test.model == MODEL) & (df_test.regime == REGIME)
                 & (df_test.dataset == ds)]
    tr = df_train[(df_train.model == MODEL) & (df_train.regime == REGIME)
                  & (df_train.dataset == train_ds)]
    folds = sorted(te.fold.unique())
    print(f"\n{ds}")
    print(f"  {'map':<12}" + "".join(f"{'N='+str(n):>20}" for n in N_SHOTS))
    for mname in MAPS:
        if mname == "constant":
            continue
        line = f"  {mname:<12}"
        for n in N_SHOTS:
            per_fold_m, per_fold_c = [], []
            for fid in folds:
                trf, tef = tr[tr.fold == fid], te[te.fold == fid]
                if len(trf) < n or len(tef) == 0:
                    continue
                pp, dd = trf["Pred"].values, trf["GT"].values
                pt, gt = tef["Pred"].values, tef["GT"].values
                acc_m, acc_c = [], []
                for _ in range(N_REPEATS):
                    idx = rng.choice(len(pp), n, replace=False)
                    p, d = pp[idx], dd[idx]
                    try:
                        o = MAPS[mname](p, d, pt)
                        if np.all(np.isfinite(o)):
                            acc_m.append(np.mean(np.abs(o - gt)))
                    except Exception:
                        pass
                    acc_c.append(np.mean(np.abs(m_const(p, d, pt) - gt)))
                if acc_m:
                    per_fold_m.append(np.mean(acc_m))
                    per_fold_c.append(np.mean(acc_c))
            if len(per_fold_m) >= 2:
                a, c = np.array(per_fold_m), np.array(per_fold_c)
                t, p_ = st.ttest_rel(a, c)
                delta = a.mean() - c.mean()
                line += f"{a.mean():>7.2f} {delta:+6.2f}{stars(p_):<4}"
                rows.append(dict(dataset=ds, method=mname, N=n, mae=a.mean(),
                                 delta=delta, p=p_))
            else:
                line += f"{'--':>20}"
        print(line)

pd.DataFrame(rows).to_csv(RESULTS / "calib_significance.csv", index=False)
print(f"\nsaved -> calib_significance.csv")


# ---------------- (2) Pearson r for Table 3 ----------------
print("\n" + "=" * 86)
print("Table 3 correlations: Pearson r (theory-consistent) vs Spearman rho")
print("QMUL capped at 10.97 m, pooled across folds -- matches current Table 3")
print("=" * 86)
print(f"{'model':<28}{'corpus':<13}{'Pearson r':>12}{'p':>10}{'Spearman':>11}{'p':>10}")
print("-" * 86)
for model, regime, label in CONDS:
    for ds in ["VoiceHome2", "STARS23", "QMUL_clean", "QMUL_0dB"]:
        g = df_test[(df_test.model == model) & (df_test.regime == regime)
                    & (df_test.dataset == ds)]
        if g.empty:
            continue
        if ds.startswith("QMUL"):
            g = g[g.GT <= CAP]
        r, pr = st.pearsonr(g.GT, g.Pred)
        rho, ps = st.spearmanr(g.GT, g.Pred)
        print(f"{label:<28}{ds:<13}{r:>+12.3f}{pr:>10.1e}{rho:>+11.3f}{ps:>10.1e}")
    print()
