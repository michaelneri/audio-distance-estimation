"""
run_calibration_variants.py
Evaluates every calibration variant defined in Section 2 of the draft:

  constant     d_cal = mean(C_N)                   (1 param, the reference)
  affine       d_cal = a*d + b                     (2 params, Eq. 1)
  offset-only  d_cal = d + b        (a == 1)       (1 param)
  scale-only   d_cal = a*d          (b == 0)       (1 param)
  log-affine   d_cal = A * d^beta                  (2 params, log-domain LS)
  shrink       Eq. (8): a_l = rho^2/(rho^2 + 1/N) * a_hat,
                        b_l = dbar - a_l * dhatbar

Also validates the decision rule of Eq. (7):  rho^2 > 1/(N+2).

Outputs
-------
  calib_variants_results.csv   per model/regime/dataset/N/method
  console tables ready to paste
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

BASE      = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
N_REPEATS = 500
N_SHOTS   = [5, 10, 20, 50]
EPS       = 1e-3

df_test  = pd.read_csv(RESULTS / "real_data_results.csv")
df_train = pd.read_csv(RESULTS / "real_data_train_results.csv")

CONDITIONS = [
    ("iwaenc_repro", "clean_trained", "Baseline, clean-trained"),
    ("iwaenc_repro", "noise_trained", "Baseline, noise-trained"),
    ("full_stack",   "clean_trained", "Full stack, clean-trained"),
    ("full_stack",   "noise_trained", "Full stack, noise-trained"),
]
DATASETS = {
    "VoiceHome2": ("VoiceHome2",     "VoiceHome2"),
    "STARS23":    ("STARS23",        "STARS23"),
    "QMUL_clean": ("QMUL_train",     "QMUL-TIMIT, clean"),
    "QMUL_0dB":   ("QMUL_train_0dB", "QMUL-TIMIT, 0 dB"),
}
METHODS = ["constant", "affine", "offset", "scale", "log-affine", "shrink"]

rng = np.random.default_rng(42)


# ── calibration maps ──────────────────────────────────────────────────────────
def apply_constant(p, d, pt):
    return np.full(len(pt), d.mean())


def apply_affine(p, d, pt):
    A = np.column_stack([p, np.ones(len(p))])
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    return pt * c[0] + c[1]


def apply_offset(p, d, pt):
    """a == 1: only the prior shift is corrected."""
    b = np.mean(d - p)
    return pt + b


def apply_scale(p, d, pt):
    """b == 0: purely multiplicative."""
    a = (p @ d) / (p @ p + 1e-12)
    return pt * a


def apply_logaffine(p, d, pt):
    """d_cal = A * d^beta, least squares in the log domain."""
    lp = np.log(np.clip(p, EPS, None))
    ld = np.log(np.clip(d, EPS, None))
    A = np.column_stack([lp, np.ones(len(lp))])
    c, *_ = np.linalg.lstsq(A, ld, rcond=None)
    return np.exp(c[1]) * np.clip(pt, EPS, None) ** c[0]


def apply_shrink(p, d, pt):
    """Eq. (8): shrink the OLS slope by rho^2 / (rho^2 + 1/N)."""
    n = len(p)
    if n < 3 or np.std(p) < 1e-12:
        return np.full(len(pt), d.mean())
    rho = np.corrcoef(p, d)[0, 1]
    if not np.isfinite(rho):
        return np.full(len(pt), d.mean())
    A = np.column_stack([p, np.ones(n)])
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    a_hat = c[0]
    w = rho ** 2 / (rho ** 2 + 1.0 / n)
    a_l = w * a_hat
    b_l = d.mean() - a_l * p.mean()
    return pt * a_l + b_l


FN = {
    "constant":   apply_constant,
    "affine":     apply_affine,
    "offset":     apply_offset,
    "scale":      apply_scale,
    "log-affine": apply_logaffine,
    "shrink":     apply_shrink,
}


def ci95(v):
    v = np.asarray(v)
    k = len(v)
    if k < 2:
        return float("nan"), float("nan"), float("nan")
    hw = sp_stats.t.ppf(0.975, df=k - 1) * v.std(ddof=1) / np.sqrt(k)
    return float(v.mean()), float(v.mean() - hw), float(v.mean() + hw)


def run(train_sub, test_sub, n, repeats):
    folds = sorted(test_sub["fold"].unique())
    reps = {m: [] for m in METHODS}
    for _ in range(repeats):
        fold_acc = {m: [] for m in METHODS}
        for fid in folds:
            tr = train_sub[train_sub["fold"] == fid]
            te = test_sub[test_sub["fold"] == fid]
            if len(tr) < n or len(te) == 0:
                continue
            idx = rng.choice(len(tr), n, replace=False)
            p, d = tr["Pred"].values[idx], tr["GT"].values[idx]
            pt, gt = te["Pred"].values, te["GT"].values
            for m in METHODS:
                try:
                    out = FN[m](p, d, pt)
                    if not np.all(np.isfinite(out)):
                        continue
                    fold_acc[m].append(np.mean(np.abs(out - gt)))
                except Exception:
                    pass
        for m in METHODS:
            if fold_acc[m]:
                reps[m].append(np.mean(fold_acc[m]))
    return {m: ci95(reps[m]) for m in METHODS}


# ── correlations (population estimates on the training split) ────────────────
def corrs(train_sub):
    rp, rs = [], []
    for fid in sorted(train_sub["fold"].unique()):
        g = train_sub[train_sub["fold"] == fid]
        if len(g) < 3:
            continue
        rp.append(sp_stats.pearsonr(g["Pred"], g["GT"])[0])
        rs.append(sp_stats.spearmanr(g["Pred"], g["GT"])[0])
    return float(np.mean(rp)), float(np.mean(rs))


# ── main sweep ────────────────────────────────────────────────────────────────
rows = []
for model, regime, label in CONDITIONS:
    for ds, (train_ds, ds_label) in DATASETS.items():
        te = df_test[(df_test.model == model) & (df_test.regime == regime)
                     & (df_test.dataset == ds)]
        tr = df_train[(df_train.model == model) & (df_train.regime == regime)
                      & (df_train.dataset == train_ds)]
        if te.empty or tr.empty:
            print(f"[SKIP] {label} / {ds}")
            continue
        uncal = float(np.mean([te[te.fold == f]["L1"].mean()
                               for f in sorted(te.fold.unique())]))
        r_p, r_s = corrs(tr)
        for n in N_SHOTS:
            res = run(tr, te, n, N_REPEATS)
            for m in METHODS:
                mean, lo, hi = res[m]
                rows.append(dict(model=model, regime=regime, label=label,
                                 dataset=ds, ds_label=ds_label, N=n,
                                 method=m, mae=mean, lo=lo, hi=hi,
                                 uncal=uncal, pearson=r_p, spearman=r_s))
        print(f"done: {label:<28} {ds}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "calib_variants_results.csv", index=False)
print(f"\nSaved -> calib_variants_results.csv  ({len(df)} rows)\n")


# ── report: all variants, full-stack noise-trained ───────────────────────────
def block(model, regime, title):
    print("=" * 78)
    print(title)
    print("=" * 78)
    sub = df[(df.model == model) & (df.regime == regime)]
    for ds, (_, ds_label) in DATASETS.items():
        s = sub[sub.dataset == ds]
        if s.empty:
            continue
        u = s.uncal.iloc[0]
        rp, rs = s.pearson.iloc[0], s.spearman.iloc[0]
        print(f"\n{ds_label}   uncal={u:.2f} m   "
              f"r(Pearson)={rp:+.3f}  rho(Spearman)={rs:+.3f}  r^2={rp**2:.3f}")
        print(f"  {'method':<12}" + "".join(f"{'N='+str(n):>9}" for n in N_SHOTS))
        for m in METHODS:
            line = f"  {m:<12}"
            for n in N_SHOTS:
                v = s[(s.N == n) & (s.method == m)]
                line += f"{v.mae.iloc[0]:>9.2f}" if len(v) else f"{'--':>9}"
            print(line)
    print()


block("full_stack", "noise_trained", "ALL VARIANTS -- Full stack, noise-trained")


# ── validation of the decision rule, Eq. (7): rho^2 > 1/(N+2) ────────────────
print("=" * 78)
print("DECISION RULE Eq.(7):  fit the slope iff  r^2 > 1/(N+2)")
print("=" * 78)
print(f"{'corpus':<20}{'model':<26}{'r^2':>7}{'N':>4}"
      f"{'thresh':>8}{'predict':>10}{'affine':>8}{'const':>8}{'actual':>9}{'ok':>4}")
print("-" * 104)
ok = tot = 0
for model, regime, label in CONDITIONS:
    for ds, (_, ds_label) in DATASETS.items():
        s = df[(df.model == model) & (df.regime == regime) & (df.dataset == ds)]
        if s.empty:
            continue
        r2 = s.pearson.iloc[0] ** 2
        for n in N_SHOTS:
            th = 1.0 / (n + 2)
            pred = "affine" if r2 > th else "constant"
            a = s[(s.N == n) & (s.method == "affine")].mae.iloc[0]
            c = s[(s.N == n) & (s.method == "constant")].mae.iloc[0]
            act = "affine" if a < c else "constant"
            good = pred == act
            ok += good; tot += 1
            print(f"{ds_label:<20}{label:<26}{r2:>7.3f}{n:>4}{th:>8.3f}"
                  f"{pred:>10}{a:>8.2f}{c:>8.2f}{act:>9}{'Y' if good else 'n':>4}")
print("-" * 104)
print(f"rule agrees with the empirical winner in {ok}/{tot} cases "
      f"({100*ok/tot:.0f}%)\n")


# ── shrinkage vs the two it interpolates ─────────────────────────────────────
print("=" * 78)
print("SHRINKAGE Eq.(8) vs affine and constant  (all models, mean over corpora)")
print("=" * 78)
print(f"{'corpus':<20}{'N':>4}{'affine':>9}{'constant':>10}{'shrink':>9}"
      f"{'best-of-2':>11}{'shrink vs best':>16}")
print("-" * 79)
for ds, (_, ds_label) in DATASETS.items():
    for n in N_SHOTS:
        s = df[(df.dataset == ds) & (df.N == n)]
        a = s[s.method == "affine"].mae.mean()
        c = s[s.method == "constant"].mae.mean()
        sh = s[s.method == "shrink"].mae.mean()
        best = min(a, c)
        print(f"{ds_label:<20}{n:>4}{a:>9.2f}{c:>10.2f}{sh:>9.2f}{best:>11.2f}"
              f"{100*(sh-best)/best:>15.1f}%")
print()
