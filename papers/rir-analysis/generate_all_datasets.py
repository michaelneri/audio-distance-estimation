#!/usr/bin/env python
"""
generate_all_datasets.py
========================
Windows-compatible script to generate all four RIR dataset variants in a
single pass, ensuring every sample shares the same room, talker, distance,
and RIR across all shortcut configurations.

Run with: python generate_all_datasets.py

This will generate:
  - synthetic_baseline/          (timing + amplitude shortcuts)
  - synthetic_gain_var/          (timing shortcut only — amplitude broken by ±6 dB gain)
  - synthetic_onset_randomized/  (amplitude shortcut only — timing broken by random onset)
  - synthetic_both/              (no shortcuts)
  - shared_rir/                  (RIR .npz files, identical for all datasets)

Timing shortcut:    propagation delay encodes distance → broken by randomising
                    the onset of the convolved signal (RIR left untouched).
Amplitude shortcut: fixed source level lets the model read distance from loudness
                    → broken by applying a random ±6 dB gain to the source.

Each sample index (e.g. sample_00042) uses the exact same room geometry,
source/mic positions, RIR, and speech segment across all four datasets.
Only the shortcut manipulations (onset randomisation and gain jitter) differ.

Estimated time: ~1-2 days (RIR computed once per sample instead of four times).
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

TARGET_FS    = 16000
CLIP_SECONDS = 10.0
N_SAMPLES    = 2500
MAX_ORDER    = 100

DIRECT_WIN_MS  = 2.0
FADE_MS        = 5.0
DIST_MIN       = 1.0
DIST_MAX       = 14.0

MFP_N_PATHS = 10
C_SOUND     = 343.0

ECHO_WIN_MS    = 20.0
ECHO_STRIDE_MS = 4.0

# Maximum random onset padding in seconds, set to the propagation delay at
# DIST_MAX so the random range fully covers the natural delay span.
MAX_ONSET_PAD_S = DIST_MAX / C_SOUND

VARIANTS = ("full", "direct", "no_early", "no_late")

# The four shortcut configurations.
# Each sample is convolved under all four; only the post-processing differs.
SHORTCUT_CONFIGS = (
    {"suffix": "baseline",           "randomize_onset": False, "gain_variation": False},
    {"suffix": "gain_var",           "randomize_onset": False, "gain_variation": True},
    {"suffix": "onset_randomized",   "randomize_onset": True,  "gain_variation": False},
    {"suffix": "both",               "randomize_onset": True,  "gain_variation": True},
)

# Default data directory (adjust if needed)
# Where the generated corpus is written. Defaults to this paper's folder; override
# with --out, since the four configurations plus RIRs total roughly 26 GB and are
# often kept on a separate drive.
DEFAULT_OUT_BASE = Path(__file__).resolve().parents[2] / "zenodo" / "data"

# The EARS anechoic corpus is not redistributable (CC BY-NC 4.0) and is not part of
# this repository. Point at your own copy with the positional argument or by setting
# SPEAKER_DISTANCE_EARS. Download: https://sp-uhh.github.io/ears_dataset/
DEFAULT_DATA_DIR = (
    Path(os.environ["SPEAKER_DISTANCE_EARS"]).expanduser()
    if os.environ.get("SPEAKER_DISTANCE_EARS")
    else None
)

# -------------------------------------------------------------------------
# SIGNAL-PROCESSING HELPERS
# -------------------------------------------------------------------------

def half_cosine_fade(n: int) -> np.ndarray:
    return ((1 - np.cos(np.linspace(0, np.pi, n))) / 2).astype(np.float32)


def fade_out_after(h: np.ndarray, cut: int, fade_len: int) -> np.ndarray:
    out = h.copy()
    stop     = min(len(out), cut + fade_len)
    ramp_len = stop - cut
    if ramp_len > 0:
        ramp = half_cosine_fade(fade_len)
        out[cut:stop] *= 1.0 - ramp[:ramp_len]
    out[stop:] = 0.0
    return out


def fade_in_from(h: np.ndarray, start: int, fade_len: int) -> np.ndarray:
    out        = h.copy()
    ramp_start = max(0, start - fade_len)
    ramp_len   = start - ramp_start
    if ramp_len > 0:
        ramp = half_cosine_fade(fade_len)
        out[ramp_start:start] *= ramp[-ramp_len:]
    out[:ramp_start] = 0.0
    return out


def mfp_mixing_time_samples(room_dim: np.ndarray, fs: int) -> int:
    lx, ly, lz = room_dim
    V     = lx * ly * lz
    S     = 2.0 * (lx * ly + lx * lz + ly * lz)
    l_mfp = 4.0 * V / S
    t_mix_s = MFP_N_PATHS * l_mfp / C_SOUND
    return int(round(t_mix_s * fs))


def echo_density_profile(
    h: np.ndarray,
    fs: int,
    win_ms: float    = ECHO_WIN_MS,
    stride_ms: float = ECHO_STRIDE_MS,
) -> tuple[np.ndarray, np.ndarray]:
    win_len  = max(3, int(win_ms    / 1000.0 * fs))
    stride   = max(1, int(stride_ms / 1000.0 * fs))
    half_win = win_len // 2
    ERFC_REF = 0.3173

    t_list, eta_list = [], []
    for t in range(half_win, len(h) - half_win, stride):
        window = h[t - half_win: t + half_win + 1]
        sigma  = float(np.sqrt(np.mean(window ** 2)))
        if sigma < 1e-12:
            eta_list.append(0.0)
        else:
            eta_list.append(float(np.mean(np.abs(window) > sigma)) / ERFC_REF)
        t_list.append(t / fs * 1000.0)

    return np.array(t_list, dtype=np.float32), np.array(eta_list, dtype=np.float32)


def load_speech(path: str, target_fs: int) -> np.ndarray:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if sr != target_fs:
        g = math.gcd(sr, target_fs)
        x = resample_poly(x, target_fs // g, sr // g)
    return x.astype(np.float32)


def fit_to_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) >= n:
        start = rng.integers(0, len(x) - n + 1)
        return x[start: start + n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def build_variants(
    h_aligned: np.ndarray,
    d_end: int,
    early_cut: int,
    fade_len: int,
) -> dict[str, np.ndarray]:
    early_cut = max(early_cut, d_end + fade_len + 1)

    h_full    = h_aligned.copy()
    h_direct  = fade_out_after(h_aligned, d_end, fade_len)
    h_no_late = fade_out_after(h_aligned, early_cut, fade_len)
    h_no_early = (fade_out_after(h_aligned, d_end, fade_len)
                  + fade_in_from(h_aligned, early_cut, fade_len))

    return {"full": h_full, "direct": h_direct,
            "no_late": h_no_late, "no_early": h_no_early}


# -------------------------------------------------------------------------
# CORE GENERATION
# -------------------------------------------------------------------------

def generate_all(
    clean_wavs: list[str],
    n_samples: int,
    out_base: Path | str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Generate all four shortcut-configuration datasets in a single pass.

    Every sample index shares the same room, RIR, talker, and speech
    segment across all four datasets.  Only the onset randomisation and
    gain jitter differ.
    """
    OUT_BASE    = Path(out_base) if out_base is not None else DEFAULT_OUT_BASE
    RIR_DIR     = OUT_BASE / "shared_rir"

    # ── Set up output directories ─────────────────────────────────────────
    # Generation overwrites: refuse unless the caller opted in, since a corpus
    # takes 1-2 days to rebuild and lives under these exact names.
    existing = [
        OUT_BASE / f"synthetic_{cfg['suffix']}" for cfg in SHORTCUT_CONFIGS
    ] + [RIR_DIR]
    existing = [p for p in existing if p.exists()]
    if existing and not overwrite:
        listing = "\n    ".join(str(p) for p in existing)
        raise FileExistsError(
            f"Output directories already exist:\n    {listing}\n"
            "  Generation would delete them. Pass --overwrite to proceed, "
            "or --out to write elsewhere."
        )

    dataset_dirs: dict[str, Path] = {}
    for cfg in SHORTCUT_CONFIGS:
        ds_root = OUT_BASE / f"synthetic_{cfg['suffix']}"
        if ds_root.exists():
            shutil.rmtree(ds_root)
        for v in VARIANTS:
            (ds_root / v).mkdir(parents=True, exist_ok=True)
        dataset_dirs[cfg["suffix"]] = ds_root

    if RIR_DIR.exists():
        shutil.rmtree(RIR_DIR)
    RIR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Generating all {len(SHORTCUT_CONFIGS)} datasets in a single pass")
    print(f"{'='*70}")
    for cfg in SHORTCUT_CONFIGS:
        print(f"  • synthetic_{cfg['suffix']:<22s}  "
              f"onset_rand={str(cfg['randomize_onset']):<5s}  "
              f"gain_var={cfg['gain_variation']}")
    print(f"  Shared RIRs  → {RIR_DIR}")
    print(f"  Samples:       {n_samples}")
    print(f"{'='*70}\n")

    # ── RNGs ──────────────────────────────────────────────────────────────
    # Main RNG: room geometry, source/mic placement, speech selection.
    #           Shared identically across all datasets.
    rng = np.random.default_rng(seed=0)

    # Shortcut RNG: gain jitter and onset padding.
    # Separate from the main RNG so that drawing shortcut values never
    # perturbs the room/talker sequence.
    shortcut_rng = np.random.default_rng(seed=42)

    # ── Pre-computed constants ────────────────────────────────────────────
    fade_len      = int(FADE_MS       / 1000.0 * TARGET_FS)
    d_halfwin     = int(DIRECT_WIN_MS / 1000.0 * TARGET_FS)
    clip_len      = int(CLIP_SECONDS * TARGET_FS)
    max_onset_pad = int(np.ceil(MAX_ONSET_PAD_S * TARGET_FS))

    # Per-dataset metadata accumulators
    all_metadata: dict[str, list[dict]] = {
        cfg["suffix"]: [] for cfg in SHORTCUT_CONFIGS
    }

    start_time = time.time()

    for i in range(n_samples):
        # ── 1. Room geometry (shared) ─────────────────────────────────────
        while True:
            dist_target = float(rng.uniform(DIST_MIN, DIST_MAX))
            min_side = dist_target + 1.0
            low  = [max(5.0, min_side), max(5.0, min_side), 3.0]
            high = [15.0, 12.0, 5.0]
            if low[0] > high[0] or low[1] > high[1]:
                continue

            room_dim = rng.uniform(low=low, high=high)
            mic_pos = rng.uniform(low=[0.5, 0.5, 1.2], high=room_dim - 0.5)
            for _ in range(200):
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction)
                src_pos = mic_pos + direction * dist_target
                if np.all(src_pos >= 0.5) and np.all(src_pos <= room_dim - 0.5):
                    break
            else:
                continue

            dist = float(np.linalg.norm(src_pos - mic_pos))
            break

        # ── 2. Simulate RIR (once) ───────────────────────────────────────
        alpha = float(rng.uniform(0.1, 0.45))
        room  = pra.ShoeBox(
            room_dim, fs=TARGET_FS, max_order=MAX_ORDER, absorption=alpha
        )
        room.add_microphone_array(
            pra.MicrophoneArray(mic_pos.reshape(3, 1), TARGET_FS)
        )
        room.add_source(src_pos.tolist())
        room.compute_rir()
        h_raw = np.array(room.rir[0][0], dtype=np.float32)

        # ── 3. Detect direct-path onset ──────────────────────────────────
        peak_thresh = 0.1 * np.max(np.abs(h_raw))
        direct_idx  = int(np.where(np.abs(h_raw) >= peak_thresh)[0][0])

        # ── 4. Mixing time via mean free path ────────────────────────────
        early_cut = mfp_mixing_time_samples(room_dim, TARGET_FS)
        early_cut = min(direct_idx + early_cut, len(h_raw) - 1)
        d_end     = direct_idx + d_halfwin

        # ── 5. Build RIR variants (shared) ───────────────────────────────
        variants = build_variants(h_raw, d_end, early_cut, fade_len)

        # ── 6. Save RIRs once ────────────────────────────────────────────
        np.savez(
            RIR_DIR / f"sample_{i:05d}_rir.npz",
            h_full            = variants["full"],
            h_direct          = variants["direct"],
            h_no_late         = variants["no_late"],
            h_no_early        = variants["no_early"],
            room_dim          = room_dim,
            distance_m        = dist,
            absorption_coeff  = alpha,
            direct_idx_raw    = direct_idx,
            early_cut_samples = early_cut,
            d_end_samples     = d_end,
        )

        # ── 7. Load & prepare speech (shared) ────────────────────────────
        speech_raw = fit_to_length(
            load_speech(
                clean_wavs[int(rng.integers(0, len(clean_wavs)))],
                TARGET_FS,
            ),
            clip_len, rng,
        )

        # ── 8. Derived room quantities (shared) ─────────────────────────
        lx, ly, lz = room_dim
        S     = 2.0 * (lx * ly + lx * lz + ly * lz)
        l_mfp = 4.0 * float(np.prod(room_dim)) / S
        mix_ms = (early_cut - direct_idx) / TARGET_FS * 1000.0

        # ── 9. Draw shortcut values for this sample ──────────────────────
        # Drawn once from the shortcut RNG; reused by whichever configs
        # need them, so gain_var and both share the same gain_db, and
        # onset_randomized and both share the same onset_pad.
        gain_db   = float(shortcut_rng.uniform(-6.0, 6.0))
        onset_pad = int(shortcut_rng.integers(0, max_onset_pad + 1))

        # ── 10. Convolve & save for each shortcut config ─────────────────
        for cfg in SHORTCUT_CONFIGS:
            sfx     = cfg["suffix"]
            do_gain = cfg["gain_variation"]
            do_rand = cfg["randomize_onset"]

            # Apply gain (or not) to the shared speech
            g = gain_db if do_gain else 0.0
            speech = speech_raw * (10.0 ** (g / 20.0))

            pad = onset_pad if do_rand else 0

            for v_name, h_v in variants.items():
                y = fftconvolve(speech, h_v, mode="full").astype(np.float32)

                if do_rand:
                    # Strip natural propagation delay, prepend random silence
                    y = y[direct_idx:]
                    y = np.concatenate(
                        [np.zeros(pad, dtype=np.float32), y]
                    )

                y = y[:clip_len]
                if len(y) < clip_len:
                    y = np.concatenate(
                        [y, np.zeros(clip_len - len(y), dtype=np.float32)]
                    )

                sf.write(
                    str(dataset_dirs[sfx] / v_name
                        / f"sample_{i:05d}_{v_name}.wav"),
                    y, TARGET_FS, subtype="FLOAT",
                )
                all_metadata[sfx].append({
                    "sample_id":        i,
                    "variant":          v_name,
                    "dist":             round(dist, 4),
                    "room_vol_m3":      round(float(np.prod(room_dim)), 2),
                    "mean_free_path_m": round(l_mfp, 4),
                    "mixing_time_ms":   round(mix_ms, 2),
                    "absorption_coeff": round(alpha, 3),
                    "source_gain_db":   round(g, 2),
                    "onset_pad_samples": pad,
                    "randomize_onset":  do_rand,
                    "gain_variation":   do_gain,
                })

        # ── Progress ─────────────────────────────────────────────────────
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg     = elapsed / (i + 1)
            eta_hrs = avg * (n_samples - i - 1) / 3600
            print(f"  [{i+1:>5}/{n_samples}]  "
                  f"dist={dist:.2f}m  t_mix={mix_ms:.1f}ms  "
                  f"gain_draw={gain_db:+.1f}dB  "
                  f"onset_draw={onset_pad}  "
                  f"ETA={eta_hrs:.1f}h")

    # ── 11. Save per-dataset metadata & plots ────────────────────────────
    for cfg in SHORTCUT_CONFIGS:
        sfx    = cfg["suffix"]
        ds_dir = dataset_dirs[sfx]
        df     = pd.DataFrame(all_metadata[sfx])
        csv_path = ds_dir / "metadata_variants.csv"
        df.to_csv(csv_path, index=False)

        full = df[df["variant"] == "full"]
        print(f"\n{'='*70}")
        print(f"  Dataset complete: synthetic_{sfx}")
        print(f"{'='*70}")
        print(f"  {n_samples * len(VARIANTS)} WAV files  →  {ds_dir}/{{variant}}/")
        print(f"  Metadata CSV               →  {csv_path}")
        print(f"  Distance range   : {full['dist'].min():.2f} – "
              f"{full['dist'].max():.2f} m")
        print(f"  Mixing time range: {full['mixing_time_ms'].min():.1f} – "
              f"{full['mixing_time_ms'].max():.1f} ms")

        _plot_distributions(df, ds_dir, RIR_DIR, cfg["randomize_onset"])

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ALL DATASETS GENERATED")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Shared RIRs ({n_samples} files)  →  {RIR_DIR}/")
    for cfg in SHORTCUT_CONFIGS:
        print(f"  • {dataset_dirs[cfg['suffix']]}")
    print(f"{'='*70}\n")


# -------------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------------

def _plot_distributions(
    df: pd.DataFrame,
    out_root: Path,
    rir_dir: Path,
    onset_randomized: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping plots.")
        return

    full      = df[df["variant"] == "full"].copy()
    dist_vals = full["dist"].to_numpy()
    mix_vals  = full["mixing_time_ms"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    onset_str = "randomized" if onset_randomized else "natural"
    fig.suptitle(
        f"Dataset distributions  (N={len(full)} samples, onset {onset_str})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    def _stat_box(ax, vals, unit):
        ax.text(
            0.97, 0.95,
            f"min  {vals.min():.2f} {unit}\n"
            f"max  {vals.max():.2f} {unit}\n"
            f"mean {vals.mean():.2f} {unit}\n"
            f"std  {vals.std():.2f} {unit}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    ax = axes[0]
    ax.hist(dist_vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(dist_vals.mean(), color="#C44E52", lw=1.8,
               label=f"mean {dist_vals.mean():.2f} m")
    ax.axvline(np.median(dist_vals), color="#DD8452", lw=1.8, ls="--",
               label=f"median {np.median(dist_vals):.2f} m")
    ax.set_xlabel("Source–mic distance (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distance distribution", fontsize=11)
    ax.legend(fontsize=9)
    _stat_box(ax, dist_vals, "m")

    ax = axes[1]
    ax.hist(mix_vals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.axvline(mix_vals.mean(), color="#C44E52", lw=1.8,
               label=f"mean {mix_vals.mean():.1f} ms")
    ax.axvline(np.median(mix_vals), color="#DD8452", lw=1.8, ls="--",
               label=f"median {np.median(mix_vals):.1f} ms")
    ax.set_xlabel("MFP mixing time (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Mixing time  (l_mfp × {MFP_N_PATHS} / c)", fontsize=11)
    ax.legend(fontsize=9)
    _stat_box(ax, mix_vals, "ms")

    ax = axes[2]
    if onset_randomized:
        rir_files = sorted(rir_dir.glob("*.npz"))
        n_diag    = min(200, len(rir_files))
        rng_diag  = np.random.default_rng(42)
        subset    = rng_diag.choice(len(rir_files), size=n_diag, replace=False)

        eta_interp_list = []
        t_common_ms     = np.arange(0, 300, float(ECHO_STRIDE_MS))

        for idx in subset:
            npz = np.load(rir_files[idx])
            h   = npz["h_full"].astype(np.float32)
            t_ms, eta = echo_density_profile(h, TARGET_FS)
            if len(t_ms) < 2:
                continue
            eta_interp_list.append(
                np.interp(t_common_ms, t_ms, eta,
                          left=0.0, right=float(eta[-1]))
            )

        if eta_interp_list:
            eta_arr  = np.stack(eta_interp_list)
            eta_mean = eta_arr.mean(axis=0)
            eta_std  = eta_arr.std(axis=0)
            ax.plot(t_common_ms, eta_mean, color="#C44E52", lw=2,
                    label="mean η(t)")
            ax.fill_between(
                t_common_ms,
                np.clip(eta_mean - eta_std, 0, None),
                np.clip(eta_mean + eta_std, 0, None),
                alpha=0.25, color="#C44E52", label="±1 std",
            )
            ax.axhline(1.0, color="grey", lw=1, ls="--",
                       label="Gaussian (η=1)")
            ax.axvline(mix_vals.mean(), color="#55A868", lw=1.5, ls=":",
                       label=f"mean t_mix = {mix_vals.mean():.0f} ms")

        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Time from direct path (ms)", fontsize=11)
        ax.set_ylabel("Echo density η(t)", fontsize=11)
        ax.set_title(f"Abel & Huang η(t)  (N={n_diag} RIRs)", fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5, 0.5,
            "Echo density profile\n(only computed for\nonset-randomized datasets)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="grey",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out_path = out_root / "dataset_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution plot  →  {out_path}")


# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the four shortcut-configuration datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ears", nargs="?", type=Path, default=DEFAULT_DATA_DIR,
        help="path to the EARS anechoic speech corpus",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_BASE,
        help="directory to write the corpus into (needs ~50 GB free)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_SAMPLES,
        help="number of acoustic scenes to generate",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="delete existing output directories instead of refusing",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="skip the confirmation prompt (for unattended runs)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RIR Distance Estimation Dataset Generator")
    print("  Single-pass pipeline — shared rooms & talkers across datasets")
    print("=" * 70)
    print(f"\nThis will generate {len(SHORTCUT_CONFIGS)} datasets × "
          f"{args.n_samples} samples = "
          f"{len(SHORTCUT_CONFIGS) * args.n_samples} total samples")
    print("Estimated time: ~1-2 days on a single CPU")
    print(f"Output location: {args.out}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        free_gb = shutil.disk_usage(str(args.out)).free / (1024 ** 3)
        required_gb = 50  # Conservative estimate
        print(f"Free space at destination: {free_gb:.1f} GB")
        if free_gb < required_gb:
            print(f"WARNING: less than {required_gb} GB available; "
                  "each configuration needs ~6 GB plus ~2 GB of RIRs.\n")
    except OSError as exc:
        print(f"Could not check disk space: {exc}\n")

    data_dir = args.ears
    if data_dir is None:
        print("ERROR: the EARS speech corpus is required but was not given.")
        print("  Pass it:   python generate_all_datasets.py /path/to/EARS")
        print("  Or set:    SPEAKER_DISTANCE_EARS=/path/to/EARS")
        print("  Download:  https://sp-uhh.github.io/ears_dataset/")
        sys.exit(1)

    clean_wavs = sorted(
        glob.glob(str(data_dir / "p[0-9][0-9][0-9]" / "*.wav"))
    )
    if not clean_wavs:
        print(f"ERROR: No EARS files found at {data_dir}")
        print("  Expected speaker folders named p001, p002, ... each holding .wav files.")
        print("\nUsage: python generate_all_datasets.py [path/to/EARS]")
        sys.exit(1)

    print(f"Found {len(clean_wavs)} speech files from "
          f"{len(set(Path(p).parent for p in clean_wavs))} talkers")
    print(f"Data directory: {data_dir}\n")

    if not args.yes:
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    generate_all(
        clean_wavs=clean_wavs,
        n_samples=args.n_samples,
        out_base=args.out,
        overwrite=args.overwrite,
    )

    print("\nNext steps:")
    print("  1. Verify each dataset's distribution plot")
    print(f"  2. Point train_val_test.py data_root at "
          f"{args.out / 'synthetic_<configuration>'}")
    print("  3. Train models with/without RMS normalization")
    print("  4. Collect the 4x2x4 results table\n")


if __name__ == "__main__":
    main()
