"""
data.py  –  Dataset, DataModule, and fold-generation utilities
for single-channel speaker distance estimation with RIR variant ablations.

Variants
--------
full      : full RIR (direct + early + late reflections)
direct    : direct path only
no_early  : full RIR without early reflections
no_late   : full RIR without late reverberation
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError("Please install soundfile: pip install soundfile") from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ("full", "direct", "no_early", "no_late")


# ---------------------------------------------------------------------------
# Fold-generation  (replaces make_folds.py)
# ---------------------------------------------------------------------------

def make_folds(
    metadata_csv: str,
    out_csv: str,
    n_folds: int = 5,
    seed: int = 1337,
) -> pd.DataFrame:
    """
    Assign each unique sample_id a fold index in [0, n_folds).

    Reads *metadata_csv* (must have a ``sample_id`` column), shuffles
    deterministically with *seed*, and writes a two-column CSV::

        sample_id, fold

    to *out_csv*.  Returns the resulting DataFrame.
    """
    meta = pd.read_csv(metadata_csv)
    if "sample_id" not in meta.columns:
        raise RuntimeError("metadata_csv must contain column 'sample_id'")

    ids = np.array(sorted(meta["sample_id"].unique().astype(int).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    rows = [
        {"sample_id": int(sid), "fold": int(k)}
        for k, chunk in enumerate(np.array_split(ids, n_folds))
        for sid in chunk
    ]

    folds = pd.DataFrame(rows).sort_values(["fold", "sample_id"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_csv, index=False)
    print(f"[make_folds] Wrote {len(folds)} rows → {out_csv}")
    return folds


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def rms_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a waveform to unit RMS (removes absolute level information)."""
    rms = np.sqrt(np.mean(audio ** 2))
    return audio if rms < eps else audio / rms


def pad_or_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - x.shape[0]), mode="constant")


def apply_random_gain_db(
    audio: np.ndarray, gain_db: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Apply a random gain uniformly sampled from [−gain_db, +gain_db] dB.
    No-op when *gain_db* ≤ 0.
    """
    if gain_db <= 0:
        return audio
    db = rng.uniform(-gain_db, gain_db)
    return (audio * 10.0 ** (db / 20.0)).astype(np.float32)


def wav_path(data_root: str, variant: str, sample_id: int) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return os.path.join(data_root, variant, f"sample_{sample_id:05d}_{variant}.wav")


# ---------------------------------------------------------------------------
# DataLoader configuration (simple dataclass – easy to override in CONFIG)
# ---------------------------------------------------------------------------

@dataclass
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticVariantFoldDataset(Dataset):
    """
    Loads waveforms for one *variant* and a subset of folds.

    Required CSVs
    -------------
    metadata_csv : columns ``sample_id``, ``variant``, ``dist``
    folds_csv    : columns ``sample_id``, ``fold``

    Returns a dict with keys:
        audio  – float32 Tensor [T]
        label  – float32 scalar Tensor (distance in metres)
        id     – int64 scalar Tensor (sample_id)
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        folds_csv: str,
        variant: str,
        folds: Sequence[int],
        sample_rate: int = 16000,
        clip_seconds: float = 10.0,
        gain_db: float = 0.0,
        gain_prob: float = 1.0,
        do_rms_normalize: bool = False,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")

        self.data_root = data_root
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.target_len = int(round(sample_rate * float(clip_seconds)))
        self.gain_db = float(gain_db)
        self.gain_prob = float(gain_prob)
        self.do_rms_normalize = do_rms_normalize
        self.rng = np.random.default_rng(int(seed))

        meta = pd.read_csv(metadata_csv)
        if not {"sample_id", "variant", "dist"}.issubset(meta.columns):
            raise RuntimeError("metadata_csv must have columns: sample_id, variant, dist")

        folds_df = pd.read_csv(folds_csv)
        if not {"sample_id", "fold"}.issubset(folds_df.columns):
            raise RuntimeError("folds_csv must have columns: sample_id, fold")

        sel_ids = folds_df[folds_df["fold"].isin([int(f) for f in folds])]["sample_id"].astype(int).tolist()
        sub = meta[(meta["variant"] == variant) & (meta["sample_id"].isin(sel_ids))].copy()

        if len(sub) == 0:
            raise RuntimeError(
                f"No samples found for variant={variant}, folds={list(folds)}. "
                "Check your metadata_csv and folds_csv."
            )

        self.df = sub.sort_values("sample_id").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = int(row["sample_id"])
        dist = float(row["dist"])

        path = wav_path(self.data_root, self.variant, sample_id)
        audio, sr = sf.read(path, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sr != self.sample_rate:
            raise RuntimeError(
                f"Sample-rate mismatch for {path}: expected {self.sample_rate}, got {sr}"
            )

        audio = pad_or_crop(audio, self.target_len)

        # Optional random gain perturbation
        if self.gain_db > 0 and self.rng.uniform() < self.gain_prob:
            audio = apply_random_gain_db(audio, self.gain_db, self.rng)

        # Optional RMS normalisation (makes the model gain-invariant)
        if self.do_rms_normalize:
            audio = rms_normalize(audio)
        

        return {
            "audio": torch.from_numpy(audio),               # [T]
            "label": torch.tensor(dist, dtype=torch.float32),
            "id":    torch.tensor(sample_id, dtype=torch.int64),
        }


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class SyntheticFoldDataModule(pl.LightningDataModule):
    """
    LightningDataModule for one variant + explicit fold lists.

    Gain augmentation is applied per split independently.
    The **test** split always receives RMS normalisation on top of any
    gain perturbation so that test-time evaluation is level-agnostic.
    """

    def __init__(
        self,
        data_root: str,
        metadata_csv: str,
        folds_csv: str,
        variant: str,
        train_folds: Sequence[int],
        val_folds: Sequence[int],
        test_folds: Sequence[int],
        sample_rate: int = 16000,
        clip_seconds: float = 10.0,
        loader: Optional[LoaderConfig] = None,
        # gain controls
        train_gain_db: float = 0.0,
        train_gain_prob: float = 1.0,
        val_gain_db: float = 0.0,
        val_gain_prob: float = 1.0,
        test_gain_db: float = 0.0,
        test_gain_prob: float = 1.0,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.data_root    = data_root
        self.metadata_csv = metadata_csv
        self.folds_csv    = folds_csv
        self.variant      = variant
        self.train_folds  = list(train_folds)
        self.val_folds    = list(val_folds)
        self.test_folds   = list(test_folds)
        self.sample_rate  = sample_rate
        self.clip_seconds = clip_seconds
        self.loader       = loader or LoaderConfig()

        self.train_gain_db   = float(train_gain_db)
        self.train_gain_prob = float(train_gain_prob)
        self.val_gain_db     = float(val_gain_db)
        self.val_gain_prob   = float(val_gain_prob)
        self.test_gain_db    = float(test_gain_db)
        self.test_gain_prob  = float(test_gain_prob)
        self.seed = int(seed)

        self._train_ds: Optional[SyntheticVariantFoldDataset] = None
        self._val_ds:   Optional[SyntheticVariantFoldDataset] = None
        self._test_ds:  Optional[SyntheticVariantFoldDataset] = None

    def _make_ds(
        self,
        folds: List[int],
        gain_db: float,
        gain_prob: float,
        do_rms_normalize: bool,
        seed_offset: int,
    ) -> SyntheticVariantFoldDataset:
        return SyntheticVariantFoldDataset(
            data_root=self.data_root,
            metadata_csv=self.metadata_csv,
            folds_csv=self.folds_csv,
            variant=self.variant,
            folds=folds,
            sample_rate=self.sample_rate,
            clip_seconds=self.clip_seconds,
            gain_db=gain_db,
            gain_prob=gain_prob,
            do_rms_normalize=do_rms_normalize,
            seed=self.seed + seed_offset,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = self._make_ds(self.train_folds, self.train_gain_db, self.train_gain_prob, True, 1)
            self._val_ds   = self._make_ds(self.val_folds,   self.val_gain_db,   self.val_gain_prob,   True, 2)
        if stage in (None, "test"):
            # Test always uses RMS normalisation (level-agnostic evaluation)
            self._test_ds  = self._make_ds(self.test_folds,  self.test_gain_db,  self.test_gain_prob,  True,  3)

    def _make_loader(self, ds: SyntheticVariantFoldDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.loader.batch_size,
            shuffle=shuffle,
            num_workers=self.loader.num_workers,
            pin_memory=self.loader.pin_memory,
            persistent_workers=self.loader.persistent_workers and self.loader.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup('fit') first"
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup('fit') first"
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "Call setup('test') first"
        return self._make_loader(self._test_ds, shuffle=False)
