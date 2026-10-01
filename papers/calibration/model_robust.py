"""
model_robust.py — SeldNetMultiTask + RobustTrainer for the robustness experiments.

Adds on top of model.py's SeldNet:
  - Auxiliary heads for log-T60 / log-mixing-time / log-room-volume,
    branching off the GRU output (mean-pooled over time).
  - Log-distance prediction option (model predicts log_dist, exponentiates
    at inference for reporting in metres).
  - SpecAugment applied inside the model after STFT feature extraction
    (training only).

RobustTrainer (LightningModule):
  - Weighted multi-task loss with toggleable auxiliary terms.
  - Optional GPU-side speech augmentation (polarity inversion, random gain,
    small time shift) via a small in-house SimpleAudioAugment -- no external
    library dependency. Training only.
  - Same metric logging shape as SeldTrainer: val/mae, test/mae are in
    metres (regardless of whether the head predicts log_dist).
  - all_test_results unchanged in schema (GT, Pred, L1, rL1, ID).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import SeldNet  # baseline arch from the user's repo


class SimpleAudioAugment(nn.Module):
    """Minimal speech augmentation, pure torch, runs on the same device as
    the input. Per-call probabilities; transforms applied independently:
      - Polarity inversion: flip the sign of the waveform.
      - Random gain in dB.
      - Small time shift (zero-padded, applied per batch -- diverse across
        batches but uniform within a batch, which is fast and good enough
        when the shift range is ±5 ms out of 10 s).

    Notes:
      * Per-sample gain and polarity (vectorised; cheap).
      * Per-batch time shift (per-sample shift would need scatter/gather --
        slower for marginal benefit at this shift magnitude).
      * Pitch shift would need resampling and is dropped (it requires the
        torch-audiomentations or torchaudio resampling pipeline).
    """
    def __init__(self,
                 p_polarity: float = 0.5,
                 p_gain: float = 0.5,
                 p_shift: float = 0.5,
                 gain_range_db: float = 6.0,
                 max_shift_ms: float = 5.0,
                 sample_rate: int = 16000):
        super().__init__()
        self.p_polarity = p_polarity
        self.p_gain = p_gain
        self.p_shift = p_shift
        self.gain_range_db = gain_range_db
        self.max_shift = int(sample_rate * max_shift_ms / 1000.0)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Accept (B, T) or (B, 1, T)
        squeezed = False
        if audio.dim() == 3:
            audio = audio.squeeze(1)
            squeezed = True
        B, T = audio.shape
        dev = audio.device

        # Polarity inversion (per-sample)
        if self.p_polarity > 0:
            flip = (torch.rand(B, 1, device=dev) < self.p_polarity).float()
            audio = audio * (1.0 - 2.0 * flip)

        # Random gain in dB (per-sample)
        if self.p_gain > 0 and self.gain_range_db > 0:
            mask = (torch.rand(B, 1, device=dev) < self.p_gain).float()
            db = (torch.rand(B, 1, device=dev) * 2 - 1) * self.gain_range_db
            gain = mask * (10 ** (db / 20.0)) + (1 - mask)
            audio = audio * gain

        # Time shift (per-batch, zero-padded)
        if self.p_shift > 0 and self.max_shift > 0:
            if float(torch.rand(1)) < self.p_shift:
                shift = int(float(torch.rand(1)) * 2 * self.max_shift) - self.max_shift
                if shift != 0:
                    audio = torch.roll(audio, shifts=shift, dims=-1)
                    if shift > 0:
                        audio[:, :shift] = 0
                    else:
                        audio[:, shift:] = 0

        if squeezed:
            audio = audio.unsqueeze(1)
        return audio

class SpecAugment(nn.Module):
    """Standard SpecAugment time + frequency masking. Applied during
    training only (controlled by self.training)."""
    def __init__(self, n_time_masks: int = 2, time_mask_param: int = 40,
                 n_freq_masks: int = 2, freq_mask_param: int = 20):
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.freq_mask_param = freq_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # x: (B, C, T, F)
        _, _, T, Fq = x.shape
        for _ in range(self.n_time_masks):
            t = torch.randint(0, max(1, self.time_mask_param), (1,)).item()
            if t > 0 and t < T:
                t0 = torch.randint(0, T - t, (1,)).item()
                x[:, :, t0:t0 + t, :] = 0
        for _ in range(self.n_freq_masks):
            f = torch.randint(0, max(1, self.freq_mask_param), (1,)).item()
            if f > 0 and f < Fq:
                f0 = torch.randint(0, Fq - f, (1,)).item()
                x[:, :, :, f0:f0 + f] = 0
        return x


# ---------------------------------------------------------------------------
# Multi-task SeldNet
# ---------------------------------------------------------------------------

class SeldNetMultiTask(nn.Module):
    """SeldNet with auxiliary regression heads.

    Architecture is byte-for-byte the same as model.py's SeldNet up to and
    including the 2-layer bidirectional GRU. From the GRU output (B, T, 2*RNN):
      - Primary head: per-frame distance + temporal pool → dist_pred (B,)
        Optionally outputs log_dist (controlled by predict_log_dist).
      - Auxiliary heads (only built when use_multitask=True):
          - log_t60, log_mt, log_vol, each via MLP on temporally-pooled features.
      - SpecAugment is applied right after STFT (training only).
    """
    def __init__(
        self,
        kernels: str = "freq",
        n_grus: int = 2,
        features_set: str = "all",
        att_conf: str = "onAll",
        predict_log_dist: bool = False,
        use_multitask: bool = False,
        use_specaugment: bool = False,
        specaug_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        base = SeldNet(kernels=kernels, n_grus=n_grus,
                       features_set=features_set, att_conf=att_conf)
        self.stft         = base.stft
        self.features_set = base.features_set
        self.att_conf     = base.att_conf
        self.n_grus       = base.n_grus
        self.kernel       = base.kernels
        self.data_in      = base.data_in
        self.n_time_frames = base.data_in[1]

        # Attention block (if any)
        self.heatmap = base.heatmap if hasattr(base, "heatmap") else None

        # CNN blocks 
        self.conv1, self.bn1 = base.conv1, base.bn1
        self.pool1_max, self.pool1_avg = base.pool1_max, base.pool1_avg
        self.conv2, self.bn2 = base.conv2, base.bn2
        self.pool2_max, self.pool2_avg = base.pool2_max, base.pool2_avg
        self.conv3, self.bn3 = base.conv3, base.bn3
        self.pool3_max, self.pool3_avg = base.pool3_max, base.pool3_avg

        # GRU stack
        if n_grus == 2:
            self.gru1, self.gru2 = base.gru1, base.gru2
        elif n_grus == 1:
            self.gru1 = base.gru1
        else:
            self.lin1, self.lin2 = base.gru_linear1, base.gru_linear2

        # Primary head (per-frame distance + final temporal pool)
        self.fc1   = base.fc1
        self.fc2   = base.fc2
        self.final = base.final

        # ----------------------- New additions -----------------------
        self.predict_log_dist = predict_log_dist
        self.use_multitask = use_multitask
        self.use_specaugment = use_specaugment

        if use_specaugment:
            self.spec_aug = SpecAugment(**(specaug_kwargs or {}))
        else:
            self.spec_aug = None

        if use_multitask:
            # Per-task MLP heads operating on mean-pooled GRU output (D=2*RNN=256)
            d_in = 2 * SeldNet.RNN_SIZE[1]
            def head():
                return nn.Sequential(
                    nn.Linear(d_in, SeldNet.FNN_SIZE),
                    nn.ELU(),
                    nn.Linear(SeldNet.FNN_SIZE, 1),
                )
            self.head_t60 = head()
            self.head_mt  = head()
            self.head_vol = head()

    @staticmethod
    def _normalize(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std  = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # --- STFT + 3-channel feature extraction (identical to SeldNet) ---
        x_re, x_im = self.stft(x)
        magn = torch.sqrt(x_re ** 2 + x_im ** 2)
        log_mag = torch.log(magn ** 2 + 1e-7)
        phase = torch.angle(x_re + 1j * x_im)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        log_mag = log_mag[..., :-1]; cos_p = cos_p[..., :-1]; sin_p = sin_p[..., :-1]

        if self.features_set == "stft":
            feats = log_mag
        elif self.features_set == "sincos":
            feats = torch.cat([cos_p, sin_p], dim=1)
        else:
            feats = torch.cat([log_mag, cos_p, sin_p], dim=1)

        feats = self._normalize(feats)

        # --- SpecAugment (training only, controlled by self.training) ---
        if self.spec_aug is not None:
            feats = self.spec_aug(feats)

        # --- Attention (optional) ---
        if self.att_conf != "Nothing":
            hm = self.heatmap(feats)
            if self.att_conf == "onAll":
                feats = feats * hm

        # --- CNN body ---
        feats = F.elu(self.bn1(self.conv1(feats))); feats = self.pool1_max(feats) + self.pool1_avg(feats)
        feats = F.elu(self.bn2(self.conv2(feats))); feats = self.pool2_max(feats) + self.pool2_avg(feats)
        feats = F.elu(self.bn3(self.conv3(feats))); feats = self.pool3_max(feats) + self.pool3_avg(feats)

        # --- Reshape to sequence ---
        B, C, T, Fq = feats.shape
        feats = feats.permute(0, 2, 1, 3).reshape(B, T, C * Fq)

        # --- Recurrent / linear ---
        if self.n_grus == 2:
            feats, _ = self.gru1(feats); feats, _ = self.gru2(feats)
        elif self.n_grus == 1:
            feats, _ = self.gru1(feats)
        else:
            feats = F.elu(self.lin1(feats)); feats = self.lin2(feats)
        # feats: (B, T, 2*RNN)

        # --- Primary head ---
        frame_pred = F.elu(self.fc2(F.elu(self.fc1(feats)))).squeeze(-1)  # (B, T)
        dist_raw = self.final(frame_pred).squeeze(-1)                     # (B,)

        out: Dict[str, torch.Tensor] = {
            "dist_raw":  dist_raw,                # whatever the head emits
            "frame":     frame_pred,              # per-frame primary head
        }

        # If predicting log-distance, the head's output is log(d). Exponentiate
        # for reporting in metres; loss is computed in log space upstream.
        if self.predict_log_dist:
            out["log_dist_pred"] = dist_raw
            out["dist_pred"] = torch.exp(dist_raw)
        else:
            out["dist_pred"] = dist_raw

        # --- Auxiliary heads ---
        if self.use_multitask:
            pooled = feats.mean(dim=1)                # (B, 2*RNN)
            out["log_t60_pred"] = self.head_t60(pooled).squeeze(-1)
            out["log_mt_pred"]  = self.head_mt(pooled).squeeze(-1)
            out["log_vol_pred"] = self.head_vol(pooled).squeeze(-1)

        return out


# ---------------------------------------------------------------------------
# Robust LightningModule
# ---------------------------------------------------------------------------

class RobustTrainer(pl.LightningModule):
    """Mirrors SeldTrainer's external contract:
      - Expects dict batches with 'audio', 'label', 'id'.
      - Also reads 'log_dist', 'log_t60', 'log_mt', 'log_vol' if available
        (added by data_robust.SyntheticVariantFoldDatasetMT).
      - Logs val/mae and test/mae in metres for ModelCheckpoint compatibility.

    Loss = w_d * MSE(primary)
         + 0.5 * MSE(frame_pred_mean, primary_target)         (combined-loss heritage)
         + w_t60 * MSE(log_t60_pred, log_t60)                 (if multitask)
         + w_mt  * MSE(log_mt_pred,  log_mt)
         + w_vol * MSE(log_vol_pred, log_vol)

    The "primary target" is log_dist when predict_log_dist=True, else dist.
    """
    def __init__(
        self,
        lr: float = 1e-3,
        # SeldNet config
        kernels: str = "freq",
        n_grus: int = 2,
        features_set: str = "all",
        att_conf: str = "onAll",
        # Ablation switches
        predict_log_dist: bool = False,
        use_multitask: bool = False,
        use_specaugment: bool = False,
        use_speech_aug: bool = False,
        # Multitask weights
        w_dist: float = 1.0,
        w_t60: float = 0.3,
        w_mt:  float = 0.3,
        w_vol: float = 0.3,
        # SpecAugment params
        specaug_n_time_masks: int = 2,
        specaug_time_mask_param: int = 40,
        specaug_n_freq_masks: int = 2,
        specaug_freq_mask_param: int = 20,
        # Speech-augmentation params
        sample_rate: int = 16000,
        speech_aug_p: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.predict_log_dist = predict_log_dist
        self.use_multitask = use_multitask
        self.use_speech_aug = use_speech_aug

        self.model = SeldNetMultiTask(
            kernels=kernels, n_grus=n_grus,
            features_set=features_set, att_conf=att_conf,
            predict_log_dist=predict_log_dist,
            use_multitask=use_multitask,
            use_specaugment=use_specaugment,
            specaug_kwargs=dict(
                n_time_masks=specaug_n_time_masks,
                time_mask_param=specaug_time_mask_param,
                n_freq_masks=specaug_n_freq_masks,
                freq_mask_param=specaug_freq_mask_param,
            ),
        )

        self.w_dist, self.w_t60, self.w_mt, self.w_vol = w_dist, w_t60, w_mt, w_vol
        self._mse = nn.MSELoss()
        self._mae = nn.L1Loss()
        self.all_test_results: list[dict] = []

        # Speech augmentation (GPU-side, in-house, no external library).
        if use_speech_aug:
            self.augment = SimpleAudioAugment(
                p_polarity=speech_aug_p,
                p_gain=speech_aug_p,
                p_shift=speech_aug_p,
                gain_range_db=6.0,
                max_shift_ms=5.0,
                sample_rate=sample_rate,
            )
        else:
            self.augment = None

    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------

    def _primary_target(self, batch):
        if self.predict_log_dist:
            return batch["log_dist"]
        return batch["label"]

    def _compute_loss(self, out, batch):
        primary_pred_key = "log_dist_pred" if self.predict_log_dist else "dist_pred"
        target = self._primary_target(batch)
        primary_pred = out[primary_pred_key]
        loss = self.w_dist * self._mse(primary_pred, target)
        # Combined-loss heritage: also penalise the per-frame mean
        loss = loss + 0.5 * self._mse(out["frame"].mean(dim=-1), target)
        if self.use_multitask:
            loss = loss + self.w_t60 * self._mse(out["log_t60_pred"], batch["log_t60"])
            loss = loss + self.w_mt  * self._mse(out["log_mt_pred"],  batch["log_mt"])
            loss = loss + self.w_vol * self._mse(out["log_vol_pred"], batch["log_vol"])
        return loss

    def _maybe_augment_audio(self, audio):
        # audio: (B, T)  →  SimpleAudioAugment accepts (B, T) directly.
        if self.augment is None or not self.training:
            return audio
        return self.augment(audio)

    # ------------------------------------------------------------------

    def training_step(self, batch, _):
        audio = self._maybe_augment_audio(batch["audio"])
        out = self.model(audio)
        loss = self._compute_loss(out, batch)
        # MAE always reported in metres for direct comparability with IWAENC
        mae_m = self._mae(out["dist_pred"], batch["label"])
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/mae",  mae_m, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        out = self.model(batch["audio"])
        loss = self._compute_loss(out, batch)
        mae_m = self._mae(out["dist_pred"], batch["label"])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae",  mae_m, on_epoch=True, prog_bar=True)
        if self.use_multitask:
            self.log("val/mae_log_t60", self._mae(out["log_t60_pred"], batch["log_t60"]), on_epoch=True)
            self.log("val/mae_log_mt",  self._mae(out["log_mt_pred"],  batch["log_mt"]),  on_epoch=True)
            self.log("val/mae_log_vol", self._mae(out["log_vol_pred"], batch["log_vol"]), on_epoch=True)
        return loss

    def on_test_start(self):
        self.all_test_results = []

    def test_step(self, batch, _):
        audio, labels, ids = batch["audio"], batch["label"], batch["id"]
        out = self.model(audio)
        mae_m = self._mae(out["dist_pred"], labels)
        self.log("test/mae", mae_m, on_epoch=True)
        pred_m = out["dist_pred"]

        # Auxiliary GT (present whenever the dataset returns log_t60/mt/vol,
        # which is always with data_robust / data_noisy).
        aux_gt = {k: batch[k] for k in ("log_t60", "log_mt", "log_vol") if k in batch}
        # Auxiliary predictions: only when multi-task is on
        aux_pred = {
            "log_t60": out.get("log_t60_pred"),
            "log_mt":  out.get("log_mt_pred"),
            "log_vol": out.get("log_vol_pred"),
        }

        for i in range(labels.shape[0]):
            gt   = float(labels[i].cpu())
            pred = float(pred_m[i].cpu())
            row = {
                "GT":   gt,
                "Pred": pred,
                "L1":   abs(pred - gt),
                "rL1":  abs(pred - gt) / max(gt, 1e-6),
                "ID":   int(ids[i].cpu()),
            }
            for key, t in aux_gt.items():
                row[f"{key}_gt"] = float(t[i].cpu())
            for key, t in aux_pred.items():
                if t is not None:
                    row[f"{key}_pred"] = float(t[i].cpu())
            self.all_test_results.append(row)

        # Also log auxiliary test MAE to the trainer metrics for W&B
        if all(v is not None for v in aux_pred.values()) and aux_gt:
            self.log("test/mae_log_t60",
                     self._mae(aux_pred["log_t60"], aux_gt["log_t60"]), on_epoch=True)
            self.log("test/mae_log_mt",
                     self._mae(aux_pred["log_mt"],  aux_gt["log_mt"]),  on_epoch=True)
            self.log("test/mae_log_vol",
                     self._mae(aux_pred["log_vol"], aux_gt["log_vol"]), on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler,
                             "monitor": "val/loss", "frequency": 1},
        }
