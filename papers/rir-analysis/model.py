"""
model.py  –  SeldNet architecture and PyTorch-Lightning training wrapper
for single-channel speaker distance estimation.

Architecture overview
---------------------
1. STFT front-end  → log-magnitude + cos/sin phase channels
2. Optional spatial-attention heatmap  (onSpec | onAll | Nothing)
3. Three CNN blocks with mixed max+avg pooling
4. 0 / 1 / 2 bidirectional GRU layers  (or linear fallback)
5. Two FC layers → per-frame distance → pooled scalar prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
from pytorch_lightning import LightningModule


# ---------------------------------------------------------------------------
# Core network
# ---------------------------------------------------------------------------

class SeldNet(nn.Module):
    """
    Configurable CRNN for distance regression.

    Parameters
    ----------
    kernels      : "freq" (1×3) | "time" (3×1) | "square" (3×3)
    n_grus       : number of bidirectional GRU layers (0 | 1 | 2)
    features_set : "stft" (log-mag only) | "sincos" (phase only) | "all"
    att_conf     : "Nothing" | "onSpec" | "onAll"
    """

    # Fixed architecture constants
    N_FFT       = 512
    HOP_LENGTH  = 256
    NB_CNN_FILT = 128
    POOL_SIZES  = [8, 8, 2]
    RNN_SIZE    = [128, 128]
    FNN_SIZE    = 128

    _KERNEL_MAP  = {"freq": (1, 3), "time": (3, 1), "square": (3, 3)}
    _FEAT_CH_MAP = {"stft": 1, "sincos": 2, "all": 3}

    def __init__(
        self,
        kernels: str      = "freq",
        n_grus: int       = 2,
        features_set: str = "all",
        att_conf: str     = "Nothing",
    ) -> None:
        super().__init__()

        if kernels not in self._KERNEL_MAP:
            raise ValueError(f"kernels must be one of {list(self._KERNEL_MAP)}")
        if features_set not in self._FEAT_CH_MAP:
            raise ValueError(f"features_set must be one of {list(self._FEAT_CH_MAP)}")
        if att_conf not in ("Nothing", "onSpec", "onAll"):
            raise ValueError("att_conf must be 'Nothing', 'onSpec', or 'onAll'")
        if n_grus not in (0, 1, 2):
            raise ValueError("n_grus must be 0, 1, or 2")

        self.n_grus       = n_grus
        self.features_set = features_set
        self.att_conf     = att_conf
        self.kernel       = self._KERNEL_MAP[kernels]

        # STFT front-end
        self.stft = STFT(n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)

        # Input dimensions [C, T_frames, F_bins]
        n_ch          = self._FEAT_CH_MAP[features_set]
        n_freq_bins   = self.N_FFT // 2                                              # 256 (trimmed)
        n_time_frames = (10 * 16000 + self.N_FFT) // (self.N_FFT - self.HOP_LENGTH) - 1
        self.data_in  = [n_ch, n_time_frames, n_freq_bins]

        # Optional attention heatmap
        if att_conf != "Nothing":
            out_ch = 1 if att_conf == "onSpec" else n_ch
            self.heatmap = nn.Sequential(
                nn.Conv2d(n_ch, 16,     kernel_size=3, padding="same", bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16,   64,     kernel_size=3, padding="same", bias=False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64,   out_ch, kernel_size=1, padding="same"),
                nn.Sigmoid(),
            )

        # CNN block 1 ─ in: n_ch, out: 8
        self.conv1     = nn.Conv2d(n_ch, 8,                self.kernel, padding="same", bias=False)
        self.bn1       = nn.BatchNorm2d(8)
        self.pool1_max = nn.MaxPool2d((1, self.POOL_SIZES[0]))
        self.pool1_avg = nn.AvgPool2d((1, self.POOL_SIZES[0]))

        # CNN block 2 ─ in: 8, out: 32
        self.conv2     = nn.Conv2d(8,  32,               self.kernel, padding="same", bias=False)
        self.bn2       = nn.BatchNorm2d(32)
        self.pool2_max = nn.MaxPool2d((1, self.POOL_SIZES[1]))
        self.pool2_avg = nn.AvgPool2d((1, self.POOL_SIZES[1]))

        # CNN block 3 ─ in: 32, out: NB_CNN_FILT
        self.conv3     = nn.Conv2d(32, self.NB_CNN_FILT, self.kernel, padding="same", bias=False)
        self.bn3       = nn.BatchNorm2d(self.NB_CNN_FILT)
        self.pool3_max = nn.MaxPool2d((1, self.POOL_SIZES[2]))
        self.pool3_avg = nn.AvgPool2d((1, self.POOL_SIZES[2]))

        # Recurrent / linear sequence module
        total_pool = self.POOL_SIZES[0] * self.POOL_SIZES[1] * self.POOL_SIZES[2]   # 128
        rnn_in = int(n_freq_bins * self.NB_CNN_FILT / total_pool)

        if n_grus == 2:
            self.gru1 = nn.GRU(rnn_in,               self.RNN_SIZE[0], bidirectional=True, batch_first=True)
            self.gru2 = nn.GRU(self.RNN_SIZE[0] * 2, self.RNN_SIZE[1], bidirectional=True, batch_first=True)
        elif n_grus == 1:
            self.gru1 = nn.GRU(rnn_in, self.RNN_SIZE[1], bidirectional=True, batch_first=True)
        else:   # n_grus == 0: replace GRUs with two linear layers
            self.lin1 = nn.Linear(rnn_in,           self.RNN_SIZE[0])
            self.lin2 = nn.Linear(self.RNN_SIZE[0], self.RNN_SIZE[1] * 2)

        # Prediction head
        self.fc1   = nn.Linear(self.RNN_SIZE[1] * 2, self.FNN_SIZE)
        self.fc2   = nn.Linear(self.FNN_SIZE,         1)
        self.final = nn.Linear(n_time_frames,         1)  # pool over time

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-channel normalisation (µ=0, σ=1)."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std  = x.std( dim=(2, 3), keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-8)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor [B, T]  –  raw waveform at 16 kHz

        Returns
        -------
        dist_pred  : Tensor [B]            – predicted distance per sample
        frame_pred : Tensor [B, T_frames]  – per-frame predictions
        log_mag    : Tensor                – detached log-magnitude (for logging/viz)
        heatmap    : Tensor | None         – attention map (None when att_conf='Nothing')
        """
        x_re, x_im = self.stft(x)

        # Spectral features
        magn      = torch.sqrt(x_re ** 2 + x_im ** 2)
        log_mag   = torch.log(magn ** 2 + 1e-7)
        phase     = torch.angle(x_re + 1j * x_im)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Trim last frequency bin → exactly N_FFT//2 bins
        log_mag   = log_mag[  :, :, :, :-1]
        cos_phase = cos_phase[:, :, :, :-1]
        sin_phase = sin_phase[:, :, :, :-1]

        # Assemble feature tensor
        if self.features_set == "stft":
            feats = log_mag
        elif self.features_set == "sincos":
            feats = torch.cat([cos_phase, sin_phase], dim=1)
        else:   # "all"
            feats = torch.cat([log_mag, cos_phase, sin_phase], dim=1)

        feats = self._normalize(feats)

        # Attention heatmap (optional)
        hm = None
        if self.att_conf != "Nothing":
            hm = self.heatmap(feats)
            if self.att_conf == "onSpec":
                log_mag_att = log_mag * hm
                #feats = self._normalize(torch.cat([log_mag_att, cos_phase, sin_phase], dim=1))
            else:   # "onAll"
                feats = feats * hm

        # CNN block 1
        feats = F.elu(self.bn1(self.conv1(feats)))
        feats = self.pool1_max(feats) + self.pool1_avg(feats)

        # CNN block 2
        feats = F.elu(self.bn2(self.conv2(feats)))
        feats = self.pool2_max(feats) + self.pool2_avg(feats)

        # CNN block 3
        feats = F.elu(self.bn3(self.conv3(feats)))
        feats = self.pool3_max(feats) + self.pool3_avg(feats)

        # Reshape to sequence: [B, T_frames, C*F]
        B, C, T, Fq = feats.shape
        feats = feats.permute(0, 2, 1, 3).reshape(B, T, C * Fq)

        # Recurrent / linear module
        if self.n_grus == 2:
            feats, _ = self.gru1(feats)
            feats, _ = self.gru2(feats)
        elif self.n_grus == 1:
            feats, _ = self.gru1(feats)
        else:
            feats = F.elu(self.lin1(feats))
            feats = self.lin2(feats)

        # Per-frame distance prediction → [B, T_frames]
        frame_pred = F.elu(self.fc2(F.elu(self.fc1(feats)))).squeeze(-1)

        # Scalar prediction (pool over time frames) → [B]
        dist_pred = self.final(frame_pred).squeeze(-1)

        return dist_pred, frame_pred, log_mag.detach(), hm.detach() if hm is not None else None


# ---------------------------------------------------------------------------
# PyTorch-Lightning wrapper
# ---------------------------------------------------------------------------

class SeldTrainer(LightningModule):
    """
    Wraps SeldNet for training with PyTorch-Lightning.

    Combined loss
    -------------
    L = 0.5 × MSE(dist_pred, gt) + 0.5 × MSE(mean_t(frame_pred), gt)

    Both the final scalar and the per-frame average are penalised, which
    encourages the recurrent layers to produce meaningful per-frame estimates
    even though ground-truth is a single scalar per clip.
    """

    def __init__(
        self,
        lr: float         = 1e-3,
        kernels: str      = "freq",
        n_grus: int       = 2,
        features_set: str = "all",
        att_conf: str     = "Nothing",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr    = lr
        self.model = SeldNet(kernels, n_grus, features_set, att_conf)
        self._mse  = nn.MSELoss()
        self._mae  = nn.L1Loss()

        # Populated incrementally by test_step(); read after trainer.test()
        self.all_test_results: list[dict] = []

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _combined_loss(
        self,
        dist_pred: torch.Tensor,
        frame_pred: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return (self._mse(dist_pred, labels) + self._mse(frame_pred.mean(dim=-1), labels)) / 2.0

    # ------------------------------------------------------------------

    def training_step(self, batch, _):
        audio, labels = batch["audio"], batch["label"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("train/loss", loss,                          on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/mae",  self._mae(dist_pred, labels), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        audio, labels = batch["audio"], batch["label"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("val/loss", loss,                          on_epoch=True, prog_bar=True)
        self.log("val/mae",  self._mae(dist_pred, labels), on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        audio, labels, ids = batch["audio"], batch["label"], batch["id"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("test/mae", self._mae(dist_pred, labels), on_epoch=True)

        for i in range(labels.shape[0]):
            gt   = float(labels[i].cpu())
            pred = float(dist_pred[i].cpu())
            self.all_test_results.append({
                "GT":   gt,
                "Pred": pred,
                "L1":   abs(pred - gt),
                "rL1":  abs(pred - gt) / max(gt, 1e-6),
                "ID":   int(ids[i].cpu()),
            })
        return loss

    def configure_optimizers(self):
        opt       = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.2, verbose=True
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "frequency": 1},
        }
