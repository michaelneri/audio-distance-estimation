"""SeldNet: CRNN for single-channel speaker distance estimation.

This is the canonical implementation, merged from three previously divergent forks
(the published TASLP code, the IWAENC RIR-analysis code, and the calibration code).
Differences that mattered are resolved as follows:

* **Parameter names** follow the newer convention (``bn1..3``, ``stft``). Checkpoints
  written by the TASLP-era code used ``batch_norm1..3`` and ``STFT``; load those
  through :func:`speaker_distance.models.compat.load_legacy_state_dict`, which remaps the keys.
* **``att_conf="onSpec"``** applies the attention heatmap to the log-magnitude channel
  and renormalises, as the published TASLP code does. Both newer forks had this branch
  commented out, so the heatmap was computed and discarded; nothing that ran depended
  on that, and the published behaviour is restored here.
* **The ``n_grus=0`` linear fallback** applies no activation between its two layers,
  as the published TASLP code does. Both newer forks inserted an ELU there; neither
  used ``n_grus=0``, so the published behaviour is authoritative.
* **Normalisation** divides by ``std + 1e-8``. The TASLP code divided by ``std``
  directly, so results are not bit-exact against that version.

``tests/test_model_equivalence.py`` pins all of the above against the published
implementation in the repository root.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchlibrosa import STFT

__all__ = ["SeldNet", "SeldTrainer"]

_EPS_NORM = 1e-8
_EPS_LOG = 1e-7


class SeldNet(nn.Module):
    """Configurable CRNN for distance regression.

    Parameters
    ----------
    kernels
        Convolution kernel shape: ``"freq"`` (1x3), ``"time"`` (3x1), ``"square"`` (3x3).
    n_grus
        Number of bidirectional GRU layers (0, 1 or 2). 0 substitutes linear layers.
    features_set
        ``"stft"`` (log-magnitude only), ``"sincos"`` (phase only), ``"all"`` (both).
    att_conf
        ``"Nothing"``, ``"onSpec"`` (attend the magnitude channel) or ``"onAll"``
        (attend every channel).

    Returns from :meth:`forward`
    ---------------------------
    ``(dist_pred [B], frame_pred [B, T], log_mag [B, C, T, F] detached, heatmap | None)``
    """

    N_FFT = 512
    HOP_LENGTH = 256
    NB_CNN_FILT = 128
    POOL_SIZES = (8, 8, 2)
    RNN_SIZE = (128, 128)
    FNN_SIZE = 128
    SAMPLE_RATE = 16000
    CLIP_SECONDS = 10

    _KERNEL_MAP = {"freq": (1, 3), "time": (3, 1), "square": (3, 3)}
    _FEAT_CH_MAP = {"stft": 1, "sincos": 2, "all": 3}
    _ATT_CHOICES = ("Nothing", "onSpec", "onAll")

    def __init__(
        self,
        kernels: str = "freq",
        n_grus: int = 2,
        features_set: str = "all",
        att_conf: str = "Nothing",
    ) -> None:
        super().__init__()

        if kernels not in self._KERNEL_MAP:
            raise ValueError(f"kernels must be one of {list(self._KERNEL_MAP)}, got {kernels!r}")
        if features_set not in self._FEAT_CH_MAP:
            raise ValueError(
                f"features_set must be one of {list(self._FEAT_CH_MAP)}, got {features_set!r}"
            )
        if att_conf not in self._ATT_CHOICES:
            raise ValueError(f"att_conf must be one of {list(self._ATT_CHOICES)}, got {att_conf!r}")
        if n_grus not in (0, 1, 2):
            raise ValueError(f"n_grus must be 0, 1 or 2, got {n_grus!r}")
        if att_conf == "onSpec" and features_set == "sincos":
            # onSpec attends the magnitude channel, which "sincos" does not carry.
            raise ValueError("att_conf='onSpec' requires a features_set that includes magnitude")

        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf
        self.kernel = self._KERNEL_MAP[kernels]

        self.stft = STFT(n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)

        n_ch = self._FEAT_CH_MAP[features_set]
        n_freq_bins = self.N_FFT // 2
        n_time_frames = (
            self.CLIP_SECONDS * self.SAMPLE_RATE + self.N_FFT
        ) // (self.N_FFT - self.HOP_LENGTH) - 1
        self.data_in = [n_ch, n_time_frames, n_freq_bins]

        if att_conf != "Nothing":
            out_ch = 1 if att_conf == "onSpec" else n_ch
            self.heatmap = nn.Sequential(
                nn.Conv2d(n_ch, 16, kernel_size=3, padding="same", bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 64, kernel_size=3, padding="same", bias=False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, out_ch, kernel_size=1, padding="same"),
                nn.Sigmoid(),
            )

        self.conv1 = nn.Conv2d(n_ch, 8, self.kernel, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1_max = nn.MaxPool2d((1, self.POOL_SIZES[0]))
        self.pool1_avg = nn.AvgPool2d((1, self.POOL_SIZES[0]))

        self.conv2 = nn.Conv2d(8, 32, self.kernel, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2_max = nn.MaxPool2d((1, self.POOL_SIZES[1]))
        self.pool2_avg = nn.AvgPool2d((1, self.POOL_SIZES[1]))

        self.conv3 = nn.Conv2d(32, self.NB_CNN_FILT, self.kernel, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(self.NB_CNN_FILT)
        self.pool3_max = nn.MaxPool2d((1, self.POOL_SIZES[2]))
        self.pool3_avg = nn.AvgPool2d((1, self.POOL_SIZES[2]))

        total_pool = self.POOL_SIZES[0] * self.POOL_SIZES[1] * self.POOL_SIZES[2]
        rnn_in = int(n_freq_bins * self.NB_CNN_FILT / total_pool)

        if n_grus == 2:
            self.gru1 = nn.GRU(rnn_in, self.RNN_SIZE[0], bidirectional=True, batch_first=True)
            self.gru2 = nn.GRU(
                self.RNN_SIZE[0] * 2, self.RNN_SIZE[1], bidirectional=True, batch_first=True
            )
        elif n_grus == 1:
            self.gru1 = nn.GRU(rnn_in, self.RNN_SIZE[1], bidirectional=True, batch_first=True)
        else:
            self.lin1 = nn.Linear(rnn_in, self.RNN_SIZE[0])
            self.lin2 = nn.Linear(self.RNN_SIZE[0], self.RNN_SIZE[1] * 2)

        self.fc1 = nn.Linear(self.RNN_SIZE[1] * 2, self.FNN_SIZE)
        self.fc2 = nn.Linear(self.FNN_SIZE, 1)
        self.final = nn.Linear(n_time_frames, 1)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-channel standardisation over the (time, freq) plane."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (x - mean) / (std + _EPS_NORM)

    def _assemble(
        self, log_mag: torch.Tensor, cos_phase: torch.Tensor, sin_phase: torch.Tensor
    ) -> torch.Tensor:
        if self.features_set == "stft":
            return log_mag
        if self.features_set == "sincos":
            return torch.cat([cos_phase, sin_phase], dim=1)
        return torch.cat([log_mag, cos_phase, sin_phase], dim=1)

    def forward(self, x: torch.Tensor):
        x_re, x_im = self.stft(x)

        magn = torch.sqrt(x_re**2 + x_im**2)
        log_mag_full = torch.log(magn**2 + _EPS_LOG)
        phase = torch.angle(x_re + 1j * x_im)

        # Trim the final bin so the frequency axis is exactly N_FFT // 2.
        log_mag = log_mag_full[:, :, :, :-1]
        cos_phase = torch.cos(phase)[:, :, :, :-1]
        sin_phase = torch.sin(phase)[:, :, :, :-1]

        feats = self._normalize(self._assemble(log_mag, cos_phase, sin_phase))

        hm = None
        if self.att_conf != "Nothing":
            hm = self.heatmap(feats)
            if self.att_conf == "onSpec":
                # Attend the magnitude channel only, then rebuild and renormalise.
                # This mirrors the published TASLP behaviour.
                feats = self._normalize(self._assemble(log_mag * hm, cos_phase, sin_phase))
            else:
                feats = feats * hm

        feats = F.elu(self.bn1(self.conv1(feats)))
        feats = self.pool1_max(feats) + self.pool1_avg(feats)

        feats = F.elu(self.bn2(self.conv2(feats)))
        feats = self.pool2_max(feats) + self.pool2_avg(feats)

        feats = F.elu(self.bn3(self.conv3(feats)))
        feats = self.pool3_max(feats) + self.pool3_avg(feats)

        b, c, t, f = feats.shape
        feats = feats.permute(0, 2, 1, 3).reshape(b, t, c * f)

        if self.n_grus == 2:
            feats, _ = self.gru1(feats)
            feats, _ = self.gru2(feats)
        elif self.n_grus == 1:
            feats, _ = self.gru1(feats)
        else:
            # No activation between the two layers, matching the published TASLP
            # code. Both newer forks inserted an ELU here; neither used n_grus=0,
            # so the published behaviour is authoritative.
            feats = self.lin2(self.lin1(feats))

        frame_pred = F.elu(self.fc2(F.elu(self.fc1(feats)))).squeeze(-1)
        dist_pred = self.final(frame_pred).squeeze(-1)

        return dist_pred, frame_pred, log_mag_full.detach(), None if hm is None else hm.detach()


class SeldTrainer(LightningModule):
    """Lightning wrapper around :class:`SeldNet`.

    The loss penalises both the pooled scalar prediction and the mean of the per-frame
    predictions, which keeps the recurrent layers producing meaningful frame-level
    estimates even though supervision is a single scalar per clip::

        L = 0.5 * MSE(dist_pred, y) + 0.5 * MSE(mean_t(frame_pred), y)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        kernels: str = "freq",
        n_grus: int = 2,
        features_set: str = "all",
        att_conf: str = "Nothing",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = SeldNet(kernels, n_grus, features_set, att_conf)
        self._mse = nn.MSELoss()
        self._mae = nn.L1Loss()

        # Appended to by test_step(); read after trainer.test().
        self.all_test_results: list[dict] = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _combined_loss(
        self, dist_pred: torch.Tensor, frame_pred: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return (self._mse(dist_pred, labels) + self._mse(frame_pred.mean(dim=-1), labels)) / 2.0

    def training_step(self, batch, batch_idx):
        audio, labels = batch["audio"], batch["label"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/mae", self._mae(dist_pred, labels), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels = batch["audio"], batch["label"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", self._mae(dist_pred, labels), on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def _as_id(value):
        """IDs are filenames for the real corpora and integers for the synthetic one."""
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        return value

    def test_step(self, batch, batch_idx):
        audio, labels, ids = batch["audio"], batch["label"], batch["id"]
        dist_pred, frame_pred, _, _ = self(audio)
        loss = self._combined_loss(dist_pred, frame_pred, labels)
        self.log("test/mae", self._mae(dist_pred, labels), on_epoch=True)

        for i in range(labels.shape[0]):
            gt = float(labels[i].cpu())
            pred = float(dist_pred[i].cpu())
            self.all_test_results.append(
                {
                    "GT": gt,
                    "Pred": pred,
                    "L1": abs(pred - gt),
                    "rL1": abs(pred - gt) / max(abs(gt), 1e-6),
                    "ID": self._as_id(ids[i]),
                }
            )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "frequency": 1},
        }
