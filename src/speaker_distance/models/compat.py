"""Checkpoint compatibility between the stage-1 code and the canonical model.

The code published with WASPAA 2023 / TASLP 2024 named its batch-norm layers
``batch_norm1..3``, its STFT front-end ``STFT``, and the ``n_grus=0`` fallback
``gru_linear1/2``. Later work renamed all three. The architecture is otherwise
identical, so a checkpoint from either naming scheme fits the canonical
:class:`~speaker_distance.models.seldnet.SeldNet` once the keys are remapped.

This is not only about historical artifacts. ``papers/estimation-2023-2024`` still
imports its own local ``model.py``, so running those training scripts *today* produces
legacy-named checkpoints; this module is what lets that output load into the shared
core. It becomes purely historical once those scripts are pointed at
:mod:`speaker_distance.models` instead.

``tests/test_model_equivalence.py`` also relies on it, transplanting weights from the
published implementation into the canonical one to prove the two agree.

Always load through :func:`load_state_dict_any`. Calling ``load_state_dict`` directly
with a legacy checkpoint raises on 18 keys; calling it with ``strict=False`` is worse,
because the convolution weights load while every batch-norm layer silently keeps its
freshly initialised statistics — which looks like success.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import torch
import torch.nn as nn

__all__ = [
    "LEGACY_KEY_PATTERNS",
    "is_legacy_state_dict",
    "remap_legacy_keys",
    "load_state_dict_any",
]

# (legacy pattern, canonical replacement) applied in order to each key.
LEGACY_KEY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^STFT\.", "stft."),
    (r"^model\.STFT\.", "model.stft."),
    (r"(^|\.)batch_norm(\d)\.", r"\1bn\2."),
    # The n_grus=0 linear fallback was renamed too.
    (r"(^|\.)gru_linear(\d)\.", r"\1lin\2."),
)

_LEGACY_MARKERS = (".batch_norm", ".gru_linear")


def is_legacy_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """True when any key uses the TASLP-era naming."""
    return any(
        k.startswith("STFT.")
        or k.startswith("model.STFT.")
        or any(marker in f".{k}" for marker in _LEGACY_MARKERS)
        for k in state_dict
    )


def remap_legacy_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Rename TASLP-era keys to the canonical scheme. Unknown keys pass through."""
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in LEGACY_KEY_PATTERNS:
            new_key = re.sub(pattern, replacement, new_key)
        if new_key in out:
            raise KeyError(f"key collision while remapping: {key!r} -> {new_key!r}")
        out[new_key] = value
    return out


def load_state_dict_any(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> nn.modules.module._IncompatibleKeys:
    """Load a state dict of either naming convention.

    Accepts a raw state dict or a full Lightning checkpoint (a mapping carrying a
    ``"state_dict"`` entry). Remapping is applied only when legacy keys are detected,
    so canonical checkpoints are untouched.
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], Mapping):
        state_dict = state_dict["state_dict"]

    if is_legacy_state_dict(state_dict):
        state_dict = remap_legacy_keys(state_dict)

    return module.load_state_dict(state_dict, strict=strict)


def load_checkpoint(
    module: nn.Module,
    path: str,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> nn.modules.module._IncompatibleKeys:
    """Load a ``.ckpt``/``.pth`` file from either era onto ``module``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return load_state_dict_any(module, ckpt, strict=strict)
