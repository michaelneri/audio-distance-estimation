"""Guards against the fork drift that this repository already suffered once.

Three copies of SeldNet existed with incompatible parameter names and one silently
disabled code path. These tests pin the canonical behaviour so the same class of
divergence is caught here rather than discovered two papers later.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from speaker_distance.models import SeldNet, is_legacy_state_dict, load_state_dict_any, remap_legacy_keys

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODEL = REPO_ROOT / "papers" / "estimation-2023-2024" / "model.py"

CONFIGS = [
    ("freq", 2, "all", "Nothing"),
    ("freq", 2, "all", "onAll"),
    ("freq", 2, "all", "onSpec"),
    ("freq", 2, "stft", "onAll"),
    ("freq", 2, "sincos", "onAll"),
    ("freq", 1, "all", "onAll"),
    ("freq", 0, "all", "Nothing"),
    ("time", 2, "all", "onAll"),
    ("square", 2, "all", "onAll"),
]

# The canonical model divides by (std + 1e-8) where the TASLP code divided by std,
# so agreement is close but not bit-exact.
ATOL = 1e-4


def _load_legacy_module():
    if not LEGACY_MODEL.exists():
        pytest.skip("legacy model.py not present")
    spec = importlib.util.spec_from_file_location("legacy_model", LEGACY_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_model"] = module
    spec.loader.exec_module(module)
    return module


def _to_legacy_names(state_dict):
    """Inverse of the compat remap, for building a synthetic legacy checkpoint."""
    out = {}
    for key, value in state_dict.items():
        new = key
        if new.startswith("stft."):
            new = "STFT." + new[len("stft.") :]
        for i in (1, 2, 3):
            new = new.replace(f"bn{i}.", f"batch_norm{i}.")
        for i in (1, 2):
            new = new.replace(f"lin{i}.", f"gru_linear{i}.")
        out[new] = value
    return out


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(str(x) for x in c))
def test_legacy_checkpoint_roundtrip(cfg):
    """A checkpoint written with TASLP-era names loads and behaves identically."""
    torch.manual_seed(0)
    net = SeldNet(*cfg).eval()

    legacy_sd = _to_legacy_names(net.state_dict())
    assert is_legacy_state_dict(legacy_sd), "renamed dict should be detected as legacy"

    torch.manual_seed(999)  # different init, so a failed load would be visible
    other = SeldNet(*cfg).eval()
    result = load_state_dict_any(other, legacy_sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    x = torch.randn(2, 160000)
    with torch.no_grad():
        a = net(x)[0]
        b = other(x)[0]
    assert torch.equal(a, b)


def test_remap_is_noop_for_canonical_keys():
    torch.manual_seed(0)
    sd = SeldNet("freq", 2, "all", "onAll").state_dict()
    assert not is_legacy_state_dict(sd)
    assert set(remap_legacy_keys(sd)) == set(sd)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(str(x) for x in c))
def test_matches_published_taslp_model(cfg):
    """Canonical output tracks the published TASLP implementation."""
    legacy = _load_legacy_module()

    if cfg[2] == "sincos" and cfg[3] == "onSpec":
        pytest.skip("onSpec is undefined without a magnitude channel")

    torch.manual_seed(0)
    legacy_net = legacy.SeldNet(*cfg).eval()
    torch.manual_seed(0)
    canonical = SeldNet(*cfg).eval()

    load_state_dict_any(canonical, legacy_net.state_dict(), strict=True)

    x = torch.randn(2, 160000)
    with torch.no_grad():
        legacy_out = legacy_net(x)[0]
        canonical_out = canonical(x)[0]

    assert torch.allclose(legacy_out, canonical_out, atol=ATOL), (
        f"max delta {(legacy_out - canonical_out).abs().max().item():.3e} exceeds {ATOL}"
    )


def test_onspec_actually_applies_the_heatmap():
    """Regression test: both newer forks had this branch commented out.

    With the heatmap applied, onSpec and Nothing must differ for the same weights.
    """
    torch.manual_seed(0)
    net = SeldNet("freq", 2, "all", "onSpec").eval()
    x = torch.randn(2, 160000)

    with torch.no_grad():
        _, _, _, hm = net(x)
    assert hm is not None, "onSpec must return a heatmap"

    # Force the heatmap towards zero; the prediction must move.
    with torch.no_grad():
        before = net(x)[0].clone()
        final_conv = [m for m in net.heatmap if isinstance(m, torch.nn.Conv2d)][-1]
        final_conv.bias.fill_(-20.0)  # sigmoid(-20) ~ 0
        after = net(x)[0]

    assert not torch.allclose(before, after, atol=1e-6), (
        "changing the heatmap did not change the output - onSpec is not being applied"
    )


def test_sincos_rejects_onspec():
    with pytest.raises(ValueError, match="magnitude"):
        SeldNet("freq", 2, "sincos", "onSpec")


def test_output_shapes():
    net = SeldNet("freq", 2, "all", "onAll").eval()
    x = torch.randn(3, 160000)
    with torch.no_grad():
        dist, frame, log_mag, hm = net(x)
    assert dist.shape == (3,)
    assert frame.shape[0] == 3 and frame.ndim == 2
    assert log_mag.ndim == 4
    assert hm is not None
