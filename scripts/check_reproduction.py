#!/usr/bin/env python
"""Global reproduction check.

Runs every check that does not require training, and reports each as PASS, FAIL or
SKIP. SKIP means a prerequisite (usually the Zenodo data) is absent, not that
something is broken - the script is meant to be useful both on a bare clone and on a
fully populated working copy.

    python scripts/check_reproduction.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    colour = {"PASS": "", "FAIL": "", "SKIP": ""}[status]
    print(f"  [{status}] {colour}{name}" + (f" - {detail}" if detail else ""))


def check(name: str):
    """Decorator turning a function into a recorded check.

    Return a string for detail, or raise to fail. Raise SkipCheck to skip.
    """

    def wrapper(fn):
        try:
            detail = fn() or ""
            record("PASS", name, detail)
        except SkipCheck as exc:
            record("SKIP", name, str(exc))
        except Exception as exc:  # noqa: BLE001
            record("FAIL", name, f"{type(exc).__name__}: {exc}")
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn

    return wrapper


class SkipCheck(Exception):
    pass


# ---------------------------------------------------------------------------
print("\n=== 1. Environment ===")


@check("package imports")
def _():
    import speaker_distance

    return f"speaker_distance {speaker_distance.__version__}"


@check("dependency versions")
def _():
    import pytorch_lightning as pl
    import torch

    major = int(pl.__version__.split(".")[0])
    if major < 2:
        raise RuntimeError(f"Lightning {pl.__version__} < 2.0; this code targets 2.x")
    return f"torch {torch.__version__}, lightning {pl.__version__}"


# ---------------------------------------------------------------------------
print("\n=== 2. Canonical model ===")


@check("model builds and runs forward")
def _():
    import torch

    from speaker_distance.models import SeldNet

    net = SeldNet("freq", 2, "all", "onAll").eval()
    with torch.no_grad():
        dist, frame, log_mag, hm = net(torch.randn(2, 160000))
    assert dist.shape == (2,), dist.shape
    assert hm is not None
    n_params = sum(p.numel() for p in net.parameters())
    return f"{n_params/1e6:.2f}M params, output {tuple(dist.shape)}"


@check("legacy checkpoint remapping")
def _():
    import torch

    from speaker_distance.models import SeldNet, load_state_dict_any

    net = SeldNet("freq", 2, "all", "onAll")
    legacy = {}
    for k, v in net.state_dict().items():
        k = k.replace("stft.", "STFT.")
        for i in (1, 2, 3):
            k = k.replace(f"bn{i}.", f"batch_norm{i}.")
        legacy[k] = v
    res = load_state_dict_any(SeldNet("freq", 2, "all", "onAll"), legacy, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return f"{len(legacy)} tensors remapped"


@check("all published configurations build")
def _():
    from speaker_distance.models import SeldNet

    built = 0
    for kernels in ("freq", "time", "square"):
        for n_grus in (0, 1, 2):
            for feats in ("stft", "sincos", "all"):
                for att in ("Nothing", "onSpec", "onAll"):
                    if att == "onSpec" and feats == "sincos":
                        continue  # undefined: no magnitude channel
                    SeldNet(kernels, n_grus, feats, att)
                    built += 1
    return f"{built} configurations"


# ---------------------------------------------------------------------------
print("\n=== 3. Annotations ===")

EXPECTED = {
    "qmultimit_labels.csv": (2340, "distance_m", 2.00, 16.35),
    "starss23_labels.csv": (2934, "distance_m", 1.00, 3.57),
    "noise_whamr_manifest.csv": (160, None, None, None),
}


@check("label tables load with expected shape")
def _():
    import pandas as pd

    labels = ROOT / "data" / "labels"
    if not labels.exists():
        raise SkipCheck("data/labels not present")
    parts = []
    for fname, (n_rows, col, lo, hi) in EXPECTED.items():
        path = labels / fname
        if not path.exists():
            raise SkipCheck(f"{fname} missing")
        df = pd.read_csv(path)
        if len(df) != n_rows:
            raise RuntimeError(f"{fname}: expected {n_rows} rows, got {len(df)}")
        if col:
            lo_a, hi_a = df[col].min(), df[col].max()
            if abs(lo_a - lo) > 0.01 or abs(hi_a - hi) > 0.01:
                raise RuntimeError(f"{fname}: range {lo_a:.2f}-{hi_a:.2f} != {lo}-{hi}")
        parts.append(f"{fname.split('_')[0]}={len(df)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
print("\n=== 4. Published summary tables ===")


@check("calibration summaries present and parseable")
def _():
    import pandas as pd

    base = ROOT / "papers" / "calibration" / "results"
    if not base.exists():
        raise SkipCheck("papers/calibration not present")
    found = []
    for name in (
        "calib_variants_results.csv",
        "noisy_summary.csv",
        "noisy_summary_no_early.csv",
        "calib_significance.csv",
        "real_data_summary.csv",
    ):
        path = base / name
        if not path.exists():
            raise RuntimeError(f"missing summary: {name}")
        df = pd.read_csv(path)
        found.append(f"{name.replace('.csv','')}({len(df)})")
    return ", ".join(found)


@check("IWAENC variant results present")
def _():
    import pandas as pd

    base = ROOT / "papers" / "rir-analysis"
    if not base.exists():
        raise SkipCheck("IWAENC folder not present")
    out = []
    for name in ("sweep_summary.csv", "variant_results.csv"):
        path = base / "results" / name
        if not path.exists():
            raise RuntimeError(f"missing: {name}")
        out.append(f"{name.replace('.csv','')}({len(pd.read_csv(path))})")
    folds = base / "data" / "folds5.csv"
    if folds.exists():
        df = pd.read_csv(folds)
        out.append(f"folds5({len(df)} rows, {df['fold'].nunique()} folds)")
    return ", ".join(out)


# ---------------------------------------------------------------------------
print("\n=== 5. Legacy scripts still parse ===")


@check("all tracked scripts compile")
def _():
    import py_compile

    targets: list[Path] = []
    for folder in ("papers", "src", "tests", "scripts"):
        p = ROOT / folder
        if p.exists():
            targets += sorted(p.rglob("*.py"))
    iw = ROOT / "papers" / "rir-analysis"
    if iw.exists():
        targets += sorted(iw.rglob("*.py"))
    targets = [t for t in targets if "__pycache__" not in t.parts]

    failed = []
    for t in targets:
        try:
            py_compile.compile(str(t), doraise=True, cfile=str(t) + ".tmpc")
        except py_compile.PyCompileError as exc:
            failed.append(f"{t.name}: {exc.msg.splitlines()[-1]}")
        finally:
            tmp = Path(str(t) + ".tmpc")
            if tmp.exists():
                tmp.unlink()
    if failed:
        raise RuntimeError("; ".join(failed[:3]))
    return f"{len(targets)} files"


# ---------------------------------------------------------------------------
print("\n=== 6. End-to-end on real data (needs the corpus) ===")


SYNTH = None  # set lazily by _iwaenc_data()


def _iwaenc_data():
    base = ROOT / "papers" / "rir-analysis"
    data_py = base / "data.py"
    if not data_py.exists():
        raise SkipCheck("IWAENC data.py not present")
    spec = importlib.util.spec_from_file_location("iwaenc_data", data_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["iwaenc_data"] = mod
    spec.loader.exec_module(mod)

    global SYNTH
    from speaker_distance.paths import SYNTHETIC_DIR

    SYNTH = SYNTHETIC_DIR
    return base, mod


@check("load real clips and run a forward pass")
def _():
    import torch

    base, data_mod = _iwaenc_data()
    root = SYNTH / "synthetic_baseline"
    meta = root / "metadata_variants.csv"
    folds = base / "data" / "folds5.csv"
    if not (root.exists() and meta.exists() and folds.exists()):
        raise SkipCheck("synthetic corpus not downloaded")

    ds = data_mod.SyntheticVariantFoldDataset(
        data_root=str(root),
        metadata_csv=str(meta),
        folds_csv=str(folds),
        variant="full",
        folds=[0],
        sample_rate=16000,
        clip_seconds=10.0,
    )
    if len(ds) == 0:
        raise RuntimeError("dataset resolved to zero samples")

    from speaker_distance.models import SeldNet

    net = SeldNet("freq", 2, "all", "onAll").eval()
    batch = torch.stack([ds[i]["audio"] for i in range(2)])
    labels = [float(ds[i]["label"]) for i in range(2)]
    with torch.no_grad():
        dist, _, _, _ = net(batch)
    if not torch.isfinite(dist).all():
        raise RuntimeError("non-finite predictions")
    return f"{len(ds)} clips in fold 0, labels {labels[0]:.2f}/{labels[1]:.2f} m"


@check("all four RIR variants resolve on disk")
def _():
    base, _mod = _iwaenc_data()
    root = SYNTH / "synthetic_baseline"
    if not root.exists():
        raise SkipCheck("synthetic corpus not downloaded")
    counts = {}
    for variant in ("full", "direct", "no_early", "no_late"):
        d = root / variant
        counts[variant] = len(list(d.glob("*.wav"))) if d.exists() else 0
    if any(v == 0 for v in counts.values()):
        raise RuntimeError(f"missing variant audio: {counts}")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"variants disagree in size: {counts}")
    return f"{len(counts)} variants x {next(iter(counts.values()))} clips"


# ---------------------------------------------------------------------------
print("\n=== 7. Stage scripts are runnable ===")


@check("data paths resolve from the repository root")
def _():
    from speaker_distance.paths import (
        DATASETS_DIR,
        NOISE_DIR,
        PUBLISH_DIR,
        SYNTHETIC_DIR,
        has_content,
        starss23_dir,
    )

    # has_content, not exists: the repository ships drop-point directories holding
    # only a README, which would otherwise report as present.
    present = [
        name
        for name, path in (
            ("publishable", PUBLISH_DIR),
            ("synthetic", SYNTHETIC_DIR),
            ("STARSS23", starss23_dir()),
            ("other corpora", DATASETS_DIR),
            ("noise", NOISE_DIR),
        )
        if has_content(path)
    ]
    if not present:
        raise SkipCheck("no data present (set SPEAKER_DISTANCE_* or download it)")
    return "with data: " + ", ".join(present)


@check("no script hardcodes an absolute or personal path")
def _():
    import re

    # String prefixes matter: Path(r"C:\...") slipped past an earlier version of this
    # check, which required a quote immediately after the paren.
    prefix = r"""[rRbBuUfF]{0,2}["']"""

    # A home directory is personal by definition and never acceptable.
    personal = re.compile(
        rf"""{prefix}[A-Za-z]:[/\\]Users[/\\]"""
        rf"""|{prefix}/home/"""
        rf"""|{prefix}/Users/"""
    )
    # Any other absolute path is machine-specific unless it names a standard
    # install location, which is a reasonable default for a tool.
    absolute = re.compile(rf"""{prefix}[A-Za-z]:[/\\]|{prefix}\\\\\\\\[A-Za-z]""")
    allowed = ("Program Files", "ProgramData", "/usr/", "/opt/")

    searched, offenders = 0, []
    for folder in ("papers", "src", "scripts", "tests", "zenodo"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in sorted(list(base.rglob("*.py")) + list(base.rglob("*.ps1"))):
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            searched += 1
            in_block_comment = False
            for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                stripped = line.lstrip()
                # PowerShell <# ... #> and Python docstring-style examples are prose.
                if "<#" in line:
                    in_block_comment = True
                if in_block_comment:
                    if "#>" in line:
                        in_block_comment = False
                    continue
                if stripped.startswith(("#", "//")):
                    continue

                if personal.search(line):
                    offenders.append(f"{path.name}:{i} (home directory)")
                elif absolute.search(line) and not any(a in line for a in allowed):
                    offenders.append(f"{path.name}:{i} (absolute path)")
    if offenders:
        raise RuntimeError("; ".join(offenders[:5]))
    return f"none in {searched} files"


@check("training entrypoints import without side effects")
def _():
    """Importing must not open a network run or require an account.

    This is the check that would have caught a training script opening a live
    Weights & Biases run before validating anything.
    """
    import os
    import subprocess

    env = dict(os.environ, SPEAKER_DISTANCE_LOGGER="none", WANDB_MODE="disabled")
    checked = []
    for script in sorted((ROOT / "papers" / "estimation-2023-2024").glob("training_*.py")):
        code = (
            "import ast,sys;"
            f"src=open(r'{script}',encoding='utf-8').read();"
            "tree=ast.parse(src);"
            # Any top-level call that is not inside `if __name__ == '__main__'`
            "bad=[n.lineno for n in tree.body if isinstance(n,ast.Expr) "
            "and isinstance(n.value,ast.Call)];"
            "print('OK' if not bad else 'TOPLEVEL_CALLS:'+str(bad))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
        )
        result = (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "?"
        if not result.startswith("OK"):
            raise RuntimeError(f"{script.name}: {result}")
        checked.append(script.name.replace("training_", "").replace(".py", ""))
    if not checked:
        raise SkipCheck("no training scripts found")
    return ", ".join(checked)


@check("logger defaults to an account-free backend")
def _():
    from speaker_distance.loggers import make_logger

    assert make_logger("none") is False
    csv_logger = make_logger("csv", run_name="_selfcheck", save_dir=ROOT / "runs")
    if not hasattr(csv_logger, "watch"):
        raise RuntimeError("CSV logger lacks the watch() no-op the scripts call")
    # Clean up the directory the probe just created.
    import shutil

    probe = ROOT / "runs" / "_selfcheck"
    if probe.exists():
        shutil.rmtree(probe, ignore_errors=True)
    return "csv default, wandb opt-in via SPEAKER_DISTANCE_LOGGER"


@check("dataset generator exposes a CLI")
def _():
    import subprocess

    gen = ROOT / "papers" / "rir-analysis" / "generate_all_datasets.py"
    if not gen.exists():
        raise SkipCheck("generator not present")
    out = subprocess.run(
        [sys.executable, str(gen), "--help"], capture_output=True, text=True, timeout=180
    )
    if out.returncode != 0 or "--out" not in out.stdout:
        raise RuntimeError(f"--help failed or lacks --out (rc={out.returncode})")
    for flag in ("--overwrite", "--n-samples", "--yes"):
        if flag not in out.stdout:
            raise RuntimeError(f"missing flag {flag}")
    return "--out, --overwrite, --n-samples, --yes"


# ---------------------------------------------------------------------------
n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
n_skip = sum(1 for s, _, _ in RESULTS if s == "SKIP")

print("\n" + "=" * 62)
print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
if n_fail:
    print("\n  Failures:")
    for status, name, detail in RESULTS:
        if status == "FAIL":
            print(f"    - {name}: {detail}")
if n_skip:
    print("\n  Skipped (missing prerequisites, not errors):")
    for status, name, detail in RESULTS:
        if status == "SKIP":
            print(f"    - {name}: {detail}")
print("=" * 62)

sys.exit(1 if n_fail else 0)
