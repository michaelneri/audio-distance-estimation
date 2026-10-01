"""Single source of truth for where data lives.

Scripts across the three stages previously each resolved corpora relative to their own
file, which broke as soon as the scripts moved. Everything now resolves from the
repository root, with environment variables for people whose data sits elsewhere
(typically on a separate drive, since the corpora total tens of gigabytes).

    from speaker_distance.paths import DATASETS_DIR, NOISE_DIR, SYNTHETIC_DIR

Environment overrides, all optional:

===========================  ===================================================
``SPEAKER_DISTANCE_DATA``    real corpora (QMULTIMIT, STARSS23, VoiceHome2)
``SPEAKER_DISTANCE_NOISE``   WHAM!-derived noise splits
``SPEAKER_DISTANCE_SYNTH``   synthetic corpus (the ``synthetic_*`` folders)
===========================  ===================================================

A script that needs data should call :func:`require` so a missing corpus produces one
clear sentence naming the directory and the variable that overrides it, rather than a
``FileNotFoundError`` a hundred lines later.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATASETS_DIR",
    "NOISE_DIR",
    "SYNTHETIC_DIR",
    "LABELS_DIR",
    "PAPERS_DIR",
    "require",
    "describe",
]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# src/speaker_distance/paths.py -> src/speaker_distance -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

PAPERS_DIR = REPO_ROOT / "papers"
LABELS_DIR = REPO_ROOT / "data" / "labels"

#: Everything this project is allowed to redistribute lives here, assembled ready for
#: upload. It is also where the code reads that data from, so there is one copy, not a
#: working copy and a publishing copy that can drift apart.
PUBLISH_DIR = _env_path("SPEAKER_DISTANCE_PUBLISH", REPO_ROOT / "zenodo" / "data")

#: Corpora that cannot be redistributed (QMULTIMIT, VoiceHome-2) and must be obtained
#: separately. Empty in a fresh clone.
DATASETS_DIR = _env_path("SPEAKER_DISTANCE_DATA", REPO_ROOT / "real datasets")
NOISE_DIR = _env_path("SPEAKER_DISTANCE_NOISE", REPO_ROOT / "noise_dataset")
SYNTHETIC_DIR = _env_path("SPEAKER_DISTANCE_SYNTH", PUBLISH_DIR)

_ENV_FOR = {
    PUBLISH_DIR: "SPEAKER_DISTANCE_PUBLISH",
    DATASETS_DIR: "SPEAKER_DISTANCE_DATA",
    NOISE_DIR: "SPEAKER_DISTANCE_NOISE",
    SYNTHETIC_DIR: "SPEAKER_DISTANCE_SYNTH",
}


def starss23_dir() -> Path:
    """Where the STARSS23 distance subset lives.

    It is redistributable, so it ships in :data:`PUBLISH_DIR`, but a copy placed with
    the other real corpora is honoured too - which is where it sat historically.
    """
    for candidate in (
        PUBLISH_DIR / "STARSS23_distance",
        DATASETS_DIR / "STARS23",
        DATASETS_DIR / "STARSS23",
    ):
        if candidate.exists():
            return candidate
    return PUBLISH_DIR / "STARSS23_distance"


def wham48_dir() -> Path:
    """The 48 kHz WHAM! noise set, used by the 0 dB evaluations.

    This is a separate download from the segmented splits in :data:`NOISE_DIR`: it
    keeps the recordings at their original sample rate and ships its own
    ``high_res_metadata.csv``. Override with ``SPEAKER_DISTANCE_WHAM48``.
    """
    override = os.environ.get("SPEAKER_DISTANCE_WHAM48")
    if override:
        return Path(override).expanduser().resolve()
    return NOISE_DIR / "high_res_wham"


def has_content(path: Path) -> bool:
    """True when ``path`` exists and holds at least one file.

    A directory that exists but is empty is the common failure mode here: the
    repository ships drop-point folders with only a README in them.
    """
    if not path.is_dir():
        return False
    return any(p.is_file() for p in path.rglob("*") if p.name != "README.md")


def require(path: Path, what: str = "dataset") -> Path:
    """Return ``path``, or raise with a message that says how to fix it."""
    if path.exists():
        return path

    env = next((v for k, v in _ENV_FOR.items() if path == k or k in path.parents), None)
    hint = f"\n  Set {env} to point at it, or " if env else "\n  "
    raise FileNotFoundError(
        f"Missing {what}: {path}"
        f"{hint}download it from the Zenodo record linked in the README."
    )


def describe() -> str:
    """Human-readable summary of resolved paths and whether each holds data.

    Reports EMPTY rather than OK for a directory that exists but has no files in it,
    since that is what a fresh clone looks like before the data is downloaded.
    """
    rows = [
        ("repository", REPO_ROOT, True),
        ("publishable data", PUBLISH_DIR, has_content(PUBLISH_DIR)),
        ("synthetic", SYNTHETIC_DIR, has_content(SYNTHETIC_DIR)),
        ("STARSS23", starss23_dir(), has_content(starss23_dir())),
        ("other corpora", DATASETS_DIR, has_content(DATASETS_DIR)),
        ("noise", NOISE_DIR, has_content(NOISE_DIR)),
        ("labels", LABELS_DIR, has_content(LABELS_DIR)),
    ]
    width = max(len(name) for name, _, _ in rows)
    out = []
    for name, path, ok in rows:
        if ok:
            status = "OK     "
        elif path.is_dir():
            status = "EMPTY  "
        else:
            status = "MISSING"
        out.append(f"  {name:<{width}}  {status}  {path}")
    return "\n".join(out)


if __name__ == "__main__":
    print(describe())
