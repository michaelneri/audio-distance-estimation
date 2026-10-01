"""Experiment logging that does not require an account.

The original training scripts called ``WandbLogger`` unconditionally, so running any of
them opened a live Weights & Biases run before a single path had been validated. Anyone
without an account hit a login prompt; anyone with one got a stray run in their
workspace.

Logging is now opt-in. The default writes CSV next to the run, which needs no account,
no network and no configuration::

    from speaker_distance.loggers import make_logger

    logger = make_logger("csv", run_name="no_late_fold0")        # default
    logger = make_logger("wandb", run_name=..., project=..., tags=[...])
    logger = make_logger("none")                                  # disable entirely

Selecting ``wandb`` when it is not installed raises immediately with an actionable
message rather than failing midway through training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = ["make_logger", "BACKENDS"]

BACKENDS = ("csv", "wandb", "none")


class _CSVLogger:
    """CSVLogger plus the few W&B-only methods the training scripts call.

    ``watch`` (gradient tracking) has no CSV equivalent and becomes a no-op, so a
    script written against WandbLogger runs unchanged against this.
    """

    def __new__(cls, *args, **kwargs):
        from pytorch_lightning.loggers import CSVLogger

        instance = CSVLogger(*args, **kwargs)
        if not hasattr(instance, "watch"):
            instance.watch = lambda *a, **k: None
        if not hasattr(instance, "experiment_finish"):
            instance.experiment_finish = lambda *a, **k: None
        return instance


def finish(logger=None) -> None:
    """End a run. Safe to call for any backend, including ``False``."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def make_logger(
    backend: str = "csv",
    *,
    run_name: str | None = None,
    project: str = "speaker-distance",
    save_dir: str | Path = "runs",
    tags: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
):
    """Build a Lightning logger. Returns ``False`` for ``"none"``.

    ``False`` is what Lightning's ``Trainer(logger=...)`` expects for "no logging",
    so the return value can be passed straight through.
    """
    backend = (backend or "csv").lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    if backend == "none":
        return False

    if backend == "csv":
        logger = _CSVLogger(save_dir=str(save_dir), name=run_name or "run")
        if config:
            logger.log_hyperparams(config)
        return logger

    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "wandb logging was requested but Weights & Biases is not installed.\n"
            "  Install it with:  pip install wandb\n"
            "  Or use the default CSV logger, which needs no account."
        ) from exc

    logger = WandbLogger(project=project, name=run_name, tags=list(tags or []))
    if config:
        logger.log_hyperparams(config)
    return logger


def add_logger_argument(parser) -> None:
    """Attach a ``--logger`` option to an ``argparse`` parser."""
    parser.add_argument(
        "--logger",
        choices=BACKENDS,
        default="csv",
        help="experiment logging backend (default: csv, which needs no account)",
    )
