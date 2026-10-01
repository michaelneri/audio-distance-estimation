# --- runnability shim -------------------------------------------------------
# Logging is opt-in: the default writes CSV and needs no account. Set
# SPEAKER_DISTANCE_LOGGER=wandb to restore Weights & Biases. Data locations come
# from the repository root, so the working directory does not matter.
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from speaker_distance.loggers import make_logger as _make_logger, finish as _finish
from speaker_distance.paths import REPO_ROOT
_BACKEND = _os.environ.get("SPEAKER_DISTANCE_LOGGER", "csv")
if __name__ == "__main__":
    # These scripts use paths relative to the repository root. Only applied when
    # run directly, so importing the module has no side effects.
    _os.chdir(REPO_ROOT)


def WandbLogger(project=None, name=None, tags=None, **_kw):
    return _make_logger(_BACKEND, run_name=name, project=project or "speaker-distance", tags=tags)


class _WandbShim:
    @staticmethod
    def finish():
        _finish()


wandb = _WandbShim()
# --- end shim ---------------------------------------------------------------
from pytorch_lightning import Trainer
from model import SeldTrainer
import torch
import pandas as pd
from VoiceHome import VHDataModule
from STARS23 import STARS23DataModule

if __name__ == "__main__":
    # VoiceHome-2
    path_annotations = 'real datasets/VoiceHome2/voiceHome-2_corpus_1.0/annotations/rooms'
    path_audios = 'real datasets/VoiceHome2/voiceHome-2_corpus_1.0/audio/noisy'
    file_npz = "VoiceHome2_splitted.npz"

    # STARS23
    path_audios_starss = 'real datasets/STARS23'

    # FIXED PARAMS
    config = {
        "max_epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "sampling_frequency": 16000,
        "dBNoise" : None,
        "kernels": "freq",
        "n_grus": 2,
        "features_set": ["sincos", "stft"],
        "att_conf": "onAll"
    }

    # FIRST VOICEHOME
    for conf in config["features_set"]:
        run_name = "Kernels{}_Gru{}_Features_{}Att_conf{}_VOICEHOME".format(config['kernels'], config['n_grus'], conf, config['att_conf'])
        model = SeldTrainer(lr=config["lr"], kernels = config['kernels'], n_grus = config['n_grus'], features_set = conf, att_conf = config['att_conf'])
        datamodule = VHDataModule(file_npz, path_annotations, path_audios, batch_size = config['batch_size'])
        wandb_logger = WandbLogger(
                                        project="Distance-Estimation-RQ1",
                                        name="{}".format(run_name),
                                        tags=["TABLE7", "Real", "Voicehome"],
                                    )
        trainer = Trainer(
                                        accelerator="gpu",
                                            devices = 1,
                                            log_every_n_steps = 50,
                                            max_epochs=config["max_epochs"],
                                            precision = 32,
                                            logger=wandb_logger,
                                        )
        wandb_logger.log_hyperparams(config)
        wandb_logger.watch(model, log_graph=False)
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        wandb.finish()
        all_results = pd.DataFrame(model.all_test_results)
        all_results.to_csv(run_name + ".csv")

    # THEN STARSS23
    for conf in config["features_set"]:
        run_name = "Kernels{}_Gru{}_Features_{}Att_conf{}_STARSS23".format(config['kernels'], config['n_grus'], conf, config['att_conf'])
        model = SeldTrainer(lr=config["lr"],  kernels = config['kernels'], n_grus = config['n_grus'], features_set = conf, att_conf = config['att_conf'])
        datamodule = STARS23DataModule(path_dataset = path_audios_starss, batch_size = config['batch_size'])
        wandb_logger = WandbLogger(
                                        project="Distance-Estimation-RQ1",
                                        name="{}".format(run_name),
                                        tags=["TABLE8", "Real", "STARSS23"],
                                    )
        trainer = Trainer(
                                            accelerator="gpu",
                                            devices = 1,
                                            log_every_n_steps = 50,
                                            max_epochs=config["max_epochs"],
                                            precision = 32,
                                            logger=wandb_logger,
                                        )
        wandb_logger.log_hyperparams(config)
        wandb_logger.watch(model, log_graph=False)
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        wandb.finish()
        all_results = pd.DataFrame(model.all_test_results)
        all_results.to_csv(run_name + ".csv")