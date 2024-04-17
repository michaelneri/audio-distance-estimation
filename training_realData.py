from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
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