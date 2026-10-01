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
from datetime import datetime
from synthetic import SyntheticDatasetModule
from model import SeldTrainer
import pandas as pd

if __name__ == "__main__":
    # PATHS
    file_path = "dataset_wav/"
    distance_files = "dataset_wav/distances.mat"
    permutation = "dataset_wav/transformed_indeces.mat"
    pathNoisesTraining = "noise_dataset/training_noise"
    pathNoisesVal = "noise_dataset/val_noise"
    pathNoisesTest = "noise_dataset/test_noise"
    # FIXED PARAMS
    config = {
        "max_epochs": 50,
        "batch_size": 16,
        "lr": 0.001,
        "sampling_frequency": 16000,
        "dBNoise" : None, # None for noiseless, [50, 40, 30, 20, 10, 5, 0] for selecting the dBs
        "kernels": "freq",
        "n_grus": 2,
        "features_set": "all",
        "att_conf": ["Nothing", "onSpec", "onAll"]
    }

    if config["dBNoise"] == None:
        for conf in config["att_conf"]:
                all_results = pd.DataFrame([], columns = ["GT", "Pred", "L1", "rL1", "ID"])
                run_name = "Kernels{}_Gru{}_Features_{}Att_conf{}".format(config['kernels'], config['n_grus'], config['features_set'], conf )

                for i in range(0, 5):
                    n = datetime.now()
                    foldTraining = list(range(0, 5))
                    foldValidation = i
                    foldTest = (i + 1) % 5
                    wandb_logger = WandbLogger(
                            project="Distance-Estimation-RQ1",
                            name="{} Fold {}".format(run_name, foldTest),
                            tags=["Clean",
                                "Synt", "TABLE2"],
                        )
                    wandb_logger.log_hyperparams(config)
                    foldTraining.remove(foldValidation)
                    foldTraining.remove(foldTest)  
                    trainer = Trainer(
                            accelerator="gpu",
                            devices = 1,
                            log_every_n_steps=50,
                            max_epochs=config["max_epochs"],
                            precision=32,
                            logger=wandb_logger,
                    )
                        # define datamodule with i as the test fold
                    data_module = SyntheticDatasetModule(
                                file_path,
                                distance_files,
                                foldTraining,
                                [foldValidation],
                                [foldTest],
                                batch_size=config["batch_size"],
                                permutation=permutation,
                                fs=config["sampling_frequency"],
                                transform=None,
                        )
                    model = SeldTrainer(lr=config["lr"], kernels = config['kernels'], n_grus = config['n_grus'], features_set = config['features_set'], att_conf = conf)
                    print("Start Training on {} folds with validation on [{}] and test on [{}].".format(foldTraining, foldValidation, foldTest))
                    wandb_logger.watch(model, log_graph=False)
                    # start training the model on the training folds
                    trainer.fit(model, datamodule=data_module)
                    trainer.test(model, datamodule=data_module)
                    wandb.finish()
                    all_results = pd.concat([all_results, pd.DataFrame(model.all_test_results)], ignore_index = True)
                # here we finished all the tests, save the results
                all_results.to_csv(run_name + ".csv")
#############################################################################
    else:  # Noisy training from scratch
        for db in config["dBNoise"]:
            for att_conf in config["att_conf"]:
                all_results = pd.DataFrame([], columns = ["GT", "Pred", "L1", "rL1", "ID"])
                run_name = "Kernels{}_Gru{}_Features_{}Att_conf{}_dB{}".format(config['kernels'], config['n_grus'], config['features_set'], att_conf, db)
                for i in range(0, 5):
                    n = datetime.now()
                    foldTraining = list(range(0, 5))
                    foldValidation = i
                    foldTest = (i + 1) % 5
                    wandb_logger = WandbLogger(
                        project="Distance-Estimation-RQ1",
                        name="{} Fold {} Noisy {}dB".format(run_name, foldTest, db),
                        tags=["{}dB".format(db), "Synt"],
                    )
                    wandb_logger.log_hyperparams(config)
                    foldTraining.remove(foldValidation)
                    foldTraining.remove(foldTest)  
                    trainer = Trainer(
                        accelerator="gpu",
                        devices = 1,
                        log_every_n_steps=50,
                        max_epochs=config["max_epochs"],
                        precision=32,
                        logger=wandb_logger,
                    )
                    data_module = SyntheticDatasetModule(
                        file_path,
                        distance_files,
                        foldTraining,
                        [foldValidation],
                        [foldTest],
                        batch_size=config["batch_size"],
                        permutation=permutation,
                        fs=config["sampling_frequency"],
                        transform=None,
                        pathNoisesTraining=pathNoisesTraining,
                        pathNoisesVal=pathNoisesVal,
                        pathNoisesTest=pathNoisesTest,
                        dBNoise=db,
                    )
                    model = SeldTrainer(lr=config["lr"], kernels = config['kernels'], n_grus = config['n_grus'], features_set = config['features_set'], att_conf = att_conf)
                    print("Start Training on {} folds with validation on [{}] \
                            and test on [{}].".format(foldTraining, foldValidation, foldTest))
                    wandb_logger.watch(model, log_graph=False)
                    # start training the model on the training folds
                    trainer.fit(model, datamodule=data_module)
                    trainer.test(model, datamodule=data_module)
                    wandb.finish()
                    all_results = pd.concat([all_results, pd.DataFrame(model.all_test_results)], ignore_index = True)
                    # here we finished all the tests, save the results
                all_results.to_csv(run_name + ".csv")
