# Do not move these imports, the order seems to matter
import torch
import pytorch_lightning as pl

import os
import warnings
import pathlib
import pickle
import sys
from pathlib import Path

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from midi.datasets import qm9_dataset, geom_dataset
from midi.diffusion_model import FullDenoisingDiffusion


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, dataset_infos, train_smiles, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos,
                                                        train_smiles=train_smiles)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    if dataset_config.name in ['qm9', "geom"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        else:
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

        train_smiles = list(datamodule.train_dataloader().dataset.smiles) if cfg.general.test_only else []

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.test_only, test=True)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        print("Resuming from {}".format(cfg.general.resume))
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.resume, test=False)

    # save the dataset infos object
    cwd = Path.cwd()
    midi_dir = cwd.parent.parent.parent
    dataset_infos_dir = midi_dir / 'model_reconstruction'
    dataset_infos_dir.mkdir(exist_ok=True)
    print('saving dataset info to: ', dataset_infos_dir.resolve())
    dataset_infos_file = dataset_infos_dir / f'{dataset_config.name}_infos.pkl'
    with open(dataset_infos_file, 'wb') as f:
        pickle.dump(dataset_infos, f)


if __name__ == '__main__':
    main()
