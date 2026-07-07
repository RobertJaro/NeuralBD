import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from neuralbd.data import BurstDataModule, BurstDataset, load_burst_from_config
from neuralbd.train.callbacks import NeuralBDOutputCallback, ProgressiveTrainingCallback
from neuralbd.train.config import load_config
from neuralbd.train.module import NeuralBDModule
from neuralbd.train.pretrain import fit_pretraining_stage


def build_datamodule(config):
    data_cfg = config["data"]
    images = load_burst_from_config(data_cfg)
    train_dataset = BurstDataset.from_numpy(
        images,
        pixel_per_ds=data_cfg["pixel_per_ds"],
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        normalization=data_cfg["normalization"],
        duplicate_channels=data_cfg["duplicate_channels"],
    )
    valid_dataset = BurstDataset.from_numpy(
        images,
        pixel_per_ds=data_cfg["pixel_per_ds"],
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        normalization=data_cfg["normalization"],
        duplicate_channels=data_cfg["duplicate_channels"],
    )
    return BurstDataModule(train_dataset, valid_dataset, num_workers=data_cfg["num_workers"])


def build_logger(config):
    logging_cfg = config["logging"]
    if not logging_cfg.get("wandb", False):
        return False
    return WandbLogger(project=logging_cfg["project"], name=logging_cfg["name"])


def train(config_path):
    config = load_config(config_path)
    base_dir = Path(config["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config)
    module = NeuralBDModule.from_config(config, images_shape=datamodule.images_shape)
    logger = build_logger(config)
    fit_pretraining_stage(module, datamodule, config, logger=logger)

    checkpoint = ModelCheckpoint(
        dirpath=base_dir,
        every_n_epochs=config["training"]["checkpoint_every_n_epochs"],
        save_last=True,
    )
    output_callback = NeuralBDOutputCallback(
        base_dir=base_dir,
        config=config,
        save_validation_arrays=config["outputs"]["save_validation_arrays"],
        save_validation_figures=config["outputs"]["save_validation_figures"],
        sample_count=config["outputs"]["sample_count"],
    )
    callbacks = [checkpoint, output_callback]
    if config["training"]["progressive"].get("enabled", False):
        callbacks.append(ProgressiveTrainingCallback(config))
    if config["logging"].get("wandb", False):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        logger=logger,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        check_val_every_n_epoch=config["training"]["check_val_every_n_epoch"],
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a NeuralBD reconstruction.")
    parser.add_argument("--config", required=True, help="Path to a NeuralBD YAML configuration.")
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main()
