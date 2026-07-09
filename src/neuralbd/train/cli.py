import argparse
import gc
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from neuralbd.data import BurstDataModule, BurstDataset, BurstPointDataset, load_burst_from_config
from neuralbd.train.callbacks import (
    NeuralBDCheckpointCallback,
    NeuralBDOutputCallback,
    ProgressiveTrainingCallback,
    progressive_sampling_points,
    progressive_validation_sampling_points,
)
from neuralbd.train.config import load_config
from neuralbd.train.module import NeuralBDModule
from neuralbd.train.pretrain import fit_pretraining_stage


def train_epoch_size(config):
    progressive_cfg = config["training"].get("progressive", {})
    if not progressive_cfg.get("enabled", False) or not progressive_cfg.get("fixed_epoch_size", True):
        return None
    epoch_iterations = progressive_cfg.get("epoch_iterations")
    if epoch_iterations is None:
        return None
    return int(epoch_iterations)


def build_datamodule(config):
    data_cfg = config["data"]
    images = load_burst_from_config(data_cfg)
    progressive_cfg = config["training"].get("progressive", {})
    registration_cfg = config["model"].get("registration", {})
    sample_frames = registration_cfg.get("enabled", False) and registration_cfg.get("sample_frames", True)
    sample_frames = sample_frames and not config.get("pretraining", {}).get("enabled", False)
    train_sampling_points = progressive_sampling_points(config)
    target_psf_size = progressive_cfg.get("target_psf_size", config["model"]["psf"]["size"][0])
    initial_validation_psf_size = progressive_cfg.get("start_psf_size", 1)
    if not progressive_cfg.get("enabled", False):
        initial_validation_psf_size = target_psf_size
    validation_sampling_points = progressive_validation_sampling_points(config, psf_size=initial_validation_psf_size)
    epoch_size = train_epoch_size(config)
    train_dataset = BurstDataset.from_numpy(
        images,
        pixel_per_ds=data_cfg["pixel_per_ds"],
        batch_size=train_sampling_points,
        shuffle=True,
        normalization=data_cfg["normalization"],
        duplicate_channels=data_cfg["duplicate_channels"],
        epoch_size=epoch_size,
        sample_frames=sample_frames,
    )
    valid_dataset = BurstPointDataset.from_numpy(
        images,
        pixel_per_ds=data_cfg["pixel_per_ds"],
        normalization=data_cfg["normalization"],
        duplicate_channels=data_cfg["duplicate_channels"],
    )
    return BurstDataModule(
        train_dataset,
        valid_dataset,
        num_workers=data_cfg["num_workers"],
        validation_batch_size=validation_sampling_points,
    )


def build_logger(config):
    logging_cfg = config["logging"]
    if not logging_cfg.get("wandb", False):
        return False
    wandb_dir = Path(config["work_dir"]) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return WandbLogger(
        project=logging_cfg["project"],
        name=logging_cfg["name"],
        save_dir=wandb_dir,
        dir=wandb_dir,
    )


def resume_checkpoint_path(config, base_dir):
    trainer_cfg = config["training"]
    explicit = trainer_cfg.get("resume_from_checkpoint")
    if explicit in {False, "false", "False", "none", "None", ""}:
        return None
    if explicit == "last":
        path = Path(base_dir) / "last.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"No last checkpoint found at {path}")
        return path
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"training.resume_from_checkpoint does not exist: {path}")
        return path
    if not trainer_cfg.get("resume", True):
        return None
    path = Path(base_dir) / "last.ckpt"
    return path if path.exists() else None


def checkpoint_psf_size(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    psf_coords = state_dict.get("convolution.psf_coords")
    if psf_coords is None:
        return None
    return tuple(int(size) for size in psf_coords.shape[:2])


def prepare_module_for_resume(module, checkpoint_path):
    psf_size = checkpoint_psf_size(checkpoint_path)
    if psf_size is None:
        return
    module.set_psf_size(psf_size)
    print(f"Initialized model PSF support from checkpoint: {psf_size[0]}x{psf_size[1]}", flush=True)


def enable_registration_frame_sampling(datamodule, config):
    registration_cfg = config["model"].get("registration", {})
    sample_frames = registration_cfg.get("enabled", False) and registration_cfg.get("sample_frames", True)
    datamodule.set_train_sample_frames(sample_frames)


def train(config_path):
    config = load_config(config_path)
    base_dir = Path(config["base_dir"])
    work_dir = Path(config["work_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resume_checkpoint_path(config, base_dir)
    if ckpt_path is not None:
        print(f"Resuming training from checkpoint: {ckpt_path}", flush=True)

    datamodule = build_datamodule(config)
    module = NeuralBDModule.from_config(config, images_shape=datamodule.images_shape)
    if ckpt_path is not None:
        prepare_module_for_resume(module, ckpt_path)
    logger = build_logger(config)
    if ckpt_path is None:
        pretrainer = fit_pretraining_stage(module, datamodule, config, logger=logger)
        if pretrainer is not None:
            del pretrainer
            gc.collect()
    enable_registration_frame_sampling(datamodule, config)

    checkpoint = ModelCheckpoint(
        dirpath=base_dir,
        every_n_epochs=config["training"]["checkpoint_every_n_epochs"],
        save_last=True,
    )
    callbacks = []
    if config["training"]["progressive"].get("enabled", False):
        callbacks.append(ProgressiveTrainingCallback(config, resume=ckpt_path is not None))
    output_callback = NeuralBDOutputCallback(config=config, sample_count=config["outputs"]["sample_count"])
    neuralbd_checkpoint = NeuralBDCheckpointCallback(base_dir=base_dir, config=config)
    callbacks.extend([checkpoint, neuralbd_checkpoint, output_callback])
    if config["logging"].get("wandb", False):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    torch.set_float32_matmul_precision("medium")
    trainer_kwargs = {}
    if "strategy" in config["training"]:
        trainer_kwargs["strategy"] = config["training"]["strategy"]

    trainer = Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        logger=logger,
        log_every_n_steps=config["training"]["log_every_n_steps"],
        num_sanity_val_steps=config["training"]["num_sanity_val_steps"],
        check_val_every_n_epoch=config["training"]["validation_every_n_epochs"],
        reload_dataloaders_every_n_epochs=1,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    trainer.fit(module, datamodule, ckpt_path=str(ckpt_path) if ckpt_path is not None else None)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a NeuralBD reconstruction.")
    parser.add_argument("--config", required=True, help="Path to a NeuralBD YAML configuration.")
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main()
