import argparse
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from nbd.data_module import NeuralBDDataModule
from nbd.nbd import NEURALBDModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--reload', action='store_true')
    args, overwrite_args = parser.parse_known_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    base_dir = config['base_dir']
    os.makedirs(base_dir, exist_ok=True)

    # Init Dataset
    save_path = os.path.join(base_dir, 'data_module.pl')
    data_config = config['data']
    if os.path.exists(save_path) and not args.reload:
        data_module = torch.load(save_path, weights_only=False)
    else:
        data_module = NeuralBDDataModule(**data_config)
        torch.save(data_module, save_path)

    # setup training config
    training_config = config['training']
    epochs = training_config['epochs'] if 'epochs' in training_config else 10000
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # Wandb Logger
    logging_config = config['logging']
    logger = WandbLogger(**logging_config)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize NeuralBD model

    if data_config['type'] == 'GREGOR':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'], weights=data_module.contrast_weights, speckle=data_module.speckle,
            sampling=data_config['psf_type'],
            **config['model'])

    elif data_config['type'] == 'MURAM':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'],
            muram=data_module.muram, psf=data_module.psfs,
            sampling=data_config['psf_type'],
            **config['model'])
    else:
        raise ValueError('Unknown data type')

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Train
    N_GPUS = torch.cuda.device_count()
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,
                      check_val_every_n_epoch=20,
                      callbacks=[lr_monitor])
    trainer.fit(neuralbd, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_dir, 'final.ckpt'))
