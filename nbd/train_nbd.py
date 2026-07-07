import argparse
import os
import sys
import numpy as np

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, LambdaCallback, ModelCheckpoint
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


    def _register_numpy_pickle_aliases():
        # NumPy 2 pickles may reference numpy._core, while NumPy 1 exposes numpy.core.
        if not hasattr(np, "_core"):
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
            sys.modules.setdefault("numpy._core.umath", np.core.umath)

    # Init Dataset
    save_path = os.path.join(base_dir, 'data_module.pl')
    data_config = config['data']
    if os.path.exists(save_path) and not args.reload:
        _register_numpy_pickle_aliases()
        data_module = torch.load(save_path, weights_only=False)
    else:
        data_module = NeuralBDDataModule(**data_config)
        torch.save(data_module, save_path)

    # setup training config
    training_config = config['training']
    epochs = training_config['epochs'] if 'epochs' in training_config else 10000
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'
    model_config = dict(config['model'])
    if 'optimizer_config' not in model_config:
        if 'optimizer_config' in training_config:
            model_config['optimizer_config'] = training_config['optimizer_config']
        elif 'optimizer' in training_config:
            model_config['optimizer_config'] = {'name': training_config['optimizer']}

    # Wandb Logger
    logging_config = config['logging']
    logger = WandbLogger(**logging_config)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize NeuralBD model

    if data_config['type'] == 'GREGOR':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'], weights=data_module.contrast_weights, speckle=data_module.speckle,
            sampling=data_config['psf_type'], psf_type=data_config['psf_type'],
            save_path=base_dir,
            **model_config)

    elif data_config['type'] == 'MURAM':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'],
            muram=data_module.muram, psf=data_module.psfs,
            sampling=data_config['psf_type'],
            psf_type=data_config['psf_type'],
            save_path=base_dir,
            **model_config)

    elif data_config['type'] == 'DKIST':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'],
            sampling=data_config['psf_type'], psf_type=data_config['psf_type'],
            save_path=base_dir,
            **model_config)

    elif data_config['type'] == 'KSO':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'],
            sampling=data_config['psf_type'],
            psf_type=data_config['psf_type'],
            save_path=base_dir,
            **model_config)

    elif data_config['type'] == 'SUIT':
        neuralbd = NEURALBDModule(
            images_shape=[data_config['crop_size'], data_config['crop_size'], data_config['n_images'], 2],
            pixel_per_ds=data_config['pixel_per_ds'],
            sampling=data_config['psf_type'],
            psf_type=data_config['psf_type'],
            save_path=base_dir,
            **model_config)

    else:
        raise ValueError('Unknown data type')

    if config['meta_state'] == 'loadme':
        meta_model_path = config['base_dir']+'/meta_model.pth'
        _register_numpy_pickle_aliases()
        meta_ckpt = torch.load(meta_model_path, map_location='cpu')
        neuralbd.image_model.load_state_dict(meta_ckpt)
        print(f"-------------------------- Loaded meta model from {meta_model_path} --------------------------")
    elif config['meta_state'] == 'none':
        print("-------------------------- Continuing without loading meta model --------------------------")
    else:
        raise ValueError('Unknown meta_state option')

    checkpoint_callback = ModelCheckpoint(dirpath=base_dir,
                                          every_n_epochs=training_config['checkpoint_every_n_epochs'] if 'checkpoint_every_n_epochs' in training_config else 5,
                                          save_last=True)

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # save callback
    save_path = os.path.join(base_dir, 'neuralbd.nbd')

    def save(*args, **kwargs):
        if data_config['psf_type'] == 'varying':
            torch.save({
                'image_model': neuralbd.image_model,
                'psf_model': neuralbd.psf_model,
                'image_coords': data_module.img_coords,
            }, save_path)
        else:
            torch.save({
                'image_model': neuralbd.image_model,
                'image_coords': data_module.img_coords,
            }, save_path)


    save_callback = LambdaCallback(on_validation_epoch_end=save)

    # Train
    torch.set_float32_matmul_precision('medium')
    N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,
                      check_val_every_n_epoch=10,
                      callbacks=[lr_monitor, checkpoint_callback, save_callback],)
    trainer.fit(neuralbd, data_module, ckpt_path=ckpt_path)
    #trainer.save_checkpoint(os.path.join(base_dir, 'final.ckpt'))
