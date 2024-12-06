import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from nstack.callback import PlotNeuralBDCallback
from nstack.nbd import NEURALBDModule

wandb_logger = WandbLogger(project='NeuralBD', entity='christoph-schirninger',
                           name='neuralbd_gregor', offline=False, dir='/gpfs/data/fs71254/schirni/nstack/Training')

module = NEURALBDModule(n_images=2, dim=512, learning_rate=1e-3, n_modes=44, psf_size=9, muram=False)

image_coordinates = np.stack(np.mgrid[:512, :512], -1)
coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)
coordinates_tensor = coordinates_tensor / 511.5
test_coords = coordinates_tensor

plot_callbacks = []
plot_callbacks += [PlotNeuralBDCallback(2, test_coords)]
#lr_monitor = LearningRateMonitor(logging_interval='step')

n_gpus = torch.cuda.device_count()
trainer = Trainer(max_epochs=10000,
                  devices=n_gpus if n_gpus > 0 else None,
                  logger=wandb_logger,
                  strategy='dp' if n_gpus > 1 else None,
                  accelerator="gpu" if n_gpus >= 1 else None,
                  num_sanity_val_steps=-1,
                  callbacks=plot_callbacks,
                  gradient_clip_val=1e-3,
                  )
trainer.fit(module)
