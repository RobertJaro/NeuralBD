import collections.abc

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import numpy as np
from astropy.io import fits
from lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset

from nstack.data.editor import get_KL_basis, get_PSFs, get_convolution, get_KL_wavefront

from nstack.data.editor import ReadSimulationEditor
from nstack.data.psfs import *

from nstack.model import PSFStackModel


class NEURALBDModule(LightningModule):

        def __init__(self, n_images, dim=512, learning_rate=1e-3, n_modes=44, psf_size=128, muram=True, **kwargs):
            super().__init__()
            self.n_images = n_images
            self.dim = dim
            self.learning_rate = learning_rate
            self.n_modes = n_modes
            self.psf_size = psf_size

            self.kl_basis = get_KL_basis(n_modes_max=self.n_modes, size=self.psf_size)
            kl_wavefront = get_KL_wavefront(self.kl_basis, self.n_modes, self.n_images, coef_range=2)
            self.psfs = get_PSFs(kl_wavefront, self.n_images)

            if muram:
                muram = ReadSimulationEditor().call('/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000')
                sim_array = muram.data
                vmin, vmax = 0, np.percentile(sim_array, 99)
                fits_array2_norm = (sim_array - vmin) / (vmax - vmin)
                fits_array2_stack = np.stack([fits_array2_norm, fits_array2_norm], -1)
                self.high_quality = fits_array2_stack[250:762, 250:762, :]
                #self.muram = fits_array2_stack
                self.images = get_convolution(self.high_quality, self.psfs, self.n_images)
                self.im_scaling = (sim_array.shape[0] - 1) / 2
            else:
                fits_array = []
                data_path = '/gpfs/data/fs71254/schirni/nstack/data/hifi_20170618_082414_sd.fts'
                data_path2 = '/gpfs/data/fs71254/schirni/nstack/data/hifi_20170618_082414_sd_speckle.fts'
                #
                for i in range(0, 200): # <-- 2022: 0, 200 ; 2022-->: 1,201
                    fits_array.append(fits.getdata(data_path, i))
                #
                h, w = fits_array[0].shape
                fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))
                #
                fits_array2 = []
                for i in range(0, 2):
                   fits_array2.append(fits.getdata(data_path2, i))
                fits_array2 = np.stack(fits_array2, -1)
                #
                fits_array = fits_array[:, :, :self.n_images]
                ##fits_array = fits_array[200:1224, 200:1224, :, :]
                fits_array2 = fits_array2[600:1112, 600:1112, :]
                fits_array = fits_array[600:1112, 600:1112, :, :]
                vmin, vmax = fits_array.min(), np.percentile(fits_array, 99)
                vmin2, vmax2 = fits_array2.min(), fits_array2.max()
                #x
                self.images = (fits_array - vmin) / (vmax - vmin)
                self.high_quality = (fits_array2 - vmin2) / (vmax2 - vmin2)

                self.im_scaling = (self.images.shape[0] - 1) / 2

            self.valid_loss = []
            self.train_loss = []

            self.model = PSFStackModel(self.n_images, self.dim, self.n_modes, self.psf_size, self.kl_basis, self.im_scaling,
                                       self.images, self.high_quality, self.psfs)

            #self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            batch_img, coord_batch = batch
            pred_img, convolved_imgs, psfs, target_imgs, ref_img, ref_psfs = self.model(coord_batch)

            loss_fn = torch.nn.HuberLoss()
            image_loss = loss_fn(convolved_imgs, batch_img)
            self.train_loss.append(image_loss.detach().cpu())
            self.log('train_loss', image_loss)
            return image_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
            return optimizer

        def validation_step(self, batch, batch_idx):
            batch_img, coord_batch = batch
            pred_img, convolved_imgs, psfs, target_imgs, ref_img, ref_psfs = self.model(coord_batch)

            loss_fn = torch.nn.HuberLoss()
            image_loss = loss_fn(convolved_imgs, batch_img)

            self.valid_loss.append(image_loss.detach().cpu())
            self.log('val_loss', image_loss)
            return image_loss

        def train_dataloader(self):
            images = self.images
            image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
            coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)
            coordinates_tensor = coordinates_tensor / self.im_scaling
            image_tensor = torch.from_numpy(images).float().reshape(-1, self.n_images, 2)
            r = torch.randperm(len(image_tensor))
            image_tensor = image_tensor[r]
            coordinates_tensor = coordinates_tensor[r]
            data_set = TensorDataset(image_tensor, coordinates_tensor)

            return DataLoader(data_set, batch_size=4096, num_workers=4)

        def val_dataloader(self):
            images = self.images
            image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
            coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)
            coordinates_tensor = coordinates_tensor / self.im_scaling
            image_tensor = torch.from_numpy(images).float().reshape(-1, self.n_images, 2)
            test_coords = coordinates_tensor
            data_set = TensorDataset(image_tensor, test_coords)

            return DataLoader(data_set, batch_size=4096, num_workers=4)