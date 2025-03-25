import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nbd.model import ImageModel


class NEURALBDModule(LightningModule):

    def __init__(self, images_shape, pixel_per_ds, weights, learning_rate=1e-3, psf_size=(29, 29),
                 model_config=None, lr_config=None, speckle=None, sampling='grid', **kwargs):
        super().__init__()
        self.images_shape = images_shape
        self.n_images = self.images_shape[2]
        self.speckle = speckle
        self.sampling = sampling

        self.learning_rate = learning_rate

        if self.sampling == 'grid':
            # Create PSF coordinates for sampling (Grid sampling)
            x_values = torch.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_size[0], dtype=torch.float32)
            y_values = torch.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_size[1], dtype=torch.float32)
            x, y = torch.meshgrid(x_values, y_values, indexing='ij')
            psf_coords = torch.stack([x, y], -1)  # psf_coords, xy
            psf_coords = psf_coords / pixel_per_ds

            self.psf_coords = nn.Parameter(psf_coords, requires_grad=False)

            # Create learnable PSFs
            log_psfs = torch.randn(*psf_size, self.n_images, dtype=torch.float32)
            self.log_psfs = nn.Parameter(log_psfs, requires_grad=True)

            # calculate area element
            area_elemnts = np.ones(psf_size, dtype=np.float32)
            area_elemnts = area_elemnts / (pixel_per_ds ** 2)
            area_elemnts = np.repeat(area_elemnts[..., None], self.n_images, axis=-1)
            self.area_elements = nn.Parameter(torch.tensor(area_elemnts, dtype=torch.float32), requires_grad=False)

        elif self.sampling == 'spherical':
            # Create PSF coordinates for sampling (Spherical sampling)
            max_radius = 10
            r_values = np.linspace(0, 1, 10, dtype=np.float32) * max_radius
            phi_values = np.linspace(0, 2 * torch.pi, 18, endpoint=False, dtype=np.float32)
            # remove r = 0
            phi, r = np.meshgrid(phi_values, r_values[1:], indexing='ij')

            # convert to cartesian
            self.x, self.y = r * np.sin(phi), r * np.cos(phi)
            psf_coords = np.stack([self.x, self.y], -1)  # shape: phi, r, 2

            # add central point
            psf_coords = psf_coords / pixel_per_ds
            self.psf_coords = nn.Parameter(torch.tensor(psf_coords, dtype=torch.float32), requires_grad=False)

            # calculate area element
            dr = np.gradient(r, axis=1)
            dtheta = np.gradient(phi, axis=0)
            area_elements = r * dtheta * dr  # shape: phi, r

            area_elements = area_elements / (pixel_per_ds ** 2)
            area_elements = np.repeat(area_elements[..., None], self.n_images, axis=-1)
            self.area_elements = nn.Parameter(torch.tensor(area_elements, dtype=torch.float32), requires_grad=False)

            # Create learnable PSFs
            log_psfs = np.random.randn(self.x.shape[0], self.x.shape[1], self.n_images).astype(np.float32)
            self.log_psfs = nn.Parameter(torch.tensor(log_psfs, dtype=torch.float32), requires_grad=True)

        else:
            raise ValueError(f'Unknown sampling method: {sampling}')

        # Create and normalize weights
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * 0.8 + 0.2
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)


        # Create image model
        model_config = model_config if model_config is not None else {}
        self.image_model = ImageModel(**model_config)

        # Learning rate scheduler
        self.lr_config = {'start': 1e-3, 'end': 1e-4, 'iterations': 1e3} if lr_config is None else lr_config

    def get_convolved_images(self, coords):
        # create grid of sampling coordinates for PSF
        # coords: batch, 2
        # psf_coords: x, y, 2

        sampling_coords = coords[:, None, :] + self.psf_coords.reshape(1, -1, 2)  # --> batch, psf_coords, 2

        # load the PSF
        # psf: x, y, n_images
        psf = self.get_psf()
        flat_psf = psf.reshape(-1, self.n_images)

        image = self.image_model(sampling_coords.reshape(-1, 2))
        image = image.reshape(coords.shape[0], -1, image.shape[-1])
        area_elements = self.area_elements.reshape(-1, self.n_images)
        # image:  batch, xy(PSF), channels
        # flat_psf: xy(PSF) , n_images
        # convolved_images = torch.einsum('...sc,sn->...nc', image, flat_psf * area_elements)
        convolved_images = torch.einsum('...sc,sn->...nc', image, flat_psf)
        # convolved_images: batch, n_images, channels

        return convolved_images

    def get_shift(self):
        shift = torch.tanh(self.shift) * self.shift_scaling
        return shift

    def get_psf(self):
        psfs = torch.exp(self.log_psfs)
        # norm = (psfs * self.area_elements).sum(dim=(0, 1), keepdims=True)
        norm = psfs.sum(dim=(0, 1), keepdims=True)
        return psfs / (norm + 1e-8)

    def training_step(self, batch, batch_idx):
        convolved_true, coords = batch

        convolved_pred = self.get_convolved_images(coords)

        # Compute loss
        image_loss = self._compute_weighted_image_loss(convolved_pred, convolved_true)

        self.log('loss', image_loss)

        return image_loss

    def _compute_weighted_image_loss(self, convolved_pred, convolved_true):
        convolved_diff = (convolved_pred - convolved_true) ** 2
        # convolved_diff: batch, n_images, channels
        # weight: n_images
        image_loss = (convolved_diff * self.weights[None, :, None]).sum(1) / self.weights.sum()
        image_loss = image_loss.mean()
        #image_loss = torch.mean(convolved_diff)
        return image_loss

    def configure_optimizers(self):
        parameters = list(self.image_model.parameters())
        parameters.append(self.log_psfs)
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):
        convolved_true, coords = batch

        convolved_pred = self.get_convolved_images(coords)
        image_pred = self.image_model(coords)

        image_loss = self._compute_weighted_image_loss(convolved_pred, convolved_true)

        return {'loss': image_loss,
                'convolved_pred': convolved_pred, 'image_pred': image_pred,
                'convolved_true': convolved_true,
                }

    def validation_epoch_end(self, outputs):
        convolved_pred = torch.cat([o['convolved_pred'] for o in outputs]).reshape(self.images_shape)
        convolved_true = torch.cat([o['convolved_true'] for o in outputs]).reshape(self.images_shape)
        image_pred = torch.cat([o['image_pred'] for o in outputs]).reshape(self.images_shape[0], self.images_shape[1],
                                                                           self.images_shape[-1])
        psfs_pred = self.get_psf()

        convolved_pred = convolved_pred.cpu().detach().numpy()
        convolved_true = convolved_true.cpu().detach().numpy()
        image_pred = image_pred.cpu().detach().numpy()
        psfs_pred = psfs_pred.cpu().detach().numpy()

        self._plot_deconvolution(convolved_true, image_pred)
        self._plot_convolved(convolved_true, convolved_pred)

        if self.sampling == 'grid':
            self._plot_psfs(psfs_pred)
        elif self.sampling == 'spherical':
            self._plot_spherical_psfs(psfs_pred)
        else:
            raise ValueError(f'Unknown sampling method for plotting: {self.sampling}')

        if self.speckle is not None:
            self._plot_deconvolution_speckle(image_pred, self.speckle)

    def _plot_deconvolution(self, convolved_true, image_pred):
        n_channels = convolved_true.shape[-1]
        fig, axs = plt.subplots(2, n_channels, figsize=(3 * n_channels, 4), dpi=300)
        for i in range(n_channels):
            ax = axs[0, i]
            ax.imshow(convolved_true[:, :, 0, i], cmap='gray', origin='lower', vmin=0, vmax=1)
            # divider1 = make_axes_locatable(axs[0, 1])
            # cax1 = divider1.append_axes("right", size="5%", pad="2%")
            # fig.colorbar(im1, ax=cax1)

            ax = axs[1, i]
            ax.imshow(image_pred[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)
            # divider2 = make_axes_locatable(axs[1, 1])
            # cax2 = divider2.append_axes("right", size="5%", pad="2%")
            # fig.colorbar(im2, ax=cax2)

        axs[0, 0].set_ylabel('Reference')
        axs[1, 0].set_ylabel('Deconvolved')
        [axs[0, i].set_title(f'Channel {i:01d}') for i in range(n_channels)]

        fig.tight_layout()
        wandb.log({'Deconvolution': fig})
        plt.close()

    def _plot_deconvolution_speckle(self, image_pred, speckle):
        n_channels = speckle.shape[-1]
        vmin_speckle, vmax_speckle = 0, np.percentile(speckle, 99)
        speckle = (speckle - vmin_speckle) / (vmax_speckle - vmin_speckle)
        fig, axs = plt.subplots(2, n_channels, figsize=(3 * n_channels, 4), dpi=300)
        for i in range(n_channels):
            ax = axs[0, i]
            ax.imshow(speckle[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)

            ax = axs[1, i]
            ax.imshow(image_pred[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)

        axs[0, 0].set_ylabel('Speckle')
        axs[1, 0].set_ylabel('Deconvolved')
        [axs[0, i].set_title(f'Channel {i:01d}') for i in range(n_channels)]

        fig.tight_layout()
        wandb.log({'Deconvolution - Speckle': fig})
        plt.close()

    def _plot_convolved(self, convolved_true, convolved_pred):
        n_channels = convolved_true.shape[-1]
        n_images = convolved_true.shape[-2]
        n_samples = min(5, n_images)
        fig, axs = plt.subplots(2, n_samples, figsize=(2 * n_samples, 5), dpi=300)
        for c in range(n_channels):
            for i in range(n_samples):
                ax = axs[0, i]
                ax.imshow(convolved_true[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=1)
                # divider1 = make_axes_locatable(axs[0, n_samples-1])
                # cax1 = divider1.append_axes("right", size="5%", pad="2%")
                # fig.colorbar(im1, ax=cax1)

                ax = axs[1, i]
                ax.imshow(convolved_pred[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=1)
                # divider2 = make_axes_locatable(axs[1, n_samples-1])
                # cax2 = divider2.append_axes("right", size="5%", pad="2%")
                # fig.colorbar(im2, ax=cax2)

                axs[0, 0].set_ylabel('True')
                axs[1, 0].set_ylabel('Prediction')
                [axs[0, i].set_title(f'Frame {i:01d}') for i in range(n_samples)]

        fig.tight_layout()
        wandb.log({'Convolution': fig})
        plt.close()

    def _plot_psfs(self, psfs):
        n_images = psfs.shape[-1]
        n_samples = min(5, n_images)
        fig, axs = plt.subplots(1, n_samples, figsize=(2 * n_samples, 4), dpi=300)
        for i in range(n_samples):
            ax = axs[i]
            im = ax.imshow(np.sqrt(psfs[:, :, i]), origin='lower', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im, cax=cax)
            ax.set_title(f'PSF {i:02d}')

        fig.tight_layout()
        wandb.log({'PSFs': fig})
        plt.close()

    def _plot_spherical_psfs(self, psfs):
        n_images = psfs.shape[-1]
        n_samples = min(5, n_images)
        psf_shift = self.shift

        fig, axs = plt.subplots(1, n_samples, figsize=(2 * n_samples, 4), dpi=300)
        for i in range(n_samples):
            ax = axs[i]
            im = ax.pcolormesh(self.x, self.y, np.sqrt(psfs[:, :, i]), origin='lower', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im, cax=cax)
            ax.set_title(f'PSF {i:02d}')

        fig.tight_layout()
        wandb.log({'PSFs': fig})
        plt.close()
