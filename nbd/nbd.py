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

    def __init__(self, images_shape, pixel_per_ds, learning_rate=1e-4, psf_size=(29, 29),
                 model_config=None, weights=None, lr_config=None, speckle=None, muram=None,
                 psf=None, sampling='grid', **kwargs):
        super().__init__()
        self.images_shape = images_shape
        self.n_images = self.images_shape[2]
        self.speckle = speckle
        self.sampling = sampling
        self.muram = muram
        self.kl_psfs = psf

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


        elif self.sampling == 'spherical':
            # Create PSF coordinates for sampling (Spherical sampling)
            max_radius = 10
            r_values = np.linspace(0, 1, 10, dtype=np.float32) * max_radius
            phi_values = np.linspace(0, 2 * torch.pi, 18, endpoint=False, dtype=np.float32)
            # remove r = 0
            self.phi, self.r = np.meshgrid(phi_values, r_values[1:], indexing='ij')

            # convert to cartesian
            x, y = self.r * np.sin(self.phi), self.r * np.cos(self.phi)
            psf_coords = np.stack([x, y], -1)  # shape: phi, r, 2

            # add central point
            psf_coords = psf_coords / pixel_per_ds
            self.psf_coords = nn.Parameter(torch.tensor(psf_coords, dtype=torch.float32), requires_grad=False)

            # calculate area element
            dr = np.gradient(self.r, axis=1)
            dphi = np.gradient(self.phi, axis=0)
            area_elements = self.r * dphi * dr  # shape: phi, r

            area_elements = area_elements / (pixel_per_ds)
            # area_elements = np.repeat(area_elements[..., None], self.n_images, axis=-1) * 6.5e1 # rmax=20
            area_elements = np.repeat(area_elements[..., None], self.n_images, axis=-1) * 2.5e2  # r_max=10
            self.area_elements = nn.Parameter(torch.tensor(area_elements, dtype=torch.float32), requires_grad=False)

            # Create learnable PSFs
            log_psfs = np.random.randn(x.shape[0], x.shape[1], self.n_images).astype(np.float32)
            self.log_psfs = nn.Parameter(torch.tensor(log_psfs, dtype=torch.float32), requires_grad=True)

        else:
            raise ValueError(f'Unknown sampling method: {sampling}')

        # Create and normalize weights
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * 0.1 + 0.9
        weights = torch.ones(self.n_images, dtype=torch.float32)
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

        # Create image model
        model_config = model_config if model_config is not None else {}
        self.image_model = ImageModel(**model_config)

        # Learning rate scheduler
        self.lr_config = {"start": 1e-4, "end": 1e-4, "iterations": 1e5} if lr_config is None else lr_config

    def get_convolved_images(self, coords):
        # create grid of sampling coordinates for PSF
        # coords: batch, 2
        # psf_coords: x, y, 2

        # add random shift to PSF coordinates
        psf_coords = self.psf_coords[None, :, :, :]  # --> 1, x, y, 2
        psf_coords = psf_coords.repeat(coords.shape[0], 1, 1, 1)  # --> batch, x, y, 2

        # calculate shift max between coordinate points
        dx_max = (psf_coords[0, 0, 0, 0] - psf_coords[0, 1, 0, 0]) / 2
        dy_max = (psf_coords[0, 0, 0, 1] - psf_coords[0, 0, 1, 1]) / 2

        # initialize random shifts per point
        d_rand = torch.rand_like(psf_coords)

        # stretch random shifts from [-dx, dx] and [-dy, dy]
        d_rand = d_rand * 2 - 1
        d_rand[..., 0] = d_rand[..., 0] * dx_max
        d_rand[..., 1] = d_rand[..., 1] * dy_max

        # apply random shifts
        psf_coords = psf_coords + d_rand  # --> batch, x, y, 2

        # compute area elements
        dx = torch.gradient(psf_coords[..., 0], dim=1)[0]
        dy = torch.gradient(psf_coords[..., 1], dim=2)[0]
        area_elements = dx * dy  # --> batch, x, y

        sampling_coords = coords[:, None, :] + psf_coords.reshape(coords.shape[0], -1, 2)  # --> batch, psf_coords, 2

        # load the PSF
        # psf: batch, x, y, n_images
        psf = self.get_psf(area_elements)
        flat_psf = psf.reshape(coords.shape[0], -1, self.n_images)
        flat_area_elements = area_elements.reshape(coords.shape[0], -1, 1)

        image = self.image_model(sampling_coords.reshape(-1, 2))
        image = image.reshape(coords.shape[0], -1, image.shape[-1])
        # image:  batch, xy(PSF), channels
        # flat_psf: batch, xy(PSF) , n_images
        # area_elements: batch, xy(PSF), 1
        convolved_images = torch.einsum('bsc,bsn->bnc', image, flat_psf * flat_area_elements)
        # convolved_images: batch, n_images, channels

        return convolved_images

    def get_shift(self):
        shift = torch.tanh(self.shift) * self.shift_scaling
        return shift

    def get_psf(self, area_elements=None):
        # area_elements: batch, x, y
        # self.log_psfs: x, y, n_images
        psfs = torch.exp(self.log_psfs) # --> x, y, n_images

        # normalize PSFs
        if area_elements is None:
            norm = psfs.sum(dim=(0, 1), keepdim=True)
        else:
            norm = (psfs[None, :, :, :] * area_elements[:, :, :, None]).sum(dim=(1, 2),
                                                                            keepdim=True)  # --> batch, 1, 1, n_images
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
        # image_loss = torch.mean(convolved_diff)
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

        vmin_pred, vmax_pred = np.min(image_pred), np.max(image_pred)
        image_pred = (image_pred - vmin_pred) / (vmax_pred - vmin_pred)

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

        if self.muram is not None:
            self._plot_deconvolution_muram(image_pred, self.muram)
            np.save('/gpfs/data/fs71254/schirni/NeuralBD_muram/image_pred.npy', image_pred)
            np.save('/gpfs/data/fs71254/schirni/NeuralBD_muram/psfs_pred.npy', psfs_pred)
            np.save('/gpfs/data/fs71254/schirni/NeuralBD_muram/convolved_true.npy', convolved_true)
            np.save('/gpfs/data/fs71254/schirni/NeuralBD_muram/convolved_pred.npy', convolved_pred)
            np.save('/gpfs/data/fs71254/schirni/NeuralBD_muram/muram.npy', self.muram)

        if self.kl_psfs is not None:
            self._plot_kl_psfs(self.kl_psfs, psfs_pred)

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
            ax.imshow(image_pred[:, :, i], cmap='gray', origin='lower')
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
        vmin_speckle, vmax_speckle = np.min(speckle), np.max(speckle)
        speckle = (speckle - vmin_speckle) / (vmax_speckle - vmin_speckle)
        fig, axs = plt.subplots(2, n_channels, figsize=(3 * n_channels, 4), dpi=300)
        for i in range(n_channels):
            ax = axs[0, i]
            ax.imshow(speckle[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)

            ax = axs[1, i]
            ax.imshow(image_pred[:, :, i], cmap='gray', origin='lower')
        axs[0, 0].set_ylabel('Speckle')
        axs[1, 0].set_ylabel('Deconvolved')
        [axs[0, i].set_title(f'Channel {i:01d}') for i in range(n_channels)]

        fig.tight_layout()
        wandb.log({'Deconvolution - Speckle': fig})
        plt.close()

    def _plot_deconvolution_muram(self, image_pred, muram):
        n_channels = image_pred.shape[-1]
        fig, axs = plt.subplots(2, n_channels, figsize=(3 * n_channels, 4), dpi=300)
        for i in range(n_channels):
            ax = axs[0, i]
            ax.imshow(muram[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)

            ax = axs[1, i]
            ax.imshow(image_pred[:, :, i], cmap='gray', origin='lower', vmin=0, vmax=1)

        axs[0, 0].set_ylabel('Muram')
        axs[1, 0].set_ylabel('Deconvolved')
        [axs[0, i].set_title(f'Channel {i:01d}') for i in range(n_channels)]

        fig.tight_layout()
        wandb.log({'Deconvolution - Muram': fig})
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
                ax.imshow(convolved_pred[:, :, i, c], cmap='gray', origin='lower')
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

        fig, axs = plt.subplots(1, n_samples, figsize=(2 * n_samples, 4), subplot_kw={'projection': 'polar'}, dpi=300)
        # Collect images for a shared colorbar
        ims = []
        for i, ax in enumerate(axs):
            im = ax.pcolormesh(self.phi, self.r, np.sqrt(psfs[:, :, i]), vmin=0, vmax=1, edgecolors='face')
            ims.append(im)
            ax.axis('off')
            ax.set_title(f'PSF {i:02d}')
        # Add a colorbar at the bottom, aligned with figure width
        cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.03])  # [left, bottom, width, height]
        fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal')
        wandb.log({'PSFs': wandb.Image(fig)})
        plt.close()

    def _plot_kl_psfs(self, kl_psfs, psfs_pred):
        n_images = psfs_pred.shape[-1]
        n_samples = min(5, n_images)
        fig, axs = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4), dpi=300)
        for i in range(n_samples):
            ax = axs[0, i]
            im = ax.imshow(np.sqrt(kl_psfs[:, :, i]), origin='lower', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im, cax=cax)
            ax.set_title(f'PSF {i:02d}')

            ax = axs[1, i]
            im = ax.imshow(np.sqrt(psfs_pred[:, :, i]), origin='lower', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im, cax=cax)
            ax.set_title(f'PSF {i:02d}')

            axs[0, 0].set_ylabel('True')
            axs[1, 0].set_ylabel('Prediction')

        fig.tight_layout()
        wandb.log({'KL_PSFs': fig})
        plt.close()
