import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nbd.data.editor import gaussian_psf
from nbd.model import ImageModel, PSFModel


class NEURALBDModule(LightningModule):

    def __init__(self, images_shape, pixel_per_ds, learning_rate=1e-4, psf_size=(29, 29),
                 model_config=None, weights=None, lr_config=None, speckle=None, muram=None,
                 psf=None, raw_frame=None, psf_type='default', **kwargs):
        super().__init__()
        self.images_shape = images_shape
        self.n_images = self.images_shape[2]
        self.speckle = speckle
        self.psf_type = psf_type
        self.muram = muram
        self.kl_psfs = psf
        self.psf_size = psf_size
        self.raw_frame = raw_frame

        self.learning_rate = learning_rate

        # Create PSF coordinates for sampling (Grid sampling)
        x_values = torch.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_size[0], dtype=torch.float32)
        y_values = torch.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_size[1], dtype=torch.float32)
        x, y = torch.meshgrid(x_values, y_values, indexing='ij')
        psf_coords = torch.stack([x, y], -1)  # psf_coords, xy
        psf_coords = psf_coords / pixel_per_ds

        self.psf_coords = nn.Parameter(psf_coords, requires_grad=False)

        if self.psf_type == 'default':
            # Create Gaussian PSFs
            log_psfs = gaussian_psf(psf_size, sigma=5, n_images=self.n_images)
            self.log_psfs = nn.Parameter(torch.tensor(log_psfs, dtype=torch.float32), requires_grad=True)


        elif self.psf_type == 'varying':
            self.psf_model = PSFModel((*psf_size, self.n_images))

        else:
            raise ValueError(f'Unknown psf method: {self.psf_type}')

        # Create and normalize weights
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * 0.1 + 0.9
        weights = torch.ones(self.n_images, dtype=torch.float32)
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)

        # Create image model
        model_config = model_config if model_config is not None else {}
        self.image_model = ImageModel(**model_config)
        # self.image_model = ImageSirenModel(**model_config)

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
        if self.psf_type == 'default':
            psf = self.get_psf(area_elements)
        elif self.psf_type == 'varying':
            psf = self.get_varying_psf(coords, area_elements)
        else:
            raise ValueError(f'Unknown psf method: {self.psf_type}')

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

    def get_psf(self, area_elements=None):
        # area_elements: batch, x, y
        # self.log_psfs: x, y, n_images
        psfs = torch.exp(self.log_psfs)  # --> x, y, n_images
        # kl_psfs = self.kl_psfs.to(self.log_psfs.device)
        # Normalize PSFs
        if area_elements is None:
            norm = psfs.sum(dim=(0, 1), keepdim=True)
            # kl_psfs = kl_psfs
        else:
            norm = (psfs[None, :, :, :] * area_elements[:, :, :, None]).sum(dim=(1, 2),
                                                                            keepdim=True)  # --> batch, 1, 1, n_images
            # kl_psfs = kl_psfs[None, :, :, :] * area_elements[:, :, :, None]
        return psfs / (norm + 1e-8)  # --> batch, x, y, n_images
        # return kl_psfs

    def get_varying_psf(self, coords, area_elements):
        # area_elements: batch, x, y
        # self.log_psfs: batch, x, y, n_images
        log_psfs = self.psf_model(coords)  # --> batch, x, y, n_images
        psfs = torch.exp(log_psfs)  # --> batch, x, y, n_images

        # Normalize PSFs
        norm = (psfs * area_elements[:, :, :, None]).sum(dim=(1, 2), keepdim=True)  # --> batch, 1, 1, n_images

        return psfs / (norm + 1e-8)  # --> batch, x, y, n_images

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
        # image_loss = (convolved_diff * self.weights[None, :, None]).sum(1) / self.weights.sum()
        # image_loss = image_loss.mean()
        image_loss = torch.mean(convolved_diff)
        return image_loss

    def configure_optimizers(self):
        if self.psf_type == 'default':
            parameters = list(self.image_model.parameters())
            parameters.append(self.log_psfs)
        elif self.psf_type == 'varying':
            parameters = list(self.image_model.parameters()) + list(self.psf_model.parameters())
        else:
            raise ValueError(f'Unknown psf method: {self.psf_type}')

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

        if self.psf_type == 'default':
            psfs_pred = self.get_psf()
        elif self.psf_type == 'varying':
            # Dummy coordinates and area elements for PSF prediction
            dummy_coords = torch.ones(1, 2, dtype=torch.float32, device=convolved_pred.device) * 0.5
            dummy_area_elements = torch.ones(1, *self.psf_size, dtype=torch.float32, device=convolved_pred.device) * 0.5
            psfs_pred = self.get_varying_psf(dummy_coords, dummy_area_elements)[0]
        else:
            raise ValueError(f'Unknown psf method: {self.psf_type}')

        convolved_pred = convolved_pred.cpu().detach().numpy()
        convolved_true = convolved_true.cpu().detach().numpy()
        image_pred = image_pred.cpu().detach().numpy()
        psfs_pred = psfs_pred.cpu().detach().numpy()

        vmin_pred, vmax_pred = np.min(image_pred), np.max(image_pred)
        image_pred = (image_pred - vmin_pred) / (vmax_pred - vmin_pred)

        self._plot_deconvolution(convolved_true, image_pred)
        self._plot_convolved(convolved_true, convolved_pred)
        self._plot_psfs(psfs_pred)

        # save PSFs and images
        if self.speckle is not None:
            self._plot_deconvolution_speckle(image_pred, self.speckle)
            gregor_save_path = '/gpfs/data/fs71254/schirni/nstack/training/GREGOR_v2'
            np.save(gregor_save_path + '/psfs_pred.npy', psfs_pred)
            np.save(gregor_save_path + '/conv_true.npy', convolved_true)
            np.save(gregor_save_path + '/conv_pred.npy', convolved_pred)

        elif self.muram is not None:
            self._plot_deconvolution_muram(image_pred, self.muram)
            save_path = '/gpfs/data/fs71254/schirni/nstack/training/NeuralBD_muram_varying_pretrain_block'
            np.save(save_path + '/psfs_pred.npy', psfs_pred)
            np.save(save_path + '/psfs_true.npy', self.kl_psfs)
            np.save(save_path + '/conv_true.npy', convolved_true)
            np.save(save_path + '/conv_pred.npy', convolved_pred)

            self._plot_kl_psfs(self.kl_psfs, psfs_pred)
        else:
            #dkist_save_path = '/gpfs/data/fs71254/schirni/nstack/training/DKIST_penumbra_v3'
            #np.save(dkist_save_path + '/psfs_pred.npy', psfs_pred)
            #np.save(dkist_save_path + '/conv_true.npy', convolved_true)
            #np.save(dkist_save_path + '/conv_pred.npy', convolved_pred)
            kso_save_path = '/gpfs/data/fs71254/schirni/nstack/training/KSO_2023_final'
            np.save(kso_save_path + '/psfs_pred.npy', psfs_pred)
            np.save(kso_save_path + '/conv_true.npy', convolved_true)
            np.save(kso_save_path + '/conv_pred.npy', convolved_pred)

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
        vmin_speckle, vmax_speckle = np.min(speckle), np.max(speckle)
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

    def _plot_deconvolution_muram(self, image_pred, muram):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
        axs[0].imshow(muram[:, :, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
        axs[1].imshow(image_pred[:, :, 0], cmap='gray', origin='lower', vmin=0, vmax=1)

        axs[0].set_title('Muram')
        axs[1].set_title('Deconvolved')

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
