import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from nbd.data.editor import gaussian_psf
from nbd.model import ImageModel, PSFModel, ObjectModel, GModel, PModel
from nbd.util import jacobian


class NeuralBDIntModule(LightningModule):

    def __init__(self, images_shape, pixel_per_ds, learning_rate=1e-4, psf_size=(29, 29),
                 model_config=None, lr_config=None, speckle=None, muram=None,
                 psf=None, raw_frame=None, **kwargs):
        super().__init__()
        self.images_shape = images_shape
        self.n_images = self.images_shape[2]
        self.speckle = speckle
        self.muram = muram
        self.kl_psfs = psf
        self.psf_size = psf_size
        self.raw_frame = raw_frame

        self.learning_rate = learning_rate

        # Initialize models
        self.object_model = ObjectModel()
        self.g_model = GModel(n_images=self.n_images)
        self.p_model = PModel(n_images=self.n_images)

        psf_x_min = - (psf_size[0] // 2) / pixel_per_ds
        psf_x_max = (psf_size[0] // 2) / pixel_per_ds
        psf_y_min = - (psf_size[1] // 2) / pixel_per_ds
        psf_y_max = (psf_size[1] // 2) / pixel_per_ds

        self.psf_x_min = nn.Parameter(torch.tensor(psf_x_min, dtype=torch.float32), requires_grad=False)
        self.psf_x_max = nn.Parameter(torch.tensor(psf_x_max, dtype=torch.float32), requires_grad=False)
        self.psf_y_min = nn.Parameter(torch.tensor(psf_y_min, dtype=torch.float32), requires_grad=False)
        self.psf_y_max = nn.Parameter(torch.tensor(psf_y_max, dtype=torch.float32), requires_grad=False)

        self.lambdas = {"image": 1.0, "psf": 1e-1, "convolution": 1e-1, "psf_positive": 1, "intensity_positive": 1}

        # Learning rate scheduler
        self.lr_config = {"start": 1e-4, "end": 1e-4, "iterations": 1e5} if lr_config is None else lr_config

    def get_convolved_images(self, coords):
        # coords: batch, 2 (x, y)
        psf_x_min = torch.ones_like(coords[..., 0:1]) * self.psf_x_min
        psf_x_max = torch.ones_like(coords[..., 0:1]) * self.psf_x_max
        psf_y_min = torch.ones_like(coords[..., 1:2]) * self.psf_y_min
        psf_y_max = torch.ones_like(coords[..., 1:2]) * self.psf_y_max

        min_coords = torch.cat([coords, psf_x_min, psf_y_min], dim=-1)
        max_coords = torch.cat([coords, psf_x_max, psf_y_max], dim=-1)

        convolved_images = self.g_model(max_coords) - self.g_model(min_coords)
        convolved_images = convolved_images[..., None].repeat(1, 1, 2)  # batch, n_images, channels
        int_psf = self.p_model(max_coords) - self.p_model(min_coords)
        return convolved_images, int_psf

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

        convolved_pred, int_psf = self.get_convolved_images(coords)

        # create random pixel coordinates
        psf_x_rand = torch.rand_like(coords[..., 0:1]) * (self.psf_x_max - self.psf_x_min) + self.psf_x_min
        psf_y_rand = torch.rand_like(coords[..., 0:1]) * (self.psf_y_max - self.psf_y_min) + self.psf_y_min

        rand_coords = torch.cat([coords, psf_x_rand, psf_y_rand], dim=-1)
        rand_coords.requires_grad = True
        p_out = self.p_model(rand_coords) # batch, n_images
        g_out = self.g_model(rand_coords) # batch, n_images

        # select random images for each sample in the batch
        batch_size = coords.shape[0]
        n_images = self.n_images
        random_image_indices = torch.randint(0, n_images, (batch_size,), device=coords.device)
        p_out = p_out[torch.arange(batch_size), random_image_indices] # batch
        g_out = g_out[torch.arange(batch_size), random_image_indices] # batch

        p_out = p_out.unsqueeze(-1) # batch, 1
        g_out = g_out.unsqueeze(-1) # batch, 1


        # Compute second derivatives for p
        jac_p = jacobian(p_out, rand_coords) # batch, n_images, 4
        dp_dpx = jac_p[:, :, 2] # batch, n_images
        jac_dp_dpx = jacobian(dp_dpx, rand_coords) # batch, n_images, 4
        d2p_dpx_dpy = jac_dp_dpx[:, :, 3] # batch, n_images
        psf = d2p_dpx_dpy # batch, n_images

        # compute second derivatives for g
        jac_g = jacobian(g_out, rand_coords) # batch, n_images, 4
        dg_dpx = jac_g[:, :, 2] # batch, n_images
        jac_dg_dx = jacobian(dg_dpx, rand_coords) # batch, n_images, 4
        d2g_dpx_dpy = jac_dg_dx[:, :, 3] # batch, n_images
        raw_intensity = d2g_dpx_dpy # batch, n_images

        shift = torch.cat([psf_x_rand, psf_y_rand], dim=-1)
        shifted_coords = coords + shift
        object = self.object_model(shifted_coords) # batch, 1

        # Compute loss
        image_loss = self._compute_weighted_image_loss(convolved_pred, convolved_true)
        psf_loss = (1 - int_psf).pow(2).mean()
        convolution_loss = (object * psf - raw_intensity).pow(2).sum(-1).mean()

        # regularize positive values
        psf_positive_loss = torch.relu(-psf).pow(2).mean()
        intensity_positive_loss = torch.relu(-int_psf).pow(2).mean()

        total_loss = (image_loss * self.lambdas["image"] +
                      psf_loss * self.lambdas["psf"] +
                      convolution_loss * self.lambdas["convolution"] +
                      psf_positive_loss * self.lambdas["psf_positive"] +
                      intensity_positive_loss * self.lambdas["intensity_positive"])

        self.log('image_loss', image_loss)
        self.log('psf_loss', psf_loss)
        self.log('convolution_loss', convolution_loss)
        self.log('psf_positive_loss', psf_positive_loss)
        self.log('intensity_positive_loss', intensity_positive_loss)
        self.log('loss', total_loss)
        return total_loss


    def _compute_weighted_image_loss(self, convolved_pred, convolved_true):
        convolved_diff = (convolved_pred - convolved_true) ** 2
        # convolved_diff: batch, n_images, channels
        # weight: n_images
        # image_loss = (convolved_diff * self.weights[None, :, None]).sum(1) / self.weights.sum()
        # image_loss = image_loss.mean()
        image_loss = torch.mean(convolved_diff)
        return image_loss

    def configure_optimizers(self):
        parameters = list(self.object_model.parameters()) + list(self.g_model.parameters()) + list(self.p_model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):
        convolved_true, coords = batch

        convolved_pred, psf_int = self.get_convolved_images(coords)
        image_pred = self.object_model(coords)
        image_pred = image_pred.repeat(1, 2)

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

        convolved_pred = convolved_pred.cpu().detach().numpy()
        convolved_true = convolved_true.cpu().detach().numpy()
        image_pred = image_pred.cpu().detach().numpy()
        # psfs_pred = psfs_pred.cpu().detach().numpy()

        vmin_pred, vmax_pred = np.min(image_pred), np.max(image_pred)
        image_pred = (image_pred - vmin_pred) / (vmax_pred - vmin_pred)

        self._plot_deconvolution(convolved_true, image_pred)
        self._plot_convolved(convolved_true, convolved_pred)
        # self._plot_psfs(psfs_pred)

        # save PSFs and images
        if self.speckle is not None:
            self._plot_deconvolution_speckle(image_pred, self.speckle)
            gregor_save_path = '/gpfs/data/fs71254/schirni/nstack/training/GREGOR_v2'
            #np.save(gregor_save_path + '/psfs_pred.npy', psfs_pred)
            np.save(gregor_save_path + '/conv_true.npy', convolved_true)
            np.save(gregor_save_path + '/conv_pred.npy', convolved_pred)

        elif self.muram is not None:
            self._plot_deconvolution_muram(image_pred, self.muram)
            save_path = '/gpfs/data/fs71254/schirni/nstack/training/NeuralBD_muram_varying_pretrain_block'
            #np.save(save_path + '/psfs_pred.npy', psfs_pred)
            np.save(save_path + '/psfs_true.npy', self.kl_psfs)
            np.save(save_path + '/conv_true.npy', convolved_true)
            np.save(save_path + '/conv_pred.npy', convolved_pred)

            #self._plot_kl_psfs(self.kl_psfs, psfs_pred)
        else:
            #dkist_save_path = '/gpfs/data/fs71254/schirni/nstack/training/DKIST_penumbra_v2'
            #np.save(dkist_save_path + '/psfs_pred.npy', psfs_pred)
            #np.save(dkist_save_path + '/conv_true.npy', convolved_true)
            #np.save(dkist_save_path + '/conv_pred.npy', convolved_pred)
            kso_save_path = '/gpfs/data/fs71254/schirni/nstack/training/KSO_2023_integration'
            # np.save(kso_save_path + '/psfs_pred.npy', psfs_pred)
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
