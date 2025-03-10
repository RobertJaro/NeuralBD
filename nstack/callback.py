import os

import numpy as np
import lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from nstack.nbd import NEURALBDModule


class PlotNeuralBDCallback(pl.Callback):
    def __init__(self, n_images, test_coords):
        self.n_images = n_images
        self.test_coords = test_coords

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: NEURALBDModule):
        with torch.no_grad():
            batch_size = 4096
            pl_module.model.eval()
            output_image = []
            output_convolved_image = []
            psfs = []
            int_scaling = []
            for i in range(np.ceil(len(self.test_coords) / batch_size).astype(int)):
                batch_coordinates = self.test_coords[i * batch_size:(i + 1) * batch_size]

                output_img, convolved_imgs, psf, target_imgs, ref_img, ref_psfs, int_scale = pl_module.model(batch_coordinates.to(pl_module.device))

                output_image += [output_img.detach().cpu().numpy()]
                output_convolved_image += [convolved_imgs.detach().cpu().numpy()]
                psfs += [psf.detach().cpu().numpy()]
                int_scaling += [int_scale.detach().cpu().numpy()]

            output_image = np.concatenate(output_image, 0).reshape((512, 512, 2))
            output_convolved_image = np.concatenate(output_convolved_image, 0).reshape((512, 512, self.n_images, 2))

            #self.plot_psf(psfs[0], plot_id="Predicted PSF")
            #self.plot_psf(ref_psfs, plot_id="Reference PSF")
            self.subplot_psf(ref_psfs, psfs[0], plot_id="PSF")
            self.plot_nbd(output_convolved_image[..., 0, 0], target_imgs[..., 0, 0], output_image[:, :, 0], ref_img[:, :, 0])
            #self.plot_scaling_factor(int_scaling[0])


    def plot_psf(self, psf, plot_id: str):
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        for i, ax in enumerate(np.ravel(axs)):
            im = ax.imshow(np.sqrt(psf[:, :, i]), origin='lower', vmin=0, vmax=1)
            ax.set_axis_off()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout()
        wandb.log({plot_id: fig})
        plt.close(fig)

    def subplot_psf(self, ref_psf, pred_psf, plot_id: str):
        fig, axs = plt.subplots(2, 10, figsize=(20, 20))
        axs[0, 0].imshow(np.sqrt(ref_psf[:, :, 0]), origin='lower', vmin=0, vmax=1)
        axs[1, 0].imshow(np.sqrt(pred_psf[:, :, 0]), origin='lower', vmin=0, vmax=1)
        axs[0, 1].imshow(np.sqrt(ref_psf[:, :, 1]), origin='lower', vmin=0, vmax=1)
        axs[1, 1].imshow(np.sqrt(pred_psf[:, :, 1]), origin='lower', vmin=0, vmax=1)
        axs[0, 2].imshow(np.sqrt(ref_psf[:, :, 2]), origin='lower', vmin=0, vmax=1)
        axs[1, 2].imshow(np.sqrt(pred_psf[:, :, 2]), origin='lower', vmin=0, vmax=1)
        axs[0, 3].imshow(np.sqrt(ref_psf[:, :, 3]), origin='lower', vmin=0, vmax=1)
        axs[1, 3].imshow(np.sqrt(pred_psf[:, :, 3]), origin='lower', vmin=0, vmax=1)
        axs[0, 4].imshow(np.sqrt(ref_psf[:, :, 4]), origin='lower', vmin=0, vmax=1)
        axs[1, 4].imshow(np.sqrt(pred_psf[:, :, 4]), origin='lower', vmin=0, vmax=1)
        axs[0, 5].imshow(np.sqrt(ref_psf[:, :, 5]), origin='lower', vmin=0, vmax=1)
        axs[1, 5].imshow(np.sqrt(pred_psf[:, :, 5]), origin='lower', vmin=0, vmax=1)
        axs[0, 6].imshow(np.sqrt(ref_psf[:, :, 6]), origin='lower', vmin=0, vmax=1)
        axs[1, 6].imshow(np.sqrt(pred_psf[:, :, 6]), origin='lower', vmin=0, vmax=1)
        axs[0, 7].imshow(np.sqrt(ref_psf[:, :, 7]), origin='lower', vmin=0, vmax=1)
        axs[1, 7].imshow(np.sqrt(pred_psf[:, :, 7]), origin='lower', vmin=0, vmax=1)
        axs[0, 8].imshow(np.sqrt(ref_psf[:, :, 8]), origin='lower', vmin=0, vmax=1)
        axs[1, 8].imshow(np.sqrt(pred_psf[:, :, 8]), origin='lower', vmin=0, vmax=1)
        axs[0, 9].imshow(np.sqrt(ref_psf[:, :, 9]), origin='lower', vmin=0, vmax=1)
        axs[1, 9].imshow(np.sqrt(pred_psf[:, :, 9]), origin='lower', vmin=0, vmax=1)
        plt.tight_layout()
        wandb.log({plot_id: fig})
        plt.close(fig)

    def plot_convolved(self, convolved, target_imgs):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].imshow(convolved[..., 0, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
        ax[1].imshow(target_imgs[..., 0, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.tight_layout()
        wandb.log({"convolution": fig})
        plt.close(fig)

    def plot_reconstruction(self, recon, ref_image):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].imshow(recon[:, :, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
        ax[1].imshow(ref_image[:, :, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.tight_layout()
        wandb.log({"reconstruction": fig})
        plt.close(fig)

    def plot_nbd(self, conv, target_conv, recons, ref_recons):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(conv, cmap='gray', origin='lower', vmin=0, vmax=1)
        axs[1, 0].imshow(target_conv, cmap='gray', origin='lower', vmin=0, vmax=1)
        axs[0, 1].imshow(recons, cmap='gray', origin='lower', vmin=0, vmax=1)
        axs[1, 1].imshow(ref_recons, cmap='gray', origin='lower', vmin=0, vmax=1)
        axs[0, 0].set_title("Convolved")
        axs[0, 1].set_title("Reconstruction")
        plt.tight_layout()
        wandb.log({"nbd": fig})
        plt.close(fig)

    def plot_scaling_factor(self, int_scale):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(int_scale[:, 0], label="Intensity Scaling")
        plt.tight_layout()
        wandb.log({"intensity_scaling": fig})
        plt.close(fig)