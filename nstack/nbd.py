import collections.abc

import torch.nn
from tqdm import tqdm

#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import numpy as np
from astropy.io import fits
from lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
from astropy.nddata import block_reduce
from scipy.ndimage import shift
from skimage import data, filters

from nstack.data.editor import get_KL_basis, get_PSFs, get_convolution, get_KL_wavefront

from nstack.data.editor import ReadSimulationEditor
from nstack.data.psfs import *

from nstack.model import PSFStackModel


class NEURALBDModule(LightningModule):

        def __init__(self, n_images, dim=512, learning_rate=1e-4, n_modes=44, psf_size=128, muram=True, **kwargs):
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
                muram = ReadSimulationEditor().call('/cl_tmp/schirnin/data/I_out_med.468000')
                sim_array = muram.data
                vmin, vmax = 0, np.percentile(sim_array, 99)
                fits_array2_norm = (sim_array - vmin) / (vmax - vmin)
                fits_array2_stack = np.stack([fits_array2_norm, fits_array2_norm], -1)
                #self.high_quality = fits_array2_stack[200:756, 200:756, :]
                self.high_quality = fits_array2_stack[300:428, 300:428, :]
                #self.muram = fits_array2_stack
                self.images = get_convolution(self.high_quality, self.psfs, self.n_images, noise=False)
                #vmin_imgs, vmax_imgs = self.images.min(), self.images.max()
                #self.images = (self.images - vmin_imgs) / (vmax_imgs - vmin_imgs)
                self.im_scaling = (sim_array.shape[0] - 1) / 2
                #self.im_scaling = (2048 - 1) / 2
            else:
                fits_array = []
                #data_path = '/gpfs/data/fs71254/schirni/Level1_Files/hifi_20220602_095015_sd.fts'
                #data_path2 = '/gpfs/data/fs71254/schirni/Level2_Files/hifi_20220602_095015_sd_speckle.fts'
                data_path = '/cl_tmp/schirnin/data/hifi_20170618_082414_sd.fts'
                data_path2 = '/cl_tmp/schirnin/data/hifi_20170618_082414_sd_speckle.fts'
                #
                for i in range(0, 200): # <-- 2022: 0, 200 ; 2022-->: 1,201
                    fits_array.append(fits.getdata(data_path, i))
                #
                h, w = fits_array[0].shape
                fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))
                #fits_array = fits_array[:, :, :, 1]
                fits_array = fits_array[:, :, :, 0]
                #cutoffs = [0.1]
                #lowpass_img = [get_filtered(fits_array[:, :, i], cutoffs) for i in tqdm(range(fits_array.shape[-1]))]
                #fits_array = np.stack(lowpass_img, -1)
                #fits_array = fits_array[0]
                cont = [contrast(fits_array[:, :, i]) for i in range(fits_array.shape[-1])]
                highest_indices = [index for index, value in sorted(enumerate(cont), key=lambda x: x[1], reverse=True)[:self.n_images]]
                fits_array = np.stack([fits_array[:, :, i] for i in highest_indices], -1)
                #shifting = [optimize_shift(fits_array[:, :, 0], fits_array[:, :, i]) for i in tqdm(range(self.n_images))]
                #fits_array = np.stack([shift(fits_array[:, :, i], shift=shifting[i][0], mode='nearest') for i in range(self.n_images)], -1)
                fits_array = fits_array[700:1212, 700:1212, :]
                fits_array = np.stack([fits_array, fits_array], -1)
                #
                fits_array2 = []
                for i in range(0, 2):
                   fits_array2.append(fits.getdata(data_path2, i))
                fits_array2 = np.stack(fits_array2, -1)
                fits_array2 = fits_array2[:, :, 0]
                fits_array2 = np.stack([fits_array2, fits_array2], -1)
                #fits_array2 = block_reduce(fits_array2, block_size=(4, 4, 1), func=np.mean)
                #
                fits_array = fits_array[:, :, :self.n_images]
                #fits_array = fits_array[900:1156, 900:1156, :, :]
                #fits_array2 = fits_array2[900:1156, 900:1156, :]
                fits_array2 = fits_array2[700:1212, 700:1212, :]
                #vmin, vmax = fits_array.min(), np.percentile(fits_array, 99)
                vmin, vmax = fits_array.min(), fits_array.max()
                #vmin = [np.min(fits_array[:, :, i]) for i in range(self.n_images)]
                #vmax = [np.max(fits_array[:, :, i]) for i in range(self.n_images)]
                vmin2, vmax2 = fits_array2.min(), fits_array2.max()

                #self.images = [(fits_array[:, :, i] - vmin[i]) / (vmax[i] - vmin[i]) for i in range(self.n_images)]
                #self.images = np.stack(self.images, -2)
                self.images = (fits_array - vmin) / (vmax - vmin)
                self.high_quality = (fits_array2 - vmin2) / (vmax2 - vmin2)

                self.im_scaling = (1024 - 1) / 2
            self.valid_loss = []
            self.train_loss = []

            self.model = PSFStackModel(self.n_images, self.dim, self.n_modes, self.psf_size, self.kl_basis, self.im_scaling,
                                       self.images, self.high_quality, self.psfs)

            #self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            batch_img, coord_batch = batch
            pred_img, convolved_imgs, psfs, target_imgs, ref_img, ref_psfs, int_scale = self.model(coord_batch)

            #loss_fn = torch.nn.HuberLoss()
            # loss_fn = torch.nn.L1Loss()
            loss_fn = torch.nn.MSELoss()
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
            pred_img, convolved_imgs, psfs, target_imgs, ref_img, ref_psfs, int_scale = self.model(coord_batch)

            #loss_fn = torch.nn.HuberLoss()
            #loss_fn = torch.nn.L1Loss()
            loss_fn = torch.nn.MSELoss()
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

            return DataLoader(data_set, batch_size=1024, num_workers=4)

        def val_dataloader(self):
            images = self.images
            image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
            coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)
            coordinates_tensor = coordinates_tensor / self.im_scaling
            image_tensor = torch.from_numpy(images).float().reshape(-1, self.n_images, 2)
            test_coords = coordinates_tensor
            data_set = TensorDataset(image_tensor, test_coords)

            return DataLoader(data_set, batch_size=1024, num_workers=4)


def hpf(image, n=5):
    """
    High-pass filter an image using a Gaussian kernel.

    Args:
        image: Input image
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        high_pass: High-pass filtered image
    """
    filter = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            filter[i, j] = np.exp(-((i - n // 2) ** 2 + (j - n // 2) ** 2) / 2)
    filter /= filter.sum()
    # Smooth the image
    smoothed = np.zeros_like(image)
    for i in range(image.shape[0] - n + 1):
        for j in range(image.shape[1] - n + 1):
            smoothed[i, j] = np.sum(image[i : i + n, j : j + n] * filter)
    return smoothed


def contrast(image):
    min = np.min(image)
    max = np.max(image)
    return (max-min)/(max+min)

def optimize_shift(img1, img2, max_shift=20):
    """Finds the best shift that maximizes the Pearson correlation coefficient."""
    best_shift = (0, 0)
    best_corr = 0.97
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_img2 = shift(img2, shift=(dx, dy), mode='nearest')
            corr = correlation_coefficient(img1, shifted_img2)
            if corr > best_corr:
                best_corr = corr
                best_shift = (dx, dy)
    return best_shift, best_corr

def correlation_coefficient(patch1, patch2):
    """
    Pearson correlation coefficient between two patches.

    Args:
        patch1: Patch of image 1
        patch2: Patch of image 2
    """
    product = np.nanmean((patch1 - np.nanmean(patch1)) * (patch2 - np.nanmean(patch2)))
    stds = np.nanstd(patch1) * np.nanstd(patch2)
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def get_filtered(image, cutoffs, squared_butterworth=True, order=3.0, npad=0):
    """
    Lowpass and highpass butterworth filtering at all specified cutoffs.
    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    cutoffs : sequence of int
        Both lowpass and highpass filtering will be performed for each cutoff
        frequency in `cutoffs`.
    squared_butterworth : bool, optional
        Whether the traditional Butterworth filter or its square is used.
    order : float, optional
        The order of the Butterworth filter
    Returns
    -------
    lowpass_filtered : list of ndarray
        List of images lowpass filtered at the frequencies in `cutoffs`.
    highpass_filtered : list of ndarray
        List of images highpass filtered at the frequencies in `cutoffs`.
    """
    lowpass_filtered = []
    for cutoff in cutoffs:
        lowpass_filtered.append(
            filters.butterworth(
                image,
                cutoff_frequency_ratio=cutoff,
                order=order,
                high_pass=False,
                squared_butterworth=squared_butterworth,
                npad=npad,
            )
        )
    return lowpass_filtered