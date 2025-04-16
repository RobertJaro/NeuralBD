from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import shift
from skimage import filters
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header

from nbd.data.KL_modes import KL


class Editor(ABC):

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


class ReadSimulationEditor(Editor):

    def call(self, filename, **kwargs):
        f = np.fromfile(filename, dtype='float32')
        nvars = f[0].astype('int')
        ny = f[1].astype('int')
        nx = f[2].astype('int')
        t_iteration = f[3].astype('int')
        arr = f[4:]
        arr = arr.reshape((nvars, nx, ny))
        index_ic = 0
        data = arr[index_ic, :, :]
        scale = (0.12144, 0.12144)
        my_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2012-01-01",
                            observer='earth', frame=frames.Helioprojective)
        header = make_fitswcs_header(data, my_coord, scale=scale * u.arcsec / u.pix)
        sim_map = Map(data, header)

        return sim_map


def PSF(complx_pupil):
    PSF = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(complx_pupil)))
    PSF = (torch.abs(PSF)) ** 2  # or PSF*PSF.conjugate()
    # PSF = PSF/ torch.sum(PSF, dim=(0, 1))  #normalizing the PSF
    return PSF


def get_KL_basis(n_modes_max, size):
    kl = KL()
    KL_modes = kl.precalculate_covariance(npix_image=size, n_modes_max=n_modes_max, first_noll=2)
    KL_modes = torch.tensor(KL_modes, dtype=torch.float32)
    return KL_modes


def get_KL_wavefront(KL_modes, n_modes_max, n_images, coef_range=2):
    coef = torch.FloatTensor(n_images, n_modes_max).uniform_(-coef_range, coef_range)
    # coef = torch.FloatTensor(n_images, n_modes_max).uniform_(0, 1)
    KL_wavefront = torch.einsum('kij,lk->lij', KL_modes, coef)
    return KL_wavefront


def generate_PSFs(wavefront, n_images):
    PSFS = torch.stack([PSF(torch.exp(1j * wavefront[i, :, :])) for i in range(n_images)], -1)
    PSFS = PSFS / (torch.sum(PSFS, dim=(0, 1), keepdim=True))
    return PSFS


def get_convolution(simulation, psfs, n_images, noise=False):
    convolved_images = np.stack([cv2.filter2D(simulation[..., 0], -1, psfs[:, :, i].numpy()) for i in range(n_images)],
                                -1)
    if noise:
        noise = np.random.normal(1, 0.005, size=convolved_images.shape)
        convolved_images += noise
    convolved_images = np.stack([convolved_images, convolved_images], -1)
    return convolved_images


def compute_rms_contrast(image):
    mean = np.mean(image)
    rms = np.sqrt(np.mean((image - mean) ** 2))
    return rms


def cutout(image, x, y, size):
    return image[x - size // 2:x + size // 2, y - size // 2:y + size // 2, :]


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


def optimize_shift(img1, img2, max_shift=20):
    """Finds the best shift that maximizes the Pearson correlation coefficient."""
    best_shift = (0, 0)
    best_corr = 0.97
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_img2 = shift(img2, shift=(dx, dy), mode='nearest')
            corr = correlation_coefficient(img1, shifted_img2)
            if corr > best_corr or best_corr is None:
                best_corr = corr
                best_shift = (dx, dy)
    return best_shift, best_corr


def shift_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """
    Shifts an image by shift_x pixels in the x-direction and shift_y pixels in the y-direction
    using NumPy only, with zero-padding instead of wrapping.

    Parameters:
    - image (np.ndarray): Input grayscale image as a 2D NumPy array.
    - shift_x (int): Number of pixels to shift along the x-axis (right if positive, left if negative).
    - shift_y (int): Number of pixels to shift along the y-axis (down if positive, up if negative).

    Returns:
    - np.ndarray: Shifted image with zero-padding.
    """
    # Create an empty array with the same shape, filled with zeros
    shifted_image = np.zeros_like(image)

    # Image dimensions
    h, w = image.shape

    # Determine valid regions to copy from and paste into
    y_start, y_end = max(0, shift_y), min(h, h + shift_y)
    x_start, x_end = max(0, shift_x), min(w, w + shift_x)

    src_y_start, src_y_end = max(0, -shift_y), min(h, h - shift_y)
    src_x_start, src_x_end = max(0, -shift_x), min(w, w - shift_x)

    # Copy the valid region from the original image to the shifted image
    shifted_image[y_start:y_end, x_start:x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]

    return shifted_image
