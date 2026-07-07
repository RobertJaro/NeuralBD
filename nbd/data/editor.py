from abc import ABC, abstractmethod
import warnings

from scipy.signal import convolve2d
import numpy as np
import torch
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LinearStretch
from scipy.ndimage import shift
from skimage import filters
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from sunpy.map.mapbase import GenericMap
from sunpy.map import all_coordinates_from_map
import torch.nn.functional as F

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


class LoadMapEditor(Editor):
    """
    Load SunPy Map editor.
    """

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            s_map.meta["timesys"] = "tai"
            return s_map, {"path": data}


class KSOPrepEditor(Editor):
    """
    KSO data preparation editor.
    """

    def __init__(self, add_rotation=False):
        self.add_rotation = add_rotation
        super().__init__()

    def call(self, kso_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kso_map.meta["waveunit"] = "AA"
            kso_map.meta["arcs_pp"] = kso_map.scale[0].value
            if "exptime" not in kso_map.meta and "exp_time" in kso_map.meta:
                kso_map.meta["exptime"] = kso_map.meta["exp_time"] / 1000

            if self.add_rotation:
                angle = -kso_map.meta.get("angle", 0)
            else:
                angle = 0
            c = np.cos(np.deg2rad(angle))
            s = np.sin(np.deg2rad(angle))

            kso_map.meta["PC1_1"] = c
            kso_map.meta["PC1_2"] = -s
            kso_map.meta["PC2_1"] = s
            kso_map.meta["PC2_2"] = c
            return kso_map


class LimbDarkeningCorrectionEditor(Editor):
    """
    Limb darkening correction editor.
    """

    def __init__(self, limb_offset=0.99):
        self.limb_offset = limb_offset

    def call(self, s_map, **kwargs):
        coords = all_coordinates_from_map(s_map)
        radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value
        radial_distance[radial_distance >= self.limb_offset] = np.NaN
        ideal_correction = np.cos(radial_distance * np.pi / 2)

        condition = np.logical_not(np.isnan(np.ravel(ideal_correction)))
        map_list = np.ravel(s_map.data)[condition]
        correction_list = np.ravel(ideal_correction)[condition]

        fit = np.polyfit(correction_list, map_list, 4)
        poly_fit = np.poly1d(fit)

        map_correction = poly_fit(ideal_correction)
        corrected_map = s_map.data / map_correction
        return Map(corrected_map, s_map.meta)


class MapToDataEditor(Editor):
    """
    SunPy map to data editor.
    """

    def call(self, s_map, **kwargs):
        return s_map.data, {"header": s_map.meta}


class ImageNormalizeEditor(Editor):
    """
    Image normalization editor.
    """

    def __init__(self, vmin=None, vmax=None, stretch=LinearStretch()):
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

    def call(self, data, **kwargs):
        data = self.norm(data).data * 2 - 1
        return data


class NanEditor(Editor):
    """
    Replace NaN values editor.
    """

    def __init__(self, nan=0):
        self.nan = nan

    def call(self, data, **kwargs):
        data = np.nan_to_num(data, nan=self.nan)
        return data


class LoadKSOMapEditor(Editor):
    """
    Load a KSO SunPy map and normalize metadata for downstream processing.

    Parameters
    ----------
    add_rotation : bool, optional
        If True, apply the negative header angle into the PC rotation matrix.
    """

    def __init__(self, add_rotation=True):
        self.add_rotation = add_rotation

    @staticmethod
    def _load_map(source):
        if isinstance(source, GenericMap):
            return source

        try:
            return Map(source)
        except Exception:
            data, header = fits.getdata(source, header=True)

            if "cunit1" not in header:
                header["cunit1"] = "arcsec"
            if "cunit2" not in header:
                header["cunit2"] = "arcsec"

            if "cdelt1" not in header and "arcs_pp" in header:
                header["cdelt1"] = header["arcs_pp"]
            if "cdelt2" not in header and "arcs_pp" in header:
                header["cdelt2"] = header["arcs_pp"]

            return Map(data, header)

    def call(self, source, **kwargs):
        kso_map = self._load_map(source)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kso_map.meta["waveunit"] = "AA"
            kso_map.meta["arcs_pp"] = kso_map.scale[0].value
            if "exptime" not in kso_map.meta and "exp_time" in kso_map.meta:
                kso_map.meta["exptime"] = kso_map.meta["exp_time"] / 1000

            if self.add_rotation:
                angle = -kso_map.meta.get("angle", 0)
            else:
                angle = 0
            c = np.cos(np.deg2rad(angle))
            s = np.sin(np.deg2rad(angle))

            kso_map.meta["PC1_1"] = c
            kso_map.meta["PC1_2"] = -s
            kso_map.meta["PC2_1"] = s
            kso_map.meta["PC2_2"] = c

            return kso_map

def PSF(complx_pupil):
    PSF = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(complx_pupil)))
    PSF = (torch.abs(PSF)) ** 2  # or PSF*PSF.conjugate()
    PSF = PSF / torch.sum(PSF, dim=(0, 1))  #normalizing the PSF
    return PSF


def get_KL_basis(n_modes_max, size):
    kl = KL()
    KL_modes = kl.precalculate_covariance(npix_image=size, n_modes_max=n_modes_max, first_noll=1)
    KL_modes /= np.max(np.abs(KL_modes), axis=(1, 2), keepdims=True)
    KL_modes = torch.tensor(KL_modes, dtype=torch.float32)
    return KL_modes


def get_KL_wavefront(KL_modes, n_modes_max, n_images, coef_range=2.0):
    coef = torch.FloatTensor(n_images, n_modes_max).uniform_(-coef_range, coef_range)
    # coef = torch.FloatTensor(n_images, n_modes_max).uniform_(0, 1)
    KL_wavefront = torch.einsum('kij,lk->lij', KL_modes, coef)
    return KL_wavefront


def generate_PSFs(wavefront, n_images):
    PSFS = torch.stack([PSF(torch.exp(1j * wavefront[i, :, :])) for i in range(n_images)], -1)
    #PSFS = PSFS / (torch.sum(PSFS, dim=(0, 1), keepdim=True))
    return PSFS


def get_convolution(simulation, psfs, n_images, noise=False):
    convolved_images = np.stack([convolve2d(simulation[..., 0], psfs[:, :, i], boundary='symm', mode='same') for i in range(n_images)],
                                -1)
    if noise:
        noise = np.random.normal(1, 0.005, size=convolved_images.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * 0.1
        convolved_images += noise
    convolved_images = np.stack([convolved_images, convolved_images], -1)
    return convolved_images

def get_convolution_einsum(simulation, psfs, n_images, noise=False):
    """
    simulation: torch.Tensor of shape (H, W, 1)
    psfs: torch.Tensor of shape (kH, kW, n_images)
    """
    H, W, _ = simulation.shape
    kH, kW, _ = psfs.shape
    pad_h, pad_w = kH // 2, kW // 2

    # Convert simulation to shape (1, 1, H, W)
    simulation = simulation.permute(2, 0, 1).unsqueeze(0)  # shape: (1, 1, H, W)

    # Apply symmetric (reflect) padding
    simulation = F.pad(simulation, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # Unfold into sliding patches: shape (1, kH*kW, L)
    patches = F.unfold(simulation, kernel_size=(kH, kW))  # No padding here

    # Reshape PSFs to (n_images, kH*kW)
    psfs_flat = psfs.reshape(-1, n_images).T  # (n_images, kH*kW)

    # Convolve using einsum
    convolved = torch.einsum('ik,bkl->bil', psfs_flat, patches)  # (n_images, L)

    # Reshape back to (n_images, H, W)
    convolved_images = convolved.view(n_images, H, W).permute(1, 2, 0)  # (H, W, n_images)

    if noise:
        noise_tensor = torch.normal(1.0, 0.005, size=convolved_images.shape, device=convolved_images.device)
        convolved_images = convolved_images * noise_tensor

    # Duplicate across last dim
    convolved_images = torch.stack([convolved_images, convolved_images], dim=-1)  # (H, W, n_images, 2)

    return convolved_images


def compute_rms_contrast(image):
    mean = np.mean(image)
    rms = np.sqrt(np.mean((image - mean) ** 2))
    return rms


def cutout(image, x, y, size):
    return image[x - size // 2:x + size // 2, y - size // 2:y + size // 2, :, 0]


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


def gaussian_psf(size, sigma, n_images):
    """
    Create a Gaussian PSF (Point Spread Function) with the given size, standard deviation (sigma) and number of images.

    Parameters:
    - size (int): Size of the PSF (size x size).
    - sigma (float): Standard deviation of the Gaussian.
    - n_images (int): Number of images to create.

    Returns:
    - np.ndarray: Gaussian PSF.
    """
    x = np.linspace(-size[0] // 2, size[0] // 2, size[0])
    y = np.linspace(-size[1] // 2, size[1] // 2, size[1])
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    psf = np.stack([psf] * n_images, axis=-1)  # Stack for n_images
    return psf

def generate_gaussian_psf(size, sigma):
    """Generate a 2D Gaussian PSF of given size and sigma using NumPy only."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= np.sum(psf)  # Normalize to make it a proper PSF
    return psf


def limb_darkening_correction(s_map, limb_offset=0.99, poly_order=4):
    """
    Apply limb darkening correction to a SunPy map.

    Parameters
    ----------
    s_map : sunpy.map.Map
        Input solar map.
    limb_offset : float, optional
        Radial cutoff in units of observed solar radius. Pixels at or above this
        value are ignored for fitting.
    poly_order : int, optional
        Polynomial degree used to fit intensity as a function of ideal correction.

    Returns
    -------
    sunpy.map.Map
        Limb-darkening corrected map.
    """
    coords = all_coordinates_from_map(s_map)
    radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value

    valid_radius = radial_distance < limb_offset
    ideal_correction = np.cos(radial_distance * np.pi / 2)
    ideal_correction[~valid_radius] = np.nan

    condition = np.isfinite(ideal_correction) & np.isfinite(s_map.data)
    map_list = s_map.data[condition]
    correction_list = ideal_correction[condition]

    if map_list.size == 0:
        raise ValueError("No valid pixels available for limb darkening fit.")

    fit = np.polyfit(correction_list, map_list, poly_order)
    poly_fit = np.poly1d(fit)

    map_correction = poly_fit(ideal_correction)
    map_correction[~np.isfinite(map_correction)] = np.nan
    map_correction[map_correction == 0] = np.nan

    corrected_map = s_map.data / map_correction
    return Map(corrected_map, s_map.meta)
