import numpy as np
import torch
from scipy import ndimage as nd
from skimage.morphology import flood


def aperture(npix=256, cent_obs=0.0, spider=0, overfill=1.0):
    """
    Compute the aperture image of a telescope

    Args:
        npix (int, optional): number of pixels of the aperture image
        cent_obs (float, optional): central obscuration fraction
        spider (int, optional): spider size in pixels

    Returns:
        real: returns the aperture of the telescope
    """
    illum = np.ones((npix, npix), dtype='d')
    x = np.arange(-npix / 2, npix / 2, dtype='d')
    y = np.arange(-npix / 2, npix / 2, dtype='d')

    xarr = np.outer(np.ones(npix, dtype='d'), x)
    yarr = np.outer(y, np.ones(npix, dtype='d'))

    rarr = np.sqrt(np.power(xarr, 2) + np.power(yarr, 2)) / (npix / 2)
    outside = np.where(rarr > 1.0 / overfill)
    inside = np.where(rarr < cent_obs)

    illum[outside] = 0.0
    if np.any(inside[0]):
        illum[inside] = 0.0

    if (spider > 0):
        start = int(npix / 2 - int(spider) / 2)
        illum[start:start + int(spider), :] = 0.0
        illum[:, start:start + int(spider)] = 0.0

    return illum

def psf_scale(wavelength, telescope_diameter, simulation_pixel_size):
    """
    Return the PSF scale appropriate for the required pixel size, wavelength and telescope diameter
    The aperture is padded by this amount; resultant pix scale is lambda/D/psf_scale, so for instance full frame 256 pix
    for 3.5 m at 532 nm is 256*5.32e-7/3.5/3 = 2.67 arcsec for psf_scale = 3

    https://www.strollswithmydog.com/wavefront-to-psf-to-mtf-physical-units/#iv

    """
    return 206265.0 * wavelength * 1e-7 / (telescope_diameter * simulation_pixel_size)

def lofdahl_scharmer_filter(Sconj_S, Sconj_I):
    den = torch.conj(Sconj_I) * Sconj_I
    H = (Sconj_S / den).real

    H = torch.fft.fftshift(H).detach().cpu().numpy()

    # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

    H = nd.median_filter(H, [1, 1, 3, 3], mode='wrap')

    filt = 1.0 - H * sigma[:, :, None, None].cpu().numpy() ** 2 * self.config['n_pixel'] ** 2
    filt[filt < 0.2] = 0.0
    filt[filt > 1.0] = 1.0

    nb, no, nx, ny = filt.shape

    mask = np.zeros_like(filt)

    for ib in range(nb):
        for io in range(no):
            mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx // 2, ny // 2),
                                       tolerance=0.9) * mask_diffraction_shift
            mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

    return torch.tensor(mask).to(Sconj_S.device)

def compute_image(self, images_ft, psf_ft, type_filter='lofdahl_scharmer'):
    """Compute the reconstructed image

    """

    Sconj_S = torch.sum(torch.conj(psf_ft) * psf_ft)
    Sconj_I = torch.sum(torch.conj(psf_ft) * images_ft)

    # Use Lofdahl & Scharmer (1994) filter
    if (type_filter == 'lofdahl_scharmer'):
        mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I)

        out = Sconj_I / Sconj_S * mask

        # Use simple Wiener filter with Gaussian prior
    #if (type_filter == 'gaussian'):
    #    out = Sconj_I / (self.sigma[:, :, None, None] + Sconj_S) * torch.fft.fftshift(
    #        self.mask_diffraction_th[None, None, :, :])
    return out

def noise_estimation(x):
    """
    Returns the optimal threshold of the singular values of a matrix, i.e., it computes the optimal number
    of singular values to keep or the optimal rank of the matrix.
    Given a matrix Y=X+sigma*Z, with Y the observed matrix, X the matrix that we want to find, Z an matrix whose elements are iid Gaussian
    random numbers with zero mean and unit variance and sigma is the standard deviation of the noise, the matrix X can be recovered in the
    least squares sense using a truncated SVD (compute the SVD of the matrix, keep only the relevant singular values and reconstruct the
    matrix). The problem is estimate the number of such singular values to keep. This function uses the theory of Gavish & Donoho (2014)
    that is valid for matrices with sizes larger than its rank, i.e., for low-rank matrices

    Args:
        matrix: matrix to estimate the rank

    Returns:
        thrKnownNoise: in case sigma is known, multiply this value with sigma and this gives the threshold
        thrUnknownNoise: returns the threshold directly
        noiseEstimation: returns the estimation of the noise
        wSVD: returns the singular values of the matrix
    """

    m, n = x.shape[-2:]
    beta = 1.0 * m / n

    w = (8.0 * beta) / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1))
    lambdaStar = np.sqrt(2.0 * (beta + 1) + w)

    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    if (m > n):
        cov = np.swapaxes(x, -2, -1) @ x
    else:
        cov = x @ np.swapaxes(x, -2, -1)

    uSVD, wSVD, vSVD = np.linalg.svd(cov)

    medianSV = np.median(np.sqrt(wSVD), axis=-1)

    muSqrt = lambdaStar / omega
    noiseEstimation = medianSV / (np.sqrt(n) * muSqrt)

    return noiseEstimation