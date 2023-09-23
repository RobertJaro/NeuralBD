import numpy as np
import torch


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
    return 206265.0 * wavelength * 1e-8 / (telescope_diameter * simulation_pixel_size)

def lofdahl_scharmer_filter(Sconj_S, Sconj_I):
    den = torch.conj(Sconj_I) * Sconj_I
    H = (Sconj_S / den).real

    H = torch.fft.fftshift(H).detach().cpu().numpy()

    # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

    H = nd.median_filter(H, [1, 1, 3, 3], mode='wrap')

    filt = 1.0 - H * self.sigma[:, :, None, None].cpu().numpy() ** 2 * self.config['n_pixel'] ** 2
    filt[filt < 0.2] = 0.0
    filt[filt > 1.0] = 1.0

    nb, no, nx, ny = filt.shape

    mask = np.zeros_like(filt)

    for ib in range(nb):
        for io in range(no):
            mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx // 2, ny // 2),
                                       tolerance=0.9) * self.mask_diffraction_shift
            mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

    return torch.tensor(mask).to(Sconj_S.device)
