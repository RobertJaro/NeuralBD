from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from sunpy.map import Map, make_fitswcs_header
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from nstack.data.KL_modes import KL

from nstack.data.psfs import PSF


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


class LoadGREGORLevel1Data(Editor):
    def call(self, path, **kwargs):
        fits_array = [fits.getdata(path, i) for i in range(200)]
        h, w = fits_array[0].shape
        fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))
        return fits_array


class LoadGREGORSpeckleData(Editor):
    def call(self, path, **kwargs):
        fits_array = [fits.getdata(path, i) for i in range(2)]
        fits_array = np.stack(fits_array, -1)
        return fits_array


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


class ReadNumpyEditor(Editor):

    def call(self, filename, **kwargs):
        data = np.load(filename)
        data = data.transpose(1, 2, 0)
        return data

class NormalizeSimulationEditor(Editor):

    def call(self, data, **kwargs):
        data = data.data
        vmin, vmax = 0, np.percentile(data, 99)
        sim_norm = (data - vmin) / (vmax - vmin)
        sim_stack = np.stack([sim_norm, sim_norm], -1)
        return sim_stack

class CropSimulationEditor(Editor):

    def call(self, data, **kwargs):
        sim_crop = data[250:762, 250:762, :]
        return sim_crop


def get_KL_basis(n_modes_max, size):
    kl = KL()
    KL_modes = kl.precalculate_covariance(npix_image=size, n_modes_max=n_modes_max, first_noll=2)
    KL_modes = torch.tensor(KL_modes, dtype=torch.float32)
    return KL_modes

def get_KL_wavefront(KL_modes, n_modes_max, n_images, coef_range=2):
    coef = torch.FloatTensor(n_images, n_modes_max).uniform_(-coef_range, coef_range)
    #coef = torch.FloatTensor(n_images, n_modes_max).uniform_(0, 1)
    KL_wavefront = torch.einsum('kij,lk->lij', KL_modes, coef)
    return KL_wavefront

def get_PSFs(wavefront, n_images):
    PSFS = torch.stack([PSF(torch.exp(1j * wavefront[i, :, :])) for i in range(n_images)], -1)
    #PSFS = PSFS[60:69, 60:69, :]
    PSFS = PSFS / (torch.sum(PSFS, dim=(0, 1)))
    return PSFS

def get_convolution(simulation, psfs, n_images):
    convolved_images = np.stack([cv2.filter2D(simulation[..., 0], -1, psfs[:, :, i].numpy()) for i in range(n_images)], -1)
    convolved_images = np.stack([convolved_images, convolved_images], -1)
    return convolved_images