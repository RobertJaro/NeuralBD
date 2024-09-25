import os

import astropy
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib.colors import LogNorm
from astropy.convolution import Gaussian2DKernel
from torch import nn, dtype
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from zernike import RZern
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nstack.data.Editor import ReadSimulationEditor
from nstack.data.psfs import *
from nstack.data.KL_modes import KL

# 60 arcsec FOV - ref
# step size of 30 arcsec --> overlap of 50%
# pixel size of 1/20 arcsec --> 600 pixels
base_path = '/mnt/disks/data/neuralBD/Training/muram_kl15_find_psf'
data_path = '/mnt/disks/data/neuralBD/I_out_med.468000'
#data_path = '/mnt/disks/data/neuralBD/hifi_20170618_082414_sd.fts'
#data_path2 = '/mnt/disks/data/neuralBD/hifi_20170618_082414_sd_speckle.fts'


### simulation data
muram = ReadSimulationEditor().call('/mnt/disks/data/neuralBD/I_out_med.468000')
sim_array = muram.data
vmin, vmax = 0, np.percentile(sim_array, 99)
fits_array2_norm = (sim_array - vmin) / (vmax - vmin)
fits_array2_stack = np.stack([fits_array2_norm, fits_array2_norm], -1)
fits_array2 = fits_array2_stack[200:712, 200:712, :]
#fits_array2 fits_array2_stack
os.makedirs(base_path, exist_ok=True)
plt.imsave(os.path.join(base_path, 'muram_simulation.jpg'), fits_array2[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')


### Initialize PSF parameters
def gaussian_2d(x, y, mean, sigma):
    exponent = -((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / (2 * sigma ** 2)
    return torch.exp(exponent)

psf_size = [15, 15]
psf_sampling_size = [15, 15]

x_values = torch.linspace(-1., 1., psf_sampling_size[0], dtype=torch.float32)
y_values = torch.linspace(-1., 1., psf_sampling_size[1], dtype=torch.float32)
x, y = torch.meshgrid(x_values, y_values, indexing='ij')

# Generate random means and standard deviations
n_images = 5
means = np.random.uniform(-0.2, 0.2, size=(n_images, 2))
stds = np.random.uniform(0.01, 0.9, size=n_images)
means = torch.tensor(means, dtype=torch.float32)
stds = torch.tensor(stds, dtype=torch.float32)

# Generate gaussian PSFs
#psfs = torch.stack([gaussian_2d(x, y, mean, std) for mean, std in zip(means, stds)], -1)
##psfs[1, 1, :] = 1
#norm = psfs.sum(axis=(0, 1), keepdims=True)
#psfs = psfs / norm

# generate normal distributed PSFs
#psfs = torch.randn(*psf_size, n_images)
#psfs = (psfs - torch.min(psfs)) / (torch.max(psfs) - torch.min(psfs))

#Zernike Basis
#print('Generating Zernike PSFs: ')
#zernike_phi = torch.zeros([psf_size[0], psf_size[1], n_images], dtype=torch.float32)
#for z in tqdm(range(n_images)):
#    modes = 44
#    wave_coef = np.random.rand(modes)
#    wave_coef = torch.tensor(wave_coef, dtype=torch.float32)
#    cart = RZern(modes)
#    cart.make_cart_grid(x.numpy() / 2, y.numpy() / 2)
#    zerneke_polynomials = torch.zeros((*x.shape, modes))
#    c = np.zeros(cart.nk)
#    plt.figure(figsize=(10, 10))
#    for i in range(1, modes+1):
#        plt.subplot(10, 10, i)
#        c *= 0.0
#        c[i] = 1.0
#        Phi = cart.eval_grid(c, matrix=True)
#        Phi = torch.tensor(Phi, dtype=torch.float32)
#        zerneke_polynomials[:, :, i - 1] = Phi
#        zerneke_polynomials = zerneke_polynomials
#        zernike_base = wave_coef[None, None, :] * zerneke_polynomials
#        plt.show()
#        zernike_phi[:, :, z] += zernike_base.sum(-1)


#psfs = PSF(zernike_phi)
#np.save(os.path.join(base_path, 'psfs.npy'), psfs)
#fig, axs = plt.subplots(10, 10, figsize=(20, 20))
#for i, ax in enumerate(np.ravel(axs)):
#    im = ax.imshow(zernike_phi[:, :, i], origin='lower')
#    ax.set_axis_off()
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    fig.colorbar(im, cax=cax, orientation='vertical')
#plt.tight_layout()
#plt.savefig(os.path.join(base_path, f'Wavefronts.jpg'), dpi=300)
#plt.close(fig)



## Karhun-Loeve Base
kl = KL()
n_modes_max = 11
KL_modes = kl.precalculate_covariance(npix_image=psf_size[0], n_modes_max=n_modes_max, first_noll=3)
KL_modes = torch.tensor(KL_modes.transpose(1, 2, 0), dtype=torch.float32)
#f, ax = plt.subplots(nrows=4, ncols=4)
#for i in range(16):
#    ax.flat[i].imshow(KL.KL[i, :, :])
#plt.show()

coef = torch.FloatTensor(n_modes_max, n_images).uniform_(0, 1)
KL_wavefront = torch.einsum('ijk,kl->ijl', KL_modes, coef)

# Plot KL wavefronts
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i, ax in enumerate(np.ravel(axs)):
    im = ax.imshow(KL_wavefront[:, :, i], origin='lower')
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'Wavefronts.jpg'), dpi=300)

PSFS = torch.stack([PSF(torch.exp(1j * KL_wavefront[:, :, i])) for i in range(n_images)], -1)

#np.save(os.path.join(base_path, 'psfs.npy'), psfs)
#psfs = torch.tensor(zernike_psfs, dtype=torch.float32)
#psfs = (psfs - torch.min(psfs)) / (torch.max(psfs) - torch.min(psfs))
#norm = psfs.sum(axis=(0, 1), keepdims=True)
#psfs = psfs / norm

# Generate convolved images
images = np.stack([astropy.convolution.convolve(fits_array2[..., 0], PSFS[:, :, i], boundary='wrap') for i in range(n_images)], -1)

psf_norm = LogNorm(vmin=1e-4, vmax=1)
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i, ax in enumerate(np.ravel(axs)):
    im = ax.imshow(PSFS[:, :, i], origin='lower')
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'psfs.jpg'), dpi=300)
plt.close(fig)
#vmin_conv, vmax_conv = 0, np.percentile(images, 99)
#images = (images - vmin_conv) / (vmax_conv - vmin_conv)
#images = images[200:712, 200:712, :]
images = np.stack([images, images], -1)
#images = (images - vmin) / (vmax - vmin)

model_path = os.path.join(base_path, f'model.pt')

#offset = 200
## GREGOR Data
#fits_array = []

#for i in range(0, 200): # <-- 2022: 0, 200 ; 2022-->: 1,201
#    fits_array.append(fits.getdata(data_path, i))

#h, w = fits_array[0].shape
#fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))

#fits_array2 = []
#for i in range(0, 2):
#    fits_array2.append(fits.getdata(data_path2, i))
#fits_array2 = np.stack(fits_array2, -1)

#fits_array = fits_array[:, :, :n_images]
#fits_array = fits_array[200:1224, 200:1224, :, :]
#fits_array2 = fits_array2[200:1224, 200:1224, :]
##fits_array = fits_array[600:1624, 600:1624, :, :]
##fits_array2 = fits_array2[600:1624, 600:1624, :]
#vmin, vmax = fits_array.min(), np.percentile(fits_array, 99)
#vmin2, vmax2 = fits_array2.min(), fits_array2.max()

#plt.imsave(os.path.join(base_path, 'speckle_data1.jpg'), fits_array2[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
#plt.imsave(os.path.join(base_path, 'speckle_data2.jpg'), fits_array2[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
#positions = np.array(positions)
#images = fits_array / vmax
#images = (fits_array - vmin) / (vmax - vmin)
#fits_array2 = (fits_array2 - vmin2) / (vmax2 - vmin2)

im_scaling = (images.shape[0] - 1) / 2
im_shift = (images.shape[0] - 1) / 2

for i in range(0, images.shape[2], 10):
    ref_image = images[:, :, i]
    # plot first two channels
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(ref_image[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
    axs[1].imshow(ref_image[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
    plt.savefig(os.path.join(base_path, f'images{i:03d}.jpg'))
    plt.close(fig)

def rms_contrast(image):
    image_diff = (image - torch.mean(image)) ** 2
    return torch.sqrt(torch.mean(image_diff))


def gaussian_2D(x, y, mean, sigma):
    exponent = -((x - mean[..., 1]) ** 2 + (y - mean[..., 0]) ** 2) / (2 * sigma ** 2)
    return torch.exp(exponent)

#images = block_reduce(images, block_size=(4, 4, 1, 1), func=np.mean)

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)  # (num_freqs)
        #freqs = torch.randn((1, num_freqs, in_coords), dtype=torch.float32)
        #self.freqs = nn.Parameter(freqs, requires_grad=True)


    def forward(self, x):
        #encoded = x[:, None, :] * self.freqs
        #encoded = encoded.reshape(x.shape[0], -1)
        #out = torch.cat([torch.sin(encoded), torch.cos(encoded)],
        #                dim=-1)  # (batch, 2*num_freqs*in_features)
        x_proj = x[:, None, :] * self.freqs[None, :, None]  # (batch, num_freqs, in_features)
        x_proj = x_proj.reshape(x.shape[0], -1)  # (batch, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (batch, 2*num_freqs*in_features)
        return out


class ImageModel(nn.Module):

    def __init__(self, dim, n_channels=2):
        super().__init__()
        posenc = PositionalEncoding(8, 20)
        d_in = nn.Linear(2 * 40, dim)
        self.d_in = nn.Sequential(posenc, d_in)

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_channels)
        self.activation = Sine()
        self.out_activation = nn.Softplus()

    def forward(self, coords):
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        x = self.out_activation(x)
        return x

class PSFModel(nn.Module):

    def __init__(self, dim, n_images):
        super().__init__()
        d_in = nn.Linear(2, dim)
        self.d_in = d_in

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_images)
        self.activation = Sine()

    def forward(self, coords):
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = torch.sigmoid(self.d_out(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_images):
        super().__init__()
        #self.activation = Sine()
        #self.d_in = nn.Linear(coord_dim, dim)
        #lin = [nn.Linear(dim, dim) for _ in range(4)]
        #self.layers = nn.ModuleList(lin)
        #self.d_out = nn.Linear(dim, coord_dim)
        self.transform_matrix = nn.Parameter(torch.ones(n_images - 1, 2, 2, dtype=torch.float32), requires_grad=True)

    def forward(self, coords):
        #x = self.activation(self.d_in(coords))
        #for l in self.layers:
        #    x = self.activation(l(x))
        #x = self.d_out(x)
        first_img_coords = coords[:, 0:1]
        coords = coords[:, 1:]

        #extended_coords  = torch.cat([coords, torch.ones_like(coords)[..., 0:1]], -1)
        # (batch, n_images - 1, 3)
        #extended_coords = torch.einsum('nij,bnj->bni', self.transform_matrix, extended_coords)
        #coords = extended_coords[..., :2]
        coords = torch.einsum('nij,bnj->bni', self.transform_matrix, coords)

        coords = torch.cat([first_img_coords, coords], -2)
        return coords

class ImageStackModel(nn.Module):

    def __init__(self, n_images, dim):
        # images = (w, h, n, c)
        super().__init__()
        self.n_images = n_images
        self.image_models = nn.ModuleList([ImageModel(dim) for _ in range(self.n_images)])
        self.coord_transform = Transformer(n_images)

    def get_transformed_images(self, coords):
        transformed_coords = self.transform_coords(coords)
        image_stack = torch.stack([model(transformed_coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def get_images(self, coords):
        image_stack = torch.stack([model(coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def transform_coords(self, coords):
        transformed_coords = self.coord_transform(coords)
        return transformed_coords

    def forward(self, coords, transform=False):
        if transform:
            coords = self.transform_coords(coords)
        image_stack = self.get_images(coords)
        return image_stack

class PSFStackModel(nn.Module):

    def __init__(self, n_images, dim):
        # images = (w, h, n, c)
        super().__init__()

        self.wavefront_coefficient = nn.Parameter(torch.ones(n_modes_max, n_images, dtype=torch.float32),
                                                  requires_grad=True)

        self.n_images = n_images
        self.modes = 11
        self.image_model = ImageModel(dim)
        self.psf_model = PSFModel(dim, n_images)
        #self.sigma = nn.Parameter(torch.ones(n_images, dtype=torch.float32) * 0.2, requires_grad=True)
        #self.mean = nn.Parameter(torch.zeros(n_images, 2, dtype=torch.float32), requires_grad=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        psf_size = [15, 15]
        psf_sampling_size = [15, 15]
        #
        x_values_psf = torch.linspace(-1., 1., psf_sampling_size[0], dtype=torch.float32)
        y_values_psf = torch.linspace(-1., 1., psf_sampling_size[1], dtype=torch.float32)
        psf_x, psf_y = torch.meshgrid(x_values_psf, y_values_psf, indexing='ij')
        self.register_buffer('psf_x', psf_x)
        self.register_buffer('psf_y', psf_y)

        #psf = torch.randn(*psf_size, n_images)
        #psf[psf_size[0] // 2, psf_size[1] // 2, :] = 1
        #self.psf_model = nn.Parameter(psf, requires_grad=True)
        self.softplus = nn.Softplus()


        x_values = torch.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_sampling_size[0], dtype=torch.float32)
        y_values = torch.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_sampling_size[1], dtype=torch.float32)
        x, y = torch.meshgrid(x_values, y_values, indexing='ij')
        grid_sampling = torch.stack([x, y], -1).reshape(-1, 2)  # batch, psf_coords, xy
        grid_sampling = grid_sampling / im_scaling
        self.grid_sampling = nn.Parameter(grid_sampling, requires_grad=False)

    def get_transformed_images(self, coords):
        transformed_coords = self.transform_coords(coords)
        image_stack = torch.stack([model(transformed_coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack


    def get_images(self, coords):
        #n_random_sample = 200
        # r spacing
        # dr = 0.01
        # r_values = torch.arange(dr, 16 * dr, step=dr, dtype=torch.float32, device=coords.device)
        # # theta spacing
        # theta_values = torch.linspace(0, 2 * torch.pi, 32, dtype=torch.float32, device=coords.device)
        # dtheta = torch.diff(theta_values)[0]
        # #
        # # create mesh
        # theta, r = torch.meshgrid(theta_values, r_values, indexing='ij')
        # #
        # # compute area elements
        # area_elements = r * dr * dtheta
        # #
        # # convert to cartesian
        # x, y = r * torch.cos(theta), r * torch.sin(theta)
        # grid_sampling = torch.stack([x, y], -1).reshape(-1, 2)
        # area_elements = area_elements.reshape(-1, 1)
        # #
        # # add central point
        # central_point = torch.zeros(1, 2, device=coords.device)
        # central_area = torch.tensor(torch.pi * (dr / 2) ** 2, dtype=torch.float32, device=coords.device).reshape(1, 1)
        # grid_sampling = torch.concatenate([grid_sampling, central_point])
        # area_elements = torch.concatenate([area_elements, central_area])
        # #
        # grid_sampling = grid_sampling[None, :, :]
        # area_elements = area_elements[None, :, :]

        # area_elements = torch.ones_like(grid_sampling)[:, :, :1]

        #random_sampling = torch.randn(coords.shape[0], n_random_sample, 2, device=coords.device) * self.psf_width
        # coords: batch, xy
        grid_sampling_coords = coords[:, None, :] - self.grid_sampling[None, :, :] # --> batch, psf_coords, xy
        #
        condition = (grid_sampling_coords[..., 0] >= -1) & (grid_sampling_coords[..., 0] <= 1) & \
                   (grid_sampling_coords[..., 1] >= -1) & (grid_sampling_coords[..., 1] <= 1)
        condition = condition[..., None]
        #
        psf = self.get_psf().to(coords.device)
        flat_psf = psf.reshape(-1, self.n_images)
        #psf = psf[None].repeat(coords.shape[0], 1, 1)

        #psf = self.get_psf().to(coords.device)
        #flat_psf = psf.reshape(-1, self.n_images)
        # psf = psf[None].repeat(coords.shape[0], 1, 1)

        #psf = self.get_psf().to(coords.device)
        #flat_psf = psf.reshape(-1, n_images)


        sampling_image = self.image_model(grid_sampling_coords.reshape(-1, 2))
        sampling_image = sampling_image.reshape(coords.shape[0], -1, sampling_image.shape[-1]) # batch, psf_coords, channels
        # sampling_image = sampling_image * condition # set values outside of image to 0
        convolved_images = torch.einsum('...sc,sn->...nc', sampling_image, flat_psf)
        #convolved_images = torch.einsum('bsc,bsn->bnc', sampling_image, psf)
        image = self.image_model(coords)
        return image, convolved_images

    def get_psf(self):
        # (x, y, 21); (100, 21)
        # (x, y, 1, 21) * (1, 1, 100, 21) --> (x, y, 100)

        #zern_psfs = (self.zerneke_polynomials[:, :, None, :] * self.zerneke_params[None, None, :, :]).sum(-1)
        #zern_psfs = zern_psfs * self.wavefront_coefficient[None, None, :]
        #zern_psfs = (zern_psfs - torch.min(zern_psfs)) / (torch.max(zern_psfs) - torch.min(zern_psfs))

        #Zernike Basis
        #modes = 44
        #cart = RZern(modes)
        #cart.make_cart_grid(self.psf_x.cpu().numpy() / 2, self.psf_y.cpu().numpy() / 2)

        #zerneke_polynomials = torch.zeros((*self.psf_x.shape, modes))
        #c = np.zeros(cart.nk)
        #zernike_phi = torch.zeros([psf_size[0], psf_size[1], n_images])
        #for z in tqdm(range(n_images)):
        #    # plt.figure(figsize=(10, 10))
        #    for i in range(1, modes + 1):
        #        # plt.subplot(10, 5, i)
        #        c *= 0.0
        #        c[i] = 1.0
        #        Phi = cart.eval_grid(c, matrix=True)
        #        Phi = torch.tensor(Phi, dtype=torch.float32)
        #        zerneke_polynomials[:, :, i - 1] = Phi
        #        zerneke_polynomials = zerneke_polynomials
        #        zernike_base = self.wavefront_coefficient[None, None, :].to(zerneke_polynomials.device) * zerneke_polynomials

        #        zernike_phi[:, :, z] += zernike_base.sum(-1)

        #pupil = torch.exp(1j * zernike_phi)

        #D = 150  # diameter of the aperture
        #lam = 650.5 * 10 ** (-6)  # wavelength of observation
        #pix = 0.5  # plate scale
        #f = 4125.3  # effective focal length
        #size = psf_size[0]  # size of detector in pixels


        #psfs = torch.zeros([size, size, n_images])
        #for j in range(n_images):
        #    modes = 6
        #    coefficients = nn.Parameter(torch.rand(modes, dtype=torch.float32), requires_grad=True)
        #    rpupil = pupil_size(D, lam, pix, size).to(torch.int)
        #    sim_phase = center(coefficients, size, rpupil)
        #    Mask = mask(rpupil, size)
        #    # Mask = np.ones((size, size))
        #    pupil_com = complex_pupil(sim_phase, Mask)
        #    psf = PSF(pupil_com)
        #    psfs[:, :, j] += psf


        wavefront = torch.einsum('ijk,kl->ijl', KL_modes.to(self.device), self.wavefront_coefficient)

        # Karhun-Loeve Base
        psfs = torch.stack([PSF(torch.exp(1j * wavefront[:, :, i])) for i in range(n_images)], -1)
        #complx_wavefront = torch.exp(1j * wavefront)
        #psfs = PSF(complx_wavefront)

        #complx_wavefront = torch.exp(1j * wavefront)
        #psfs = PSF(complx_wavefront)
        #psfs = np.load(os.path.join(base_path, 'psfs.npy'))
        #psfs = torch.tensor(psfs, dtype=torch.float32)
        #psfs = PSF(pupil.detach().numpy())
        #psfs = np.load(os.path.join(base_path, 'psfs.npy'))
        #zernike_psfs = torch.tensor(psfs, dtype=torch.float32)
        # self.register_buffer('zerneke_polynomials', zerneke_polynomials)

        #psfs = zernike_psfs
        #psfs = (psfs - torch.min(psfs)) / (torch.max(psfs) - torch.min(psfs))

        #psf = gaussian_2D(self.psf_x[:, :, None], self.psf_y[:, :, None], self.mean[None, None, :, :], self.sigma[None, None, :])
        #psfs = psf

        #psfs = self.psf_model(self.grid_sampling.reshape(-1, 2))
        #psfs = psfs.reshape(psf_size[0], psf_size[1], self.n_images)

        #psf = self.softplus(self.psf_model)
        #norm = zern_psfs.sum(dim=(0, 1), keepdims=True)
        #norm = psfs.sum(dim=(0, 1), keepdims=True)
        return psfs

    def forward(self, coords):
        return self.get_images(coords)

# optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_images = images.shape[-2]
model = PSFStackModel(n_images, 256)
parallel_model = nn.DataParallel(model)
parallel_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# loss
loss_fn = torch.nn.HuberLoss()


# training
epochs = 10000
batch_size = 2048 * torch.cuda.device_count() if torch.cuda.is_available() else 1024


shift_range = 0.02


image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)

#coordinates_tensor = (coordinates_tensor - im_shift) / im_scaling
coordinates_tensor = coordinates_tensor / im_scaling

# check coordinates
print(f'MIN;  {coordinates_tensor[..., 0].min()} , {coordinates_tensor[..., 1].min()}')
print(f'MAX;  {coordinates_tensor[..., 0].max()} , {coordinates_tensor[..., 1].max()}')
print(f'PSF coordinates MIN; {model.grid_sampling[..., 0].min()} , {model.grid_sampling[..., 1].min()}')
print(f'PSF coordinates MAX; {model.grid_sampling[..., 0].max()} , {model.grid_sampling[..., 1].max()}')

# image_tensor = torch.from_numpy(images).float().view(-1, 1)
image_tensor = torch.from_numpy(images).float().reshape(-1, n_images, 2)
print(f'Image tensor: SHAPE {image_tensor.shape}; MIN {image_tensor.min()}; MAX {image_tensor.max()}')

test_coords = coordinates_tensor

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    print('model restored')



for epoch in range(epochs):
    r = torch.randperm(len(image_tensor))
    image_tensor = image_tensor[r]
    coordinates_tensor = coordinates_tensor[r]

    total_loss = []
    radial_regularization_loss = []
    huber_loss = []
    for i in tqdm(range(np.ceil(len(coordinates_tensor) / batch_size).astype(int))):
        batch_coordinates = coordinates_tensor[i * batch_size:(i + 1) * batch_size]
        batch_image = image_tensor[i * batch_size:(i + 1) * batch_size]
        batch_coordinates = batch_coordinates.to(device)
        batch_image = batch_image.to(device)

        img, convolved_imgs = parallel_model(batch_coordinates)
        psf = model.get_psf().to(device)

        #x_grid = torch.linspace(-1, 1, psf.shape[0], device=device, dtype=torch.float32)
        #y_grid = torch.linspace(-1, 1, psf.shape[1], device=device, dtype=torch.float32)
        #xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')

        #radial = xx ** 2 + yy ** 2
        #radial_regularization_term = (psf * radial[:, :, None])
        #radial_regularization_term = (psf * radial[:, :, None]) ** 2

        #loss = loss_fn(convolved_imgs, batch_image) + (1 - rms_contrast(img)) * 1e-3
        #loss = loss_fn(convolved_imgs, batch_image)
        image_loss = loss_fn(convolved_imgs, batch_image)
        #loss = image_loss + radial_regularization_term.mean() * 1e-2
        loss = image_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(0, 1)

        total_loss += [loss.detach().cpu().numpy()]
        #radial_regularization_loss += [radial_regularization_term.detach().cpu().numpy()]
        huber_loss += [image_loss.detach().cpu().numpy()]

    with torch.no_grad():
        # print loss
        print(f'epoch: {epoch + 1}; total_loss: {np.mean(total_loss)}')
        #print(f'epoch: {epoch + 1}; regularization_loss: {np.mean(radial_regularization_loss)}')
        print(f'epoch: {epoch + 1}; image_loss: {np.mean(huber_loss)}')
        # print pretty
        if (epoch + 1) % 5 != 0:
            continue

        output_image = []
        output_convolved_image = []
        for i in range(np.ceil(len(test_coords) / batch_size).astype(int)):
            batch_coordinates = test_coords[i * batch_size:(i + 1) * batch_size]
            batch_coordinates = batch_coordinates.to(device)

            output_img, convolved_imgs = parallel_model(batch_coordinates)

            output_image += [output_img.detach().cpu().numpy()]
            output_convolved_image += [convolved_imgs.detach().cpu().numpy()]

        output_image = np.concatenate(output_image, 0).reshape((*image_coordinates.shape[:-1], 2))
        output_convolved_image = np.concatenate(output_convolved_image, 0).reshape((*image_coordinates.shape[:-1], n_images, 2))

        fig, axs = plt.subplots(4, 2, figsize=(6, 6))
        im = axs[0, 0].imshow(fits_array2[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[0, 0])
        im = axs[0, 1].imshow(fits_array2[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[0, 1])
        im = axs[1, 0].imshow(output_image[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[1, 0])
        im = axs[1, 1].imshow(output_image[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[1, 1])
        im = axs[2, 0].imshow(output_convolved_image[..., 1, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[2, 0])
        im = axs[2, 1].imshow(output_convolved_image[..., 1, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[2, 1])
        im = axs[3, 0].imshow(ref_image[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[3, 0])
        im = axs[3, 1].imshow(ref_image[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=axs[3, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'images_{epoch + 1:04d}.jpg'), dpi=300)
        plt.close(fig)

        plt.imsave(os.path.join(base_path, f'output_image_0_{epoch + 1:04d}.jpg'), output_image[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        #plt.imsave(os.path.join(base_path, f'output_image_1_{epoch + 1:04d}.jpg'), output_image[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.imsave(os.path.join(base_path, f'output_convolved_image_0_{epoch + 1:04d}.jpg'), output_convolved_image[..., 1, 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        #plt.imsave(os.path.join(base_path, f'output_convolved_image_1_{epoch + 1:04d}.jpg'), output_convolved_image[..., 1, 1], cmap='gray', vmin=0, vmax=1, origin='lower')
        plt.imsave(os.path.join(base_path, f'ref_image_0_{epoch + 1:04d}.jpg'), ref_image[..., 0], cmap='gray', vmin=0, vmax=1, origin='lower')
        #plt.imsave(os.path.join(base_path, f'ref_image_1_{epoch + 1:04d}.jpg'), ref_image[..., 1], cmap='gray', vmin=0, vmax=1, origin='lower')

        # plot PSF
        psf = model.get_psf().cpu().numpy()
        fig, axs = plt.subplots(1, 5, figsize=(20, 20))
        for i, ax in enumerate(np.ravel(axs)):
            im = ax.imshow(psf[:, :, i], origin='lower')
            ax.set_axis_off()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'psf_{epoch + 1:04d}.jpg'), dpi=300)
        plt.close(fig)

        torch.save({'model': model}, model_path)
