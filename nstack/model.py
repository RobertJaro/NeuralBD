import numpy as np
import torch
from torch import nn
from lightning import LightningModule


class PositionalEncoding(LightningModule):
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
        #freqs = 2 ** torch.FloatTensor(num_freqs).uniform_(0, max_freq)
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


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


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


class PSFStackModel(LightningModule):

    def __init__(self, n_images, dim, modes, psf_size, kl_basis, im_scaling, images, simulation, ref_psfs):
        # images = (w, h, n, c)
        super().__init__()
        self.n_images = n_images
        self.modes = modes
        self.psf_size = psf_size
        self.kl_basis = kl_basis
        self.im_scaling = im_scaling
        self.images = images
        self.high_quality = simulation
        self.ref_psfs = ref_psfs
        self.wavefront_coefficient = nn.Parameter(torch.randn(n_images, self.modes, dtype=torch.float32),
                                                  requires_grad=True)
        self.image_model = ImageModel(dim)
        self.sigma = nn.Parameter(torch.ones(n_images, dtype=torch.float32) * 0.5, requires_grad=True)
        self.mean = nn.Parameter(torch.zeros(n_images, 2, dtype=torch.float32), requires_grad=True)

        x_values_psf = torch.linspace(-1., 1., self.psf_size, dtype=torch.float32)
        y_values_psf = torch.linspace(-1., 1., self.psf_size, dtype=torch.float32)
        psf_x, psf_y = torch.meshgrid(x_values_psf, y_values_psf, indexing='ij')
        self.register_buffer('psf_x', psf_x)
        self.register_buffer('psf_y', psf_y)

        self.softplus = nn.Softplus()

        x_values = torch.linspace(-(self.psf_size // 2), self.psf_size // 2, self.psf_size, dtype=torch.float32)
        y_values = torch.linspace(-(self.psf_size // 2), self.psf_size // 2, self.psf_size, dtype=torch.float32)
        x, y = torch.meshgrid(x_values, y_values, indexing='ij')
        grid_sampling = torch.stack([x, y], -1).reshape(-1, 2)  # psf_coords, xy
        grid_sampling = grid_sampling / self.im_scaling
        self.grid_sampling = nn.Parameter(grid_sampling, requires_grad=False)

    def get_transformed_images(self, coords):
        transformed_coords = self.transform_coords(coords)
        image_stack = torch.stack([model(transformed_coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def get_images(self, coords):
        # coords: batch, xy
        grid_sampling_coords = coords[:, None, :] + self.grid_sampling[None, :, :]  # --> batch, psf_coords, xy
        #
        psf = self.get_psf().to(coords.device)
        flat_psf = psf.reshape(-1, self.n_images)

        sampling_image = self.image_model(grid_sampling_coords.reshape(-1, 2))
        sampling_image = sampling_image.reshape(coords.shape[0], -1, sampling_image.shape[-1])  # batch, psf_coords, channels
        # sampling_image = sampling_image * condition # set values outside of image to 0
        convolved_images = torch.einsum('...sc,sn->...nc', sampling_image, flat_psf)
        image = self.image_model(coords)
        return image, convolved_images, psf, self.images, self.high_quality, self.ref_psfs

    def get_psf(self):
        # Karhun-Loeve Base
        wavefront_coefficient = self.wavefront_coefficient
        wavefront_coefficient = torch.tanh(wavefront_coefficient)
        wavefront = torch.einsum('kij,lk->lij', self.kl_basis.to(wavefront_coefficient.device), wavefront_coefficient)
        psfs = torch.stack([self.PSF(torch.exp(1j * wavefront[i, :, :])) for i in range(self.n_images)], -1)
        #psfs = psfs[60:69, 60:69, :] #crop to 9x9

        norm = psfs.sum(dim=(0, 1), keepdims=True)
        return psfs / norm

    def PSF(self, complx_pupil):
        PSF = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(complx_pupil)))
        PSF = (torch.abs(PSF))**2 #or PSF*PSF.conjugate()
        return PSF

    def forward(self, coords):
        return self.get_images(coords)