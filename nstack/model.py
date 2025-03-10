import numpy as np
import torch
from torch import nn
from lightning import LightningModule


class PositionalEncoding(nn.Module):

    def __init__(self, in_features, num_freqs=128, min_freq=-2, max_freq=8):
        super().__init__()
        frequencies = 2 ** torch.linspace(min_freq, max_freq, num_freqs) * torch.pi
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.d_output = in_features * (1 + num_freqs * 2)

    def forward(self, x):
        encoded = torch.einsum('...i,j->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded), x], -1)
        return encoded

class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=128, scale=2.0 ** 1):
        super().__init__()
        frequencies = torch.randn(num_freqs, d_input) * scale
        self.frequencies = nn.Parameter(2 * torch.pi * frequencies, requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = torch.einsum('...j,ij->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class ImageModel(nn.Module):

    def __init__(self, dim, n_channels=2, posencoding=False):
        super().__init__()
        if posencoding:
            posenc = GaussianPositionalEncoding(2, scale=2.0 ** 5)
            #posenc = PositionalEncoding(2, min_freq=0, max_freq=8)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(n_channels, dim)

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
        #x = self.out_activation(x)
        x = 10 ** x
        return x


class PSFStackModel(nn.Module):

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
        #self.wavefront_coefficient = nn.Parameter(torch.randn(n_images, self.modes, dtype=torch.float32),
        #                                          requires_grad=True)
        self.find_psfs = nn.Parameter(torch.randn(self.psf_size, self.psf_size, n_images, dtype=torch.float32),
                                    requires_grad=True)
        self.intensity_scaling = nn.Parameter(torch.ones(n_images, 2, dtype=torch.float32), requires_grad=True)
        self.image_model = ImageModel(dim, posencoding=True)
        #self.sigma = nn.Parameter(torch.ones(n_images, dtype=torch.float32) * 0.5, requires_grad=True)
        #self.mean = nn.Parameter(torch.zeros(n_images, 2, dtype=torch.float32), requires_grad=True)

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
        convolved_images = torch.einsum('...sc,sn->...nc', sampling_image, flat_psf)
        convolved_images = convolved_images * self.intensity_scaling[None, :, :]
        image = self.image_model(coords)
        return image, convolved_images, psf, self.images, self.high_quality, self.ref_psfs, self.intensity_scaling

    def get_psf(self):
        # Karhun-Loeve Base
        #wavefront_coefficient = self.wavefront_coefficient
        #wavefront_coefficient = 2 * torch.tanh(wavefront_coefficient)
        #wavefront = torch.einsum('kij,lk->lij', self.kl_basis.to(wavefront_coefficient.device), wavefront_coefficient)
        #psfs = torch.stack([self.PSF(torch.exp(1j * wavefront[i, :, :])) for i in range(self.n_images)], -1)

        # find random psfs
        psfs = self.find_psfs
        #psfs = self.softplus(psfs)
        psfs = torch.exp(psfs)

        norm = psfs.sum(dim=(0, 1), keepdims=True)
        return psfs / (norm + 1e-8)

    def PSF(self, complx_pupil):
        PSF = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(complx_pupil)))
        PSF = (torch.abs(PSF))**2 #or PSF*PSF.conjugate()
        return PSF

    def forward(self, coords):
        return self.get_images(coords)