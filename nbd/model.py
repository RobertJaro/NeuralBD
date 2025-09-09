import numpy as np
import torch
from torch import nn


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

    def __init__(self, d_input, num_freqs=128, scale=2.0 ** 3):
        super().__init__()
        frequencies = torch.randn(num_freqs, d_input) * scale
        self.frequencies = nn.Parameter(2 * torch.pi * frequencies, requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = torch.einsum('...j,ij->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded

class SymmetricBoundaryEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_val = 0.0
        self.max_val = 128 / 511.5

    def forward(self, coords):
        # Apply sinus symmetric boundary mapping
        # Normalize input to [0, π] before applying sinusoidal transformation
        norm_coords = (coords - self.min_val) / (self.max_val - self.min_val)
        # x = torch.cos(norm_coords[..., 0] * np.pi)
        # y = torch.cos(norm_coords[..., 1] * np.pi)
        # coords = torch.stack([x, y], dim=-1)
        norm_coords = norm_coords * np.pi
        coords = torch.cos(norm_coords)
        return coords


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class ImageModel(nn.Module):

    def __init__(self, dim=512, n_channels=2, posencoding=True, posenc_scale=2.0 ** 2):
        super().__init__()
        self.symm_encoding = SymmetricBoundaryEncoding()

        if posencoding:
            posenc = GaussianPositionalEncoding(2, scale=posenc_scale)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            self.d_in = nn.Linear(n_channels, dim)

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_channels)
        self.activation = Sine()

    def forward(self, coords):
        #x = self.symm_encoding(coords)
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        x = 10 ** x
        return x

class PSFModel(nn.Module):

    def __init__(self, psf_shape, dim=64, n_layers=4):
        super().__init__()
        self.psf_shape = psf_shape
        self.dim = dim

        self.d_in = nn.Linear(2, dim)
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, np.prod(psf_shape))
        self.activation = Sine()

    def forward(self, coords):
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        x = x.reshape(-1, *self.psf_shape)
        return x