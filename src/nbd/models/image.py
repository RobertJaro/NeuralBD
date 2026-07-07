import torch
from torch import nn

from nbd.models.siren import SirenModel


class ImageSirenModel(nn.Module):
    def __init__(
        self,
        n_channels=1,
        dim=256,
        n_layers=8,
        w0=1.0,
        w0_init=5.0,
        output_activation="softplus",
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.model = SirenModel(
            in_dim=2,
            out_dim=n_channels,
            dim=dim,
            n_layers=n_layers,
            w0=w0,
            w0_init=w0_init,
        )
        self.output_activation = output_activation
        self.eps = eps

    def forward(self, coords):
        x = self.model(coords)
        if self.output_activation == "identity":
            return x
        if self.output_activation == "softplus":
            return torch.nn.functional.softplus(x) + self.eps
        if self.output_activation == "exp":
            return torch.exp(x)
        if self.output_activation == "power10":
            return torch.pow(10.0, x)
        raise ValueError(f"Unknown image output activation: {self.output_activation}")
