import torch
from torch import nn

from neuralbd.models.siren import SirenModel


def gaussian_psf(psf_size, sigma=5.0, n_frames=1, n_channels=None):
    x = torch.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_size[0], dtype=torch.float32)
    y = torch.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_size[1], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    psf = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma**2))
    psf = psf[..., None].repeat(1, 1, n_frames)
    if n_channels is not None:
        psf = psf[..., None].repeat(1, 1, 1, n_channels)
    return psf


def normalize_psf(psf, area_elements=None, eps=1e-8):
    if area_elements is None:
        spatial_dims = (0, 1)
        norm = psf.sum(dim=spatial_dims, keepdim=True)
    else:
        if area_elements.ndim == 3:
            is_batched = psf.ndim >= 4 and tuple(psf.shape[1:3]) == tuple(area_elements.shape[1:3])
            if not is_batched:
                psf = psf[None, ...]
            spatial_dims = (1, 2)
        else:
            spatial_dims = (0, 1)
        while area_elements.ndim < psf.ndim:
            area_elements = area_elements.unsqueeze(-1)
        norm = (psf * area_elements).sum(dim=spatial_dims, keepdim=True)
    return psf / (norm + eps)


class FixedPSFModel(nn.Module):
    def __init__(self, n_frames, psf_size=(29, 29), sigma=5.0, n_channels=1, channel_mode="shared"):
        super().__init__()
        if channel_mode not in {"shared", "per_channel"}:
            raise ValueError("channel_mode must be 'shared' or 'per_channel'")
        self.channel_mode = channel_mode
        self.n_channels = n_channels
        initial_psf = gaussian_psf(
            psf_size,
            sigma=sigma,
            n_frames=n_frames,
            n_channels=n_channels if channel_mode == "per_channel" else None,
        )
        self.log_psfs = nn.Parameter(initial_psf.clamp_min(1e-8).log())
        self.psf_size = tuple(int(size) for size in psf_size)
        self.active_psf_size = self.psf_size

    def resize(self, psf_size):
        psf_size = tuple(int(size) for size in psf_size)
        if psf_size[0] > self.psf_size[0] or psf_size[1] > self.psf_size[1]:
            raise ValueError("Active PSF size cannot exceed the learned parameter support")
        self.active_psf_size = psf_size

    def _active_log_psfs(self):
        if self.active_psf_size == self.psf_size:
            return self.log_psfs
        x0 = (self.psf_size[0] - self.active_psf_size[0]) // 2
        y0 = (self.psf_size[1] - self.active_psf_size[1]) // 2
        return self.log_psfs[x0:x0 + self.active_psf_size[0], y0:y0 + self.active_psf_size[1], ...]

    def forward(self, coords=None, psf_coords=None, area_elements=None):
        psf = torch.exp(self._active_log_psfs())
        psf = normalize_psf(psf, area_elements)
        if coords is None:
            return psf
        if psf.ndim == 5:
            return psf
        if psf.ndim == 4:
            if self.channel_mode == "shared":
                return psf
            return psf[None, ...].expand(coords.shape[0], -1, -1, -1, -1)
        return psf[None, ...].expand(coords.shape[0], -1, -1, -1)


class ContinuousPSFModel(nn.Module):
    supports_sample_permutation = True

    def __init__(self, n_frames, n_channels=1, channel_mode="shared", dim=128, n_layers=4, w0=1.0, w0_init=30.0):
        super().__init__()
        if channel_mode not in {"shared", "per_channel"}:
            raise ValueError("channel_mode must be 'shared' or 'per_channel'")
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.channel_mode = channel_mode
        self.n_outputs = n_frames * n_channels if channel_mode == "per_channel" else n_frames
        self.model = SirenModel(
            in_dim=2,
            out_dim=self.n_outputs,
            dim=dim,
            n_layers=n_layers,
            w0=w0,
            w0_init=w0_init,
        )

    def forward(self, coords=None, psf_coords=None, area_elements=None):
        if psf_coords is None:
            raise ValueError("psf_coords are required for SIREN PSF evaluation")
        if psf_coords.ndim == 3:
            batch_size = 1 if coords is None else coords.shape[0]
            psf_coords = psf_coords[None, ...].expand(batch_size, -1, -1, -1)
        n_x, n_y = psf_coords.shape[1:3]
        log_psf = self.model(psf_coords.reshape(-1, 2))
        if self.channel_mode == "per_channel":
            log_psf = log_psf.reshape(psf_coords.shape[0], n_x, n_y, self.n_frames, self.n_channels)
        else:
            log_psf = log_psf.reshape(psf_coords.shape[0], n_x, n_y, self.n_frames)
        psf = torch.exp(log_psf)
        return normalize_psf(psf, area_elements)


class SpatialPSFModel(ContinuousPSFModel):
    supports_sample_permutation = True

    def __init__(self, n_frames, n_channels=1, channel_mode="shared", dim=128, n_layers=4, w0=1.0, w0_init=5.0):
        nn.Module.__init__(self)
        if channel_mode not in {"shared", "per_channel"}:
            raise ValueError("channel_mode must be 'shared' or 'per_channel'")
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.channel_mode = channel_mode
        self.n_outputs = n_frames * n_channels if channel_mode == "per_channel" else n_frames
        self.model = SirenModel(
            in_dim=4,
            out_dim=self.n_outputs,
            dim=dim,
            n_layers=n_layers,
            w0=w0,
            w0_init=w0_init,
        )

    def forward(self, coords, psf_coords, area_elements=None):
        if psf_coords.ndim == 3:
            psf_coords = psf_coords[None, ...].expand(coords.shape[0], -1, -1, -1)
        if psf_coords.shape[0] != coords.shape[0]:
            psf_coords = psf_coords.expand(coords.shape[0], -1, -1, -1)
        n_x, n_y = psf_coords.shape[1:3]
        coords_expanded = coords[:, None, None, :].expand(-1, n_x, n_y, -1)
        model_coords = torch.cat([coords_expanded, psf_coords], dim=-1)
        log_psf = self.model(model_coords.reshape(-1, 4))
        if self.channel_mode == "per_channel":
            log_psf = log_psf.reshape(coords.shape[0], n_x, n_y, self.n_frames, self.n_channels)
        else:
            log_psf = log_psf.reshape(coords.shape[0], n_x, n_y, self.n_frames)
        psf = torch.exp(log_psf)
        return normalize_psf(psf, area_elements)
