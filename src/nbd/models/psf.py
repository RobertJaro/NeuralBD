import torch
import torch.nn.functional as F
from torch import nn

from nbd.models.siren import SirenModel


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

    def resize(self, psf_size):
        psf_size = tuple(psf_size)
        with torch.no_grad():
            psf = torch.exp(self.log_psfs).detach()
            if psf.ndim == 3:
                resized = F.interpolate(
                    psf.permute(2, 0, 1)[:, None, ...],
                    size=psf_size,
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].permute(1, 2, 0)
            else:
                n_x, n_y, n_frames, n_channels = psf.shape
                resized = F.interpolate(
                    psf.permute(2, 3, 0, 1).reshape(n_frames * n_channels, 1, n_x, n_y),
                    size=psf_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(n_frames, n_channels, *psf_size).permute(2, 3, 0, 1)
            resized = normalize_psf(resized).clamp_min(1e-8).log()
        self.log_psfs = nn.Parameter(resized.to(device=self.log_psfs.device, dtype=self.log_psfs.dtype))

    def forward(self, coords=None, psf_coords=None, area_elements=None):
        psf = torch.exp(self.log_psfs)
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


class SirenPSFModel(nn.Module):
    supports_sample_permutation = True

    def __init__(self, n_frames, n_channels=1, channel_mode="shared", dim=128, n_layers=4, w0=1.0, w0_init=5.0):
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


class SpatialPSFModel(SirenPSFModel):
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
