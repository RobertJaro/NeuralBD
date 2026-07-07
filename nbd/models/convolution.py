import torch
from torch import nn


def build_psf_grid(psf_size=(29, 29), pixel_per_ds=1.0, device=None, dtype=torch.float32):
    x = torch.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_size[0], device=device, dtype=dtype)
    y = torch.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_size[1], device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1) / pixel_per_ds
    if psf_size[0] > 1 and psf_size[1] > 1:
        dx = torch.abs(coords[1, 0, 0] - coords[0, 0, 0])
        dy = torch.abs(coords[0, 1, 1] - coords[0, 0, 1])
        area = torch.ones(psf_size, device=device, dtype=dtype) * dx * dy
    else:
        area = torch.ones(psf_size, device=device, dtype=dtype)
    return coords, area


class NeuralBDConvolution(nn.Module):
    def __init__(
        self,
        image_model,
        psf_model,
        psf_size=(29, 29),
        pixel_per_ds=1.0,
        jitter=False,
        permute_psf_samples=False,
    ):
        super().__init__()
        self.image_model = image_model
        self.psf_model = psf_model
        self.psf_size = tuple(psf_size)
        self.pixel_per_ds = pixel_per_ds
        self.jitter = jitter
        self.permute_psf_samples = permute_psf_samples
        psf_coords, area_elements = build_psf_grid(psf_size=self.psf_size, pixel_per_ds=pixel_per_ds)
        self.register_buffer("psf_coords", psf_coords)
        self.register_buffer("area_elements", area_elements)

    def set_psf_size(self, psf_size):
        self.psf_size = tuple(psf_size)
        psf_coords, area_elements = build_psf_grid(
            psf_size=self.psf_size,
            pixel_per_ds=self.pixel_per_ds,
            device=self.psf_coords.device,
            dtype=self.psf_coords.dtype,
        )
        self.psf_coords = psf_coords
        self.area_elements = area_elements

    def _sample_psf_coords(self, batch_size, device, dtype):
        psf_coords = self.psf_coords.to(device=device, dtype=dtype)[None, ...].expand(batch_size, -1, -1, -1)
        area_elements = self.area_elements.to(device=device, dtype=dtype)[None, ...].expand(batch_size, -1, -1)
        if self.jitter and self.psf_size[0] > 1 and self.psf_size[1] > 1:
            dx = torch.abs(psf_coords[0, 1, 0, 0] - psf_coords[0, 0, 0, 0]) / 2
            dy = torch.abs(psf_coords[0, 0, 1, 1] - psf_coords[0, 0, 0, 1]) / 2
            noise = torch.rand_like(psf_coords) * 2 - 1
            noise[..., 0] *= dx
            noise[..., 1] *= dy
            psf_coords = psf_coords + noise
        if self.permute_psf_samples and getattr(self.psf_model, "supports_sample_permutation", False):
            n_samples = self.psf_size[0] * self.psf_size[1]
            order = torch.randperm(n_samples, device=device)
            psf_coords = psf_coords.reshape(batch_size, n_samples, 2)[:, order].reshape(batch_size, *self.psf_size, 2)
            area_elements = area_elements.reshape(batch_size, n_samples)[:, order].reshape(batch_size, *self.psf_size)
        return psf_coords, area_elements

    def forward(self, coords):
        batch_size = coords.shape[0]
        psf_coords, area_elements = self._sample_psf_coords(batch_size, coords.device, coords.dtype)
        psf = self.psf_model(coords=coords, psf_coords=psf_coords, area_elements=area_elements)
        sampling_coords = coords[:, None, None, :] + psf_coords
        image = self.image_model(sampling_coords.reshape(-1, 2))
        image = image.reshape(batch_size, -1, image.shape[-1])
        flat_area = area_elements.reshape(batch_size, -1, 1)
        if psf.ndim == 5:
            flat_psf = psf.reshape(batch_size, -1, psf.shape[-2], psf.shape[-1])
            return torch.einsum("bsc,bsnc->bnc", image, flat_psf * flat_area[..., None])
        flat_psf = psf.reshape(batch_size, -1, psf.shape[-1])
        return torch.einsum("bsc,bsn->bnc", image, flat_psf * flat_area)
