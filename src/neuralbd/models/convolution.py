import torch
from torch import nn


def _as_psf_size(psf_size):
    if isinstance(psf_size, int):
        return (int(psf_size), int(psf_size))
    return tuple(int(size) for size in psf_size)


def build_psf_grid(psf_size=(29, 29), pixel_per_ds=1.0, device=None, dtype=torch.float32):
    psf_size = _as_psf_size(psf_size)
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


def _infer_n_frames(psf_model):
    if hasattr(psf_model, "n_frames"):
        return int(psf_model.n_frames)
    if hasattr(psf_model, "log_psfs"):
        return int(psf_model.log_psfs.shape[2])
    return 1


class NeuralBDConvolution(nn.Module):
    def __init__(
        self,
        image_model,
        psf_model,
        psf_size=(29, 29),
        pixel_per_ds=1.0,
        jitter=False,
        permute_psf_samples=False,
        frame_shift_enabled=False,
        n_frames=None,
        max_frame_shift_pixels=None,
        frame_shift_anchor="first_frame",
    ):
        super().__init__()
        object.__setattr__(self, "image_model", image_model)
        object.__setattr__(self, "psf_model", psf_model)
        self.psf_size = _as_psf_size(psf_size)
        self.pixel_per_ds = pixel_per_ds
        self.jitter = jitter
        self.permute_psf_samples = permute_psf_samples
        self.frame_shift_enabled = bool(frame_shift_enabled)
        self.n_frames = int(n_frames or _infer_n_frames(psf_model))
        self.max_frame_shift_pixels = max_frame_shift_pixels
        self.frame_shift_anchor = frame_shift_anchor
        if self.frame_shift_enabled:
            self.raw_frame_shifts = nn.Parameter(torch.zeros(self.n_frames, 2))
        else:
            self.register_buffer("raw_frame_shifts", torch.zeros(self.n_frames, 2))
        psf_coords, area_elements = self._build_sampling_grid()
        self.register_buffer("psf_coords", psf_coords)
        self.register_buffer("area_elements", area_elements)

    def _build_sampling_grid(self, device=None, dtype=torch.float32):
        return build_psf_grid(
            psf_size=self.psf_size,
            pixel_per_ds=self.pixel_per_ds,
            device=device,
            dtype=dtype,
        )

    def set_psf_size(self, psf_size):
        self.psf_size = _as_psf_size(psf_size)
        psf_coords, area_elements = self._build_sampling_grid(
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
            sample_shape = area_elements.shape[1:]
            n_samples = sample_shape[0] * sample_shape[1]
            order = torch.randperm(n_samples, device=device)
            psf_coords = psf_coords.reshape(batch_size, n_samples, 2)[:, order].reshape(batch_size, *sample_shape, 2)
            area_elements = area_elements.reshape(batch_size, n_samples)[:, order].reshape(batch_size, *sample_shape)
        return psf_coords, area_elements

    def frame_shifts(self):
        shifts = self.raw_frame_shifts
        if self.max_frame_shift_pixels is not None:
            shifts = torch.tanh(shifts) * (float(self.max_frame_shift_pixels) / float(self.pixel_per_ds))
        if self.frame_shift_anchor == "first_frame":
            shifts = shifts - shifts[:1]
        elif self.frame_shift_anchor == "mean":
            shifts = shifts - shifts.mean(dim=0, keepdim=True)
        elif self.frame_shift_anchor != "none":
            raise ValueError("frame_shift_anchor must be 'first_frame', 'mean', or 'none'")
        return shifts

    def _select_frame_psf(self, psf, frame_indices):
        frame_indices = frame_indices.to(device=psf.device, dtype=torch.long)
        batch_indices = torch.arange(frame_indices.shape[0], device=psf.device)
        if psf.ndim == 3:
            return psf[:, :, frame_indices].permute(2, 0, 1)
        if psf.ndim == 4:
            return psf[batch_indices, :, :, frame_indices]
        if psf.ndim == 5:
            return psf[batch_indices, :, :, frame_indices, :]
        raise ValueError(f"Unsupported PSF shape for frame selection: {tuple(psf.shape)}")

    def _forward_sampled_frames(self, coords, frame_indices):
        batch_size = coords.shape[0]
        psf_coords, area_elements = self._sample_psf_coords(batch_size, coords.device, coords.dtype)
        psf = self.psf_model(coords=coords, psf_coords=psf_coords, area_elements=area_elements)
        frame_indices = frame_indices.to(device=coords.device, dtype=torch.long)
        shifts = self.frame_shifts().to(device=coords.device, dtype=coords.dtype)[frame_indices]
        sampling_coords = coords[:, None, None, :] + psf_coords + shifts[:, None, None, :]
        image = self.image_model(sampling_coords.reshape(-1, 2))
        image = image.reshape(batch_size, -1, image.shape[-1])
        flat_area = area_elements.reshape(batch_size, -1, 1)
        frame_psf = self._select_frame_psf(psf, frame_indices)
        if frame_psf.ndim == 4:
            flat_psf = frame_psf.reshape(batch_size, -1, frame_psf.shape[-1])
            return torch.sum(image * flat_psf * flat_area, dim=1)
        flat_psf = frame_psf.reshape(batch_size, -1, 1)
        return torch.sum(image * flat_psf * flat_area, dim=1)

    def _forward_shifted_all_frames(self, coords):
        batch_size = coords.shape[0]
        psf_coords, area_elements = self._sample_psf_coords(batch_size, coords.device, coords.dtype)
        psf = self.psf_model(coords=coords, psf_coords=psf_coords, area_elements=area_elements)
        shifts = self.frame_shifts().to(device=coords.device, dtype=coords.dtype)
        sampling_coords = (
            coords[:, None, None, None, :]
            + psf_coords[:, :, :, None, :]
            + shifts[None, None, None, :, :]
        )
        image = self.image_model(sampling_coords.reshape(-1, 2))
        image = image.reshape(batch_size, -1, self.n_frames, image.shape[-1])
        flat_area = area_elements.reshape(batch_size, -1, 1, 1)
        if psf.ndim == 5:
            flat_psf = psf.reshape(batch_size, -1, psf.shape[-2], psf.shape[-1])
            return torch.sum(image * flat_psf * flat_area, dim=1)
        if psf.ndim == 4:
            flat_psf = psf.reshape(batch_size, -1, psf.shape[-1], 1)
        elif psf.ndim == 3:
            flat_psf = psf.reshape(1, -1, psf.shape[-1], 1)
        else:
            raise ValueError(f"Unsupported PSF shape: {tuple(psf.shape)}")
        return torch.sum(image * flat_psf * flat_area, dim=1)

    def forward(self, coords, frame_indices=None):
        if frame_indices is not None:
            return self._forward_sampled_frames(coords, frame_indices)
        if self.frame_shift_enabled:
            return self._forward_shifted_all_frames(coords)
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
