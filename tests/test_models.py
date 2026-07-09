import torch
from torch import nn

from neuralbd.models import (
    FixedPSFModel,
    ImageModel,
    NeuralBDConvolution,
    ContinuousPSFModel,
    SpatialPSFModel,
    build_psf_grid,
)


def test_image_model_forward_shape():
    model = ImageModel(n_channels=2, dim=16, n_layers=2)
    coords = torch.rand(5, 2)
    out = model(coords)
    assert out.shape == (5, 2)
    assert torch.all(out > 0)


def test_fixed_psf_normalization():
    model = FixedPSFModel(n_frames=3, psf_size=(5, 5))
    psf = model()
    assert psf.shape == (5, 5, 3)
    assert torch.allclose(psf.sum(dim=(0, 1)), torch.ones(3), atol=1e-5)


def test_fixed_psf_per_channel_normalization():
    model = FixedPSFModel(n_frames=3, n_channels=2, channel_mode="per_channel", psf_size=(5, 5))
    psf = model()
    assert psf.shape == (5, 5, 3, 2)
    assert torch.allclose(psf.sum(dim=(0, 1)), torch.ones(3, 2), atol=1e-5)


def test_spatial_psf_normalization():
    coords = torch.rand(4, 2)
    psf_coords, area = build_psf_grid((5, 5), pixel_per_ds=1.0)
    psf_coords = psf_coords[None, ...].expand(coords.shape[0], -1, -1, -1)
    area = area[None, ...].expand(coords.shape[0], -1, -1)
    model = SpatialPSFModel(n_frames=2, dim=16, n_layers=2)
    psf = model(coords, psf_coords, area)
    assert psf.shape == (4, 5, 5, 2)
    assert torch.allclose((psf * area[..., None]).sum(dim=(1, 2)), torch.ones(4, 2), atol=1e-5)


def test_continuous_psf_normalization():
    coords = torch.rand(4, 2)
    psf_coords, area = build_psf_grid((5, 5), pixel_per_ds=1.0)
    psf_coords = psf_coords[None, ...].expand(coords.shape[0], -1, -1, -1)
    area = area[None, ...].expand(coords.shape[0], -1, -1)
    model = ContinuousPSFModel(n_frames=2, n_channels=2, channel_mode="per_channel", dim=16, n_layers=2)
    psf = model(coords, psf_coords, area)
    assert psf.shape == (4, 5, 5, 2, 2)
    assert torch.allclose((psf * area[..., None, None]).sum(dim=(1, 2)), torch.ones(4, 2, 2), atol=1e-5)


def test_spatial_psf_per_channel_normalization():
    coords = torch.rand(4, 2)
    psf_coords, area = build_psf_grid((5, 5), pixel_per_ds=1.0)
    psf_coords = psf_coords[None, ...].expand(coords.shape[0], -1, -1, -1)
    area = area[None, ...].expand(coords.shape[0], -1, -1)
    model = SpatialPSFModel(n_frames=2, n_channels=3, channel_mode="per_channel", dim=16, n_layers=2)
    psf = model(coords, psf_coords, area)
    assert psf.shape == (4, 5, 5, 2, 3)
    assert torch.allclose((psf * area[..., None, None]).sum(dim=(1, 2)), torch.ones(4, 2, 3), atol=1e-5)


def test_convolution_shape_shared_psf():
    image_model = ImageModel(n_channels=3, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=3, psf_size=(5, 5))
    convolution = NeuralBDConvolution(image_model, psf_model, psf_size=(5, 5), pixel_per_ds=32.0)
    out = convolution(torch.rand(7, 2))
    assert out.shape == (7, 3, 3)


def test_convolution_shape_per_channel_psf():
    image_model = ImageModel(n_channels=2, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=3, n_channels=2, channel_mode="per_channel", psf_size=(5, 5))
    convolution = NeuralBDConvolution(image_model, psf_model, psf_size=(5, 5), pixel_per_ds=32.0)
    out = convolution(torch.rand(7, 2))
    assert out.shape == (7, 3, 2)


def test_convolution_sampled_frames_match_full_frames_for_parameter_psf():
    torch.manual_seed(1)
    image_model = ImageModel(n_channels=2, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=4, n_channels=2, channel_mode="per_channel", psf_size=(5, 5))
    convolution = NeuralBDConvolution(image_model, psf_model, psf_size=(5, 5), pixel_per_ds=32.0)
    coords = torch.rand(6, 2)
    frame_indices = torch.tensor([0, 1, 2, 3, 1, 0])

    full = convolution(coords)
    sampled = convolution(coords, frame_indices=frame_indices)

    assert sampled.shape == (6, 2)
    assert torch.allclose(sampled, full[torch.arange(coords.shape[0]), frame_indices], atol=1e-6)


def test_convolution_learned_frame_shift_samples_single_frame():
    torch.manual_seed(2)
    image_model = ImageModel(n_channels=1, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=3, psf_size=(3, 3))
    convolution = NeuralBDConvolution(
        image_model,
        psf_model,
        psf_size=(3, 3),
        pixel_per_ds=8.0,
        frame_shift_enabled=True,
        n_frames=3,
        max_frame_shift_pixels=2.0,
    )
    coords = torch.rand(5, 2)
    frame_indices = torch.tensor([0, 1, 2, 1, 0])

    out = convolution(coords, frame_indices=frame_indices)

    assert out.shape == (5, 1)
    assert convolution.raw_frame_shifts.requires_grad
    assert torch.allclose(convolution.frame_shifts()[0], torch.zeros(2))


def test_continuous_psf_permutation_uses_same_jittered_coords_and_areas(monkeypatch):
    class RecordingImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.coords = None

        def forward(self, coords):
            self.coords = coords.detach().clone()
            return torch.ones(coords.shape[0], 1, device=coords.device, dtype=coords.dtype)

    class RecordingContinuousPSF(nn.Module):
        supports_sample_permutation = True
        n_frames = 1

        def __init__(self):
            super().__init__()
            self.psf_coords = None
            self.area_elements = None

        def forward(self, coords=None, psf_coords=None, area_elements=None):
            self.psf_coords = psf_coords.detach().clone()
            self.area_elements = area_elements.detach().clone()
            return torch.ones(*psf_coords.shape[:3], 1, device=psf_coords.device, dtype=psf_coords.dtype)

    order = torch.tensor([3, 1, 0, 2])
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: order.to(device=device))

    image_model = RecordingImageModel()
    psf_model = RecordingContinuousPSF()
    convolution = NeuralBDConvolution(
        image_model,
        psf_model,
        psf_size=(2, 2),
        pixel_per_ds=1.0,
        jitter=True,
        permute_psf_samples=True,
    )
    convolution.area_elements = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    coords = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    convolution(coords)

    image_offsets = image_model.coords.reshape(coords.shape[0], 2, 2, 2) - coords[:, None, None, :]
    expected_area = convolution.area_elements.reshape(-1)[order].reshape(2, 2)

    assert torch.allclose(image_offsets, psf_model.psf_coords, atol=1e-5)
    assert torch.allclose(psf_model.area_elements, expected_area[None, ...].expand(coords.shape[0], -1, -1))
