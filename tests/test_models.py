import torch

from neuralbd.models import FixedPSFModel, ImageSirenModel, NeuralBDConvolution, SirenPSFModel, SpatialPSFModel, build_psf_grid


def test_image_siren_forward_shape():
    model = ImageSirenModel(n_channels=2, dim=16, n_layers=2)
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


def test_siren_psf_normalization():
    coords = torch.rand(4, 2)
    psf_coords, area = build_psf_grid((5, 5), pixel_per_ds=1.0)
    psf_coords = psf_coords[None, ...].expand(coords.shape[0], -1, -1, -1)
    area = area[None, ...].expand(coords.shape[0], -1, -1)
    model = SirenPSFModel(n_frames=2, n_channels=2, channel_mode="per_channel", dim=16, n_layers=2)
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
    image_model = ImageSirenModel(n_channels=3, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=3, psf_size=(5, 5))
    convolution = NeuralBDConvolution(image_model, psf_model, psf_size=(5, 5), pixel_per_ds=32.0)
    out = convolution(torch.rand(7, 2))
    assert out.shape == (7, 3, 3)


def test_convolution_shape_per_channel_psf():
    image_model = ImageSirenModel(n_channels=2, dim=16, n_layers=2)
    psf_model = FixedPSFModel(n_frames=3, n_channels=2, channel_mode="per_channel", psf_size=(5, 5))
    convolution = NeuralBDConvolution(image_model, psf_model, psf_size=(5, 5), pixel_per_ds=32.0)
    out = convolution(torch.rand(7, 2))
    assert out.shape == (7, 3, 2)
