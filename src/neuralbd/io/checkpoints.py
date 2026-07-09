import torch


def save_model_state(path, module, config=None):
    psf_size = tuple(int(size) for size in module.convolution.psf_size)
    learned_psf_size = tuple(int(size) for size in getattr(module.psf_model, "psf_size", psf_size))
    active_psf_size = tuple(int(size) for size in getattr(module.psf_model, "active_psf_size", psf_size))
    state = {
        "format_version": "0.3",
        "method": module.method,
        "model_state_dict": {
            "image": module.image_model.state_dict(),
            "psf": module.psf_model.state_dict(),
        },
        "config": config,
        "metadata": {
            "images_shape": tuple(module.images_shape),
            "image_shape": tuple(module.images_shape[:2]),
            "n_frames": module.n_frames,
            "n_channels": module.n_channels,
            "pixel_per_ds": module.pixel_per_ds,
            "coordinate_layout": "height,width",
            "coordinate_order": "x,y",
            "coordinate_origin": "center",
            "psf_size": psf_size,
            "active_psf_size": active_psf_size,
            "learned_psf_size": learned_psf_size,
            "jitter": bool(module.convolution.jitter),
            "permute_psf_samples": bool(module.convolution.permute_psf_samples),
        },
    }
    torch.save(state, path)
