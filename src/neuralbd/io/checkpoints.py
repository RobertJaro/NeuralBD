import torch


def save_model_state(path, module, config=None):
    state = {
        "format_version": "0.2",
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
        },
    }
    torch.save(state, path)
