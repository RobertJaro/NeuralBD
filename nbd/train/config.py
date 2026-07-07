from copy import deepcopy
from pathlib import Path

import yaml


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def default_config():
    return {
        "base_dir": "runs/neuralbd",
        "method": "standard",
        "data": {
            "type": "numpy",
            "path": None,
            "array_key": None,
            "n_images": None,
            "pixel_per_ds": 1.0,
            "batch_size": 2048,
            "num_workers": 0,
            "normalization": "minmax",
            "duplicate_channels": False,
            "channels": None,
            "frame_indices": None,
            "frame_slice": None,
            "subframe": None,
        },
        "model": {
            "image": {
                "type": "siren",
                "n_channels": None,
                "dim": 256,
                "n_layers": 8,
                "w0": 1.0,
                "w0_init": 5.0,
                "output_activation": "softplus",
            },
            "psf": {
                "type": "fixed",
                "representation": "parameters",
                "size": [29, 29],
                "sigma": 5.0,
                "dim": 128,
                "n_layers": 4,
                "w0": 1.0,
                "w0_init": 5.0,
                "jitter": False,
                "permute_samples": False,
                "channel_mode": "shared",
            },
        },
        "pretraining": {
            "enabled": False,
            "epochs": 0,
            "learning_rate": 1e-4,
            "target": "first_frame",
            "frame_index": 0,
        },
        "training": {
            "epochs": 1000,
            "accelerator": "auto",
            "devices": "auto",
            "precision": "32-true",
            "log_every_n_steps": 50,
            "check_val_every_n_epoch": 10,
            "checkpoint_every_n_epochs": 10,
            "learning_rate": {
                "start": 1e-4,
                "end": 1e-4,
                "iterations": 100000,
            },
            "progressive": {
                "enabled": False,
                "start_psf_size": 3,
                "target_psf_size": None,
                "n_stages": 5,
                "batch_size_start": None,
                "batch_size_end": None,
                "learning_rate_start": None,
                "learning_rate_end": None,
                "stages": None,
            },
        },
        "logging": {
            "wandb": False,
            "project": "NeuralBD",
            "name": "neuralbd",
        },
        "outputs": {
            "directory": None,
            "save_validation_arrays": True,
            "save_validation_figures": True,
            "sample_count": 5,
            "reference_path": None,
            "reference_array_key": None,
        },
    }


def normalize_config(config):
    cfg = _deep_update(default_config(), deepcopy(config or {}))
    method = cfg["method"].lower()
    if method not in {"standard", "spatial"}:
        raise ValueError("method must be 'standard' or 'spatial'")
    cfg["method"] = method

    psf_cfg = cfg["model"]["psf"]
    channel_mode = psf_cfg.get("channel_mode", "shared")
    if channel_mode not in {"shared", "per_channel"}:
        raise ValueError("model.psf.channel_mode must be 'shared' or 'per_channel'")
    psf_cfg["channel_mode"] = channel_mode

    if method == "standard":
        psf_cfg["type"] = "fixed"
    elif method == "spatial":
        psf_cfg["type"] = "spatial"
        psf_cfg["representation"] = "siren"

    representation = psf_cfg.get("representation", "parameters")
    if representation not in {"parameters", "siren"}:
        raise ValueError("model.psf.representation must be 'parameters' or 'siren'")
    if cfg["method"] == "spatial" and representation != "siren":
        raise ValueError("Spatial NeuralBD requires model.psf.representation='siren'")
    psf_cfg["representation"] = representation

    psf_size = psf_cfg.get("size", [29, 29])
    if isinstance(psf_size, int):
        psf_size = [psf_size, psf_size]
    if len(psf_size) != 2:
        raise ValueError("model.psf.size must be an int or a pair")
    psf_cfg["size"] = [int(psf_size[0]), int(psf_size[1])]

    progressive_cfg = cfg["training"]["progressive"]
    if progressive_cfg["target_psf_size"] is None:
        progressive_cfg["target_psf_size"] = psf_cfg["size"][0]

    if cfg["pretraining"]["target"] not in {"first_frame", "frame", "mean"}:
        raise ValueError("pretraining.target must be 'first_frame', 'frame', or 'mean'")

    if cfg["outputs"]["directory"] is None:
        cfg["outputs"]["directory"] = str(Path(cfg["base_dir"]) / "outputs")

    return cfg


def load_config(path):
    with open(path, "r") as stream:
        return normalize_config(yaml.safe_load(stream))
