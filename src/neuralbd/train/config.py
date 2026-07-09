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


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value, name):
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _as_size_pair(value, name):
    if isinstance(value, int):
        value = [value, value]
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return [int(value[0]), int(value[1])]


def _as_range_pair(value, name):
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    start, stop = float(value[0]), float(value[1])
    if start >= stop:
        raise ValueError(f"{name} must be increasing")
    return [start, stop]


def default_config():
    return {
        "base_dir": "runs/neuralbd",
        "work_dir": None,
        "method": "standard",
        "data": {
            "type": "numpy",
            "path": None,
            "array_key": None,
            "n_images": None,
            "pixel_per_ds": 1.0,
            "num_workers": 8,
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
                "type": "default",
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
            "registration": {
                "enabled": False,
                "sample_frames": True,
                "anchor": "first_frame",
                "max_pixels": None,
                "regularization": 0.0,
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
            "num_sanity_val_steps": 0,
            "validation_every_n_epochs": 10,
            "check_val_every_n_epoch": 10,
            "checkpoint_every_n_epochs": 10,
            "resume": True,
            "resume_from_checkpoint": None,
            "learning_rate": {
                "start": 1e-4,
                "end": 1e-4,
                "iterations": 100000,
            },
            "progressive": {
                "enabled": True,
                "start_psf_size": 1,
                "target_psf_size": None,
                "increase_every_n_epochs": 100,
                "sampling_points": 16384,
                "validation_sampling_points": 16384,
                "fixed_epoch_size": True,
                "epoch_iterations": 1000,
                "learning_rate_start": None,
                "learning_rate_end": None,
                "stages": None,
            },
        },
        "logging": {
            "wandb": True,
            "project": "NeuralBD",
            "name": "neuralbd",
        },
        "outputs": {
            "sample_count": 5,
            "figure_subregion_size": 512,
            "figure_subregion": None,
            "reference_path": None,
            "reference_array_key": None,
        },
    }


def normalize_config(config):
    input_training_cfg = (config or {}).get("training", {})
    input_progressive_cfg = input_training_cfg.get("progressive", {})
    cfg = _deep_update(default_config(), deepcopy(config or {}))
    if cfg["work_dir"] is None:
        cfg["work_dir"] = cfg["base_dir"]
    method = cfg["method"].lower()
    if method not in {"standard", "spatial"}:
        raise ValueError("method must be 'standard' or 'spatial'")
    cfg["method"] = method

    cfg["data"]["num_workers"] = _nonnegative_int(cfg["data"]["num_workers"], "data.num_workers")

    psf_cfg = cfg["model"]["psf"]
    if psf_cfg.get("type") == "fixed":
        psf_cfg["type"] = "default"
    psf_type = psf_cfg.get("type", "default")
    if psf_type not in {"default", "spatial"}:
        raise ValueError("model.psf.type must be 'default' or 'spatial'")
    psf_cfg["type"] = psf_type
    channel_mode = psf_cfg.get("channel_mode", "shared")
    if channel_mode not in {"shared", "per_channel"}:
        raise ValueError("model.psf.channel_mode must be 'shared' or 'per_channel'")
    psf_cfg["channel_mode"] = channel_mode

    if method == "standard":
        psf_cfg["type"] = "default"
    elif method == "spatial":
        psf_cfg["type"] = "spatial"
        psf_cfg["representation"] = "siren"

    representation = psf_cfg.get("representation", "parameters")
    if representation not in {"parameters", "siren"}:
        raise ValueError("model.psf.representation must be 'parameters' or 'siren'")
    if cfg["method"] == "spatial" and representation != "siren":
        raise ValueError("Spatial NeuralBD requires model.psf.representation='siren'")
    psf_cfg["representation"] = representation

    sampling = psf_cfg.pop("sampling", "cartesian")
    if sampling != "cartesian":
        raise ValueError("model.psf.sampling must be 'cartesian'")

    if "target_size" in psf_cfg:
        psf_cfg["size"] = psf_cfg["target_size"]
    psf_cfg.pop("target_size", None)

    psf_cfg["size"] = _as_size_pair(psf_cfg.get("size", [29, 29]), "model.psf.size")

    registration_cfg = cfg["model"]["registration"]
    registration_cfg["enabled"] = bool(registration_cfg.get("enabled", False))
    registration_cfg["sample_frames"] = bool(registration_cfg.get("sample_frames", True))
    anchor = registration_cfg.get("anchor", "first_frame")
    if anchor not in {"first_frame", "mean", "none"}:
        raise ValueError("model.registration.anchor must be 'first_frame', 'mean', or 'none'")
    registration_cfg["anchor"] = anchor
    if registration_cfg.get("max_pixels") is not None:
        max_pixels = float(registration_cfg["max_pixels"])
        if max_pixels <= 0:
            raise ValueError("model.registration.max_pixels must be positive")
        registration_cfg["max_pixels"] = max_pixels
    regularization = float(registration_cfg.get("regularization", 0.0))
    if regularization < 0:
        raise ValueError("model.registration.regularization must be non-negative")
    registration_cfg["regularization"] = regularization

    progressive_cfg = cfg["training"]["progressive"]
    if "start" in progressive_cfg:
        progressive_cfg["start_psf_size"] = progressive_cfg["start"]
    progressive_cfg.pop("start", None)
    if "sampling_points" not in input_progressive_cfg and "sampling_points_start" in input_progressive_cfg:
        progressive_cfg["sampling_points"] = progressive_cfg["sampling_points_start"]
    if "sampling_points" not in input_progressive_cfg and "training_points_start" in input_progressive_cfg:
        progressive_cfg["sampling_points"] = progressive_cfg["training_points_start"]
    if "sampling_points" not in input_progressive_cfg and "batch_size_start" in input_progressive_cfg:
        progressive_cfg["sampling_points"] = progressive_cfg["batch_size_start"]
    if "sampling_points" not in input_progressive_cfg and "batch_size" in (config or {}).get("data", {}):
        progressive_cfg["sampling_points"] = cfg["data"]["batch_size"]
    cfg["data"].pop("batch_size", None)
    progressive_cfg.pop("sampling_points_start", None)
    progressive_cfg.pop("batch_size_start", None)
    progressive_cfg.pop("training_points_start", None)
    if progressive_cfg["target_psf_size"] is None:
        progressive_cfg["target_psf_size"] = psf_cfg["size"][0]
    progressive_cfg["start_psf_size"] = _positive_int(
        progressive_cfg["start_psf_size"], "training.progressive.start_psf_size"
    )
    progressive_cfg["target_psf_size"] = _positive_int(
        progressive_cfg["target_psf_size"], "training.progressive.target_psf_size"
    )
    progressive_cfg["increase_every_n_epochs"] = _positive_int(
        progressive_cfg["increase_every_n_epochs"], "training.progressive.increase_every_n_epochs"
    )
    progressive_cfg["sampling_points"] = _positive_int(
        progressive_cfg["sampling_points"], "training.progressive.sampling_points"
    )
    if progressive_cfg["epoch_iterations"] is not None:
        progressive_cfg["epoch_iterations"] = _positive_int(
            progressive_cfg["epoch_iterations"], "training.progressive.epoch_iterations"
        )
    if progressive_cfg.get("validation_sampling_points") is not None:
        progressive_cfg["validation_sampling_points"] = _positive_int(
            progressive_cfg["validation_sampling_points"], "training.progressive.validation_sampling_points"
        )
    if progressive_cfg.get("stages"):
        for idx, stage in enumerate(progressive_cfg["stages"]):
            stage_name = f"training.progressive.stages[{idx}]"
            stage["epochs"] = _positive_int(stage["epochs"], f"{stage_name}.epochs")
            stage["psf_size"] = _as_size_pair(stage["psf_size"], f"{stage_name}.psf_size")
            if "training_points" not in stage and "batch_size" in stage:
                stage["training_points"] = stage["batch_size"]
            stage.pop("batch_size", None)
            if stage.get("training_points") is not None:
                stage["training_points"] = _positive_int(
                    stage["training_points"], f"{stage_name}.training_points"
                )
    if progressive_cfg.get("enabled", False):
        target_size = int(progressive_cfg["target_psf_size"])
        if psf_cfg["size"][0] < target_size or psf_cfg["size"][1] < target_size:
            psf_cfg["size"] = [target_size, target_size]

    if cfg["pretraining"]["target"] not in {"first_frame", "frame", "mean"}:
        raise ValueError("pretraining.target must be 'first_frame', 'frame', or 'mean'")

    training_cfg = cfg["training"]
    if "validation_every_n_epochs" in input_training_cfg:
        validation_every_n_epochs = training_cfg["validation_every_n_epochs"]
    elif "check_val_every_n_epoch" in input_training_cfg:
        validation_every_n_epochs = training_cfg["check_val_every_n_epoch"]
    else:
        validation_every_n_epochs = training_cfg["validation_every_n_epochs"]
    validation_every_n_epochs = int(validation_every_n_epochs)
    if validation_every_n_epochs <= 0:
        raise ValueError("training.validation_every_n_epochs must be positive")
    training_cfg["validation_every_n_epochs"] = validation_every_n_epochs
    training_cfg["check_val_every_n_epoch"] = validation_every_n_epochs

    if cfg["outputs"]["figure_subregion_size"] is not None:
        cfg["outputs"]["figure_subregion_size"] = _positive_int(
            cfg["outputs"]["figure_subregion_size"], "outputs.figure_subregion_size"
        )
    figure_subregion = cfg["outputs"].get("figure_subregion")
    if figure_subregion is not None:
        x_range = figure_subregion.get("x_range", figure_subregion.get("x"))
        y_range = figure_subregion.get("y_range", figure_subregion.get("y"))
        if x_range is None and y_range is None:
            raise ValueError("outputs.figure_subregion requires x/y ranges")
        cfg["outputs"]["figure_subregion"] = {
            "x": _as_range_pair(x_range, "outputs.figure_subregion.x") if x_range is not None else None,
            "y": _as_range_pair(y_range, "outputs.figure_subregion.y") if y_range is not None else None,
        }

    return cfg


def load_config(path):
    with open(path, "r") as stream:
        return normalize_config(yaml.safe_load(stream))
