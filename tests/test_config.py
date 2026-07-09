from neuralbd.train import normalize_config
from neuralbd.train.config import load_config


def test_normalize_config_sets_spatial_psf():
    cfg = normalize_config({"method": "spatial", "model": {"psf": {"size": 9}}})
    assert cfg["method"] == "spatial"
    assert cfg["model"]["psf"]["type"] == "spatial"
    assert cfg["model"]["psf"]["size"] == [9, 9]


def test_normalize_config_sets_default_psf():
    cfg = normalize_config({"method": "standard"})
    assert cfg["model"]["psf"]["type"] == "default"
    assert cfg["model"]["registration"]["enabled"] is False


def test_normalize_config_accepts_legacy_fixed_psf_type():
    cfg = normalize_config({"model": {"psf": {"type": "fixed"}}})
    assert cfg["model"]["psf"]["type"] == "default"


def test_normalize_config_rejects_unknown_psf_type():
    try:
        normalize_config({"model": {"psf": {"type": "other"}}})
    except ValueError as exc:
        assert "model.psf.type" in str(exc)
    else:
        raise AssertionError("Expected unknown PSF type validation to fail")


def test_normalize_config_defaults_work_dir_to_base_dir():
    cfg = normalize_config({"base_dir": "runs/test"})
    assert cfg["work_dir"] == "runs/test"


def test_normalize_config_defaults_data_num_workers_to_eight():
    cfg = normalize_config({})
    assert cfg["data"]["num_workers"] == 8


def test_normalize_config_accepts_explicit_data_num_workers():
    cfg = normalize_config({"data": {"num_workers": 2}})
    assert cfg["data"]["num_workers"] == 2


def test_normalize_config_rejects_negative_data_num_workers():
    try:
        normalize_config({"data": {"num_workers": -1}})
    except ValueError as exc:
        assert "data.num_workers" in str(exc)
    else:
        raise AssertionError("Expected data.num_workers validation to fail")


def test_normalize_config_keeps_explicit_work_dir():
    cfg = normalize_config({"base_dir": "runs/test", "work_dir": "work/test"})
    assert cfg["base_dir"] == "runs/test"
    assert cfg["work_dir"] == "work/test"


def test_normalize_config_validates_figure_subregion_size():
    cfg = normalize_config({"outputs": {"figure_subregion_size": 256}})
    assert cfg["outputs"]["figure_subregion_size"] == 256

    cfg = normalize_config({"outputs": {"figure_subregion_size": None}})
    assert cfg["outputs"]["figure_subregion_size"] is None

    try:
        normalize_config({"outputs": {"figure_subregion_size": 0}})
    except ValueError as exc:
        assert "outputs.figure_subregion_size" in str(exc)
    else:
        raise AssertionError("Expected figure subregion size validation to fail")


def test_normalize_config_validates_figure_subregion_ranges():
    cfg = normalize_config({"outputs": {"figure_subregion": {"x": [100, 200], "y_range": [300, 400]}}})

    assert cfg["outputs"]["figure_subregion"] == {"x": [100.0, 200.0], "y": [300.0, 400.0]}

    try:
        normalize_config({"outputs": {"figure_subregion": {"x": [2, 1]}}})
    except ValueError as exc:
        assert "outputs.figure_subregion.x" in str(exc)
    else:
        raise AssertionError("Expected figure subregion range validation to fail")


def test_normalize_config_accepts_per_channel_psf():
    cfg = normalize_config({"model": {"psf": {"channel_mode": "per_channel"}}})
    assert cfg["model"]["psf"]["channel_mode"] == "per_channel"


def test_normalize_config_accepts_learned_registration_shift():
    cfg = normalize_config({
        "model": {
            "registration": {
                "enabled": True,
                "anchor": "mean",
                "max_pixels": 3,
                "regularization": 1e-4,
            },
        },
    })
    assert cfg["model"]["registration"] == {
        "enabled": True,
        "sample_frames": True,
        "anchor": "mean",
        "max_pixels": 3.0,
        "regularization": 1e-4,
    }


def test_normalize_config_rejects_invalid_registration_shift():
    try:
        normalize_config({"model": {"registration": {"anchor": "middle"}}})
    except ValueError as exc:
        assert "model.registration.anchor" in str(exc)
    else:
        raise AssertionError("Expected registration anchor validation to fail")


def test_normalize_config_rejects_non_cartesian_psf_sampling():
    try:
        normalize_config({"model": {"psf": {"representation": "siren", "sampling": "circular"}}})
    except ValueError as exc:
        assert "model.psf.sampling" in str(exc)
    else:
        raise AssertionError("Expected non-cartesian PSF sampling validation to fail")


def test_normalize_config_sets_spatial_siren_psf():
    cfg = normalize_config({"method": "spatial"})
    assert cfg["model"]["psf"]["type"] == "spatial"
    assert cfg["model"]["psf"]["representation"] == "siren"


def test_normalize_config_progressive_target_defaults_to_psf_size():
    cfg = normalize_config({"model": {"psf": {"size": 65}}})
    assert cfg["training"]["progressive"]["target_psf_size"] == 65


def test_normalize_config_progressive_defaults():
    cfg = normalize_config({})
    assert cfg["training"]["progressive"]["enabled"] is True
    assert cfg["training"]["progressive"]["start_psf_size"] == 1
    assert cfg["training"]["progressive"]["increase_every_n_epochs"] == 100
    assert cfg["training"]["progressive"]["sampling_points"] == 16384
    assert cfg["training"]["progressive"]["validation_sampling_points"] == 16384
    assert cfg["training"]["progressive"]["fixed_epoch_size"] is True
    assert cfg["training"]["progressive"]["epoch_iterations"] == 1000


def test_normalize_config_accepts_progressive_start_alias():
    cfg = normalize_config({"training": {"progressive": {"start": 3}}})
    assert cfg["training"]["progressive"]["start_psf_size"] == 3


def test_normalize_config_accepts_sampling_points_alias():
    cfg = normalize_config({"training": {"progressive": {"training_points_start": 2048}}})
    assert cfg["training"]["progressive"]["sampling_points"] == 2048
    assert "training_points_start" not in cfg["training"]["progressive"]


def test_normalize_config_accepts_sampling_points_start_alias():
    cfg = normalize_config({"training": {"progressive": {"sampling_points_start": 2048}}})
    assert cfg["training"]["progressive"]["sampling_points"] == 2048
    assert "sampling_points_start" not in cfg["training"]["progressive"]


def test_normalize_config_moves_legacy_data_batch_size_to_sampling_points():
    cfg = normalize_config({"data": {"batch_size": 512}})
    assert "batch_size" not in cfg["data"]
    assert cfg["training"]["progressive"]["sampling_points"] == 512


def test_normalize_config_normalizes_explicit_stage_batch_size_alias():
    cfg = normalize_config({
        "training": {
            "progressive": {
                "stages": [
                    {"epochs": 2, "psf_size": 3, "batch_size": 128},
                ],
            },
        },
    })
    stage = cfg["training"]["progressive"]["stages"][0]
    assert stage["psf_size"] == [3, 3]
    assert stage["training_points"] == 128
    assert "batch_size" not in stage


def test_normalize_config_rejects_nonpositive_progressive_values():
    try:
        normalize_config({"training": {"progressive": {"epoch_iterations": 0}}})
    except ValueError as exc:
        assert "training.progressive.epoch_iterations" in str(exc)
    else:
        raise AssertionError("Expected progressive epoch iteration validation to fail")


def test_normalize_config_accepts_psf_target_size_alias():
    cfg = normalize_config({"model": {"psf": {"target_size": 65}}})
    assert cfg["model"]["psf"]["size"] == [65, 65]


def test_normalize_config_validates_every_ten_epochs_by_default():
    cfg = normalize_config({})
    assert cfg["training"]["validation_every_n_epochs"] == 10
    assert cfg["training"]["check_val_every_n_epoch"] == 10
    assert cfg["training"]["num_sanity_val_steps"] == 0


def test_normalize_config_accepts_validation_interval_alias():
    cfg = normalize_config({"training": {"check_val_every_n_epoch": 3}})
    assert cfg["training"]["validation_every_n_epochs"] == 3
    assert cfg["training"]["check_val_every_n_epoch"] == 3


def test_normalize_config_rejects_nonpositive_validation_interval():
    try:
        normalize_config({"training": {"validation_every_n_epochs": 0}})
    except ValueError as exc:
        assert "validation_every_n_epochs" in str(exc)
    else:
        raise AssertionError("Expected validation interval validation to fail")


def test_default_config_logs_to_neuralbd_wandb_project():
    cfg = normalize_config({})
    assert cfg["logging"]["wandb"] is True
    assert cfg["logging"]["project"] == "NeuralBD"


def test_example_configs_use_default_wandb_project():
    for path in [
        "examples/configs/standard_numpy.yaml",
        "examples/configs/spatial_numpy.yaml",
        "examples/configs/gregor_hifi.yaml",
    ]:
        cfg = load_config(path)
        assert cfg["logging"]["wandb"] is True
        assert cfg["logging"]["project"] == "NeuralBD"
        assert "batch_size" not in cfg["data"]


def test_gregor_example_trains_single_selected_channel():
    cfg = load_config("examples/configs/gregor_hifi.yaml")

    assert cfg["data"]["channels"] == [0]
    assert cfg["training"]["progressive"]["sampling_points"] == 524288
    assert cfg["training"]["progressive"]["validation_sampling_points"] == 16384
