from neuralbd.train import normalize_config


def test_normalize_config_sets_spatial_psf():
    cfg = normalize_config({"method": "spatial", "model": {"psf": {"size": 9}}})
    assert cfg["method"] == "spatial"
    assert cfg["model"]["psf"]["type"] == "spatial"
    assert cfg["model"]["psf"]["size"] == [9, 9]


def test_normalize_config_sets_output_directory():
    cfg = normalize_config({"base_dir": "runs/test"})
    assert cfg["outputs"]["directory"] == "runs/test/outputs"


def test_normalize_config_accepts_per_channel_psf():
    cfg = normalize_config({"model": {"psf": {"channel_mode": "per_channel"}}})
    assert cfg["model"]["psf"]["channel_mode"] == "per_channel"


def test_normalize_config_sets_spatial_siren_psf():
    cfg = normalize_config({"method": "spatial"})
    assert cfg["model"]["psf"]["type"] == "spatial"
    assert cfg["model"]["psf"]["representation"] == "siren"


def test_normalize_config_progressive_target_defaults_to_psf_size():
    cfg = normalize_config({"model": {"psf": {"size": 65}}})
    assert cfg["training"]["progressive"]["target_psf_size"] == 65
