import types

import numpy as np
import pytest
import torch

from neuralbd.data import BurstDataModule, BurstDataset
from neuralbd.evaluation import NeuralBDOutput
from neuralbd.io import save_model_state
from neuralbd.io.outputs import restore_image_order
from neuralbd.train.callbacks import (
    NeuralBDOutputCallback,
    ProgressiveTrainingCallback,
    progressive_validation_sampling_points,
    rank_zero_only,
)
from neuralbd.train import NeuralBDModule, normalize_config
from neuralbd.train.cli import build_datamodule, build_logger, checkpoint_psf_size, resume_checkpoint_path, train
from neuralbd.train.pretrain import fit_pretraining_stage, pretrain_image_model


def test_training_and_validation_steps():
    images = np.random.default_rng(1).random((8, 8, 2)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=16, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "method": "standard",
        "model": {
            "image": {"n_channels": 1, "dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    batch = train[0]
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    out = module.validation_step(batch, 0)
    assert torch.isfinite(out)
    assert module.validation_batches[0][0]["convolved_pred"].shape == (16, 2, 1)


def test_training_step_supports_learned_registration_shift_sampled_frames():
    images = np.random.default_rng(8).random((8, 8, 4)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=16, shuffle=False, sample_frames=True)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"n_channels": 1, "dim": 16, "n_layers": 2},
            "psf": {"size": 5},
            "registration": {"enabled": True, "max_pixels": 3, "regularization": 1e-4},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    batch = train[0]

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert module.convolution.raw_frame_shifts.requires_grad


def test_build_datamodule_derives_sampling_points_from_progressive_psf(monkeypatch):
    images = np.random.default_rng(0).random((8, 8, 2)).astype("float32")
    monkeypatch.setattr("neuralbd.train.cli.load_burst_from_config", lambda _cfg: images)
    cfg = normalize_config({
        "model": {"psf": {"target_size": 5}},
        "data": {"num_workers": 3},
        "training": {
            "progressive": {
                "start_psf_size": 1,
                "target_psf_size": 5,
                "sampling_points": 32,
            },
        },
    })

    dm = build_datamodule(cfg)

    assert dm.train_dataset.batch_size == 32
    assert len(dm.train_dataset) == 1000
    assert dm.valid_dataset.batch_size == 16384
    assert dm.num_workers == 3


def test_build_datamodule_samples_training_frames_for_registration(monkeypatch):
    images = np.random.default_rng(5).random((8, 8, 4)).astype("float32")
    monkeypatch.setattr("neuralbd.train.cli.load_burst_from_config", lambda _cfg: images)
    cfg = normalize_config({
        "model": {"registration": {"enabled": True}},
        "training": {"progressive": {"sampling_points": 16}},
    })

    dm = build_datamodule(cfg)
    train_batch = dm.train_dataset[0]
    valid_batch = next(iter(dm.val_dataloader()))

    assert train_batch["images"].shape == (16, 1)
    assert train_batch["frame_indices"].shape == (16,)
    assert valid_batch["images"].shape == (64, 4, 1)
    assert "frame_indices" not in valid_batch
    assert len(dm.valid_dataset) == 64


def test_build_datamodule_validation_exposes_full_point_count(monkeypatch):
    images = np.random.default_rng(6).random((8, 8, 2, 1)).astype("float32")
    monkeypatch.setattr("neuralbd.train.cli.load_burst_from_config", lambda _cfg: images)
    cfg = normalize_config({"training": {"progressive": {"sampling_points": 1024}}})

    dm = build_datamodule(cfg)

    assert len(dm.train_dataset) == 1000
    assert len(dm.valid_dataset) == 64
    assert len(dm.val_dataloader()) == 1


def test_build_datamodule_keeps_fixed_epoch_iterations_after_sampling_change(monkeypatch):
    images = np.random.default_rng(0).random((8, 8, 2)).astype("float32")
    monkeypatch.setattr("neuralbd.train.cli.load_burst_from_config", lambda _cfg: images)
    cfg = normalize_config({
        "model": {"psf": {"target_size": 7}},
        "training": {
            "progressive": {
                "start_psf_size": 1,
                "target_psf_size": 7,
                "sampling_points": 32,
                "fixed_epoch_size": True,
                "epoch_iterations": 7,
            },
        },
    })

    dm = build_datamodule(cfg)
    assert len(dm.train_dataset) == 7

    dm.set_train_sampling_points(3)

    assert dm.train_dataset.batch_size == 3
    assert len(dm.train_dataset) == 7


def test_build_logger_stores_wandb_under_work_dir(tmp_path, monkeypatch):
    calls = []

    class WandbLoggerStub:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("neuralbd.train.cli.WandbLogger", WandbLoggerStub)
    cfg = normalize_config({
        "work_dir": str(tmp_path / "work"),
        "logging": {"wandb": True, "project": "Project", "name": "Run"},
    })

    logger = build_logger(cfg)

    assert isinstance(logger, WandbLoggerStub)
    assert calls == [{
        "project": "Project",
        "name": "Run",
        "save_dir": tmp_path / "work" / "wandb",
        "dir": tmp_path / "work" / "wandb",
    }]
    assert (tmp_path / "work" / "wandb").is_dir()


def test_resume_checkpoint_path_uses_last_checkpoint(tmp_path):
    base_dir = tmp_path / "run"
    base_dir.mkdir()
    last = base_dir / "last.ckpt"
    last.write_bytes(b"checkpoint")

    cfg = normalize_config({"base_dir": str(base_dir)})

    assert resume_checkpoint_path(cfg, base_dir) == last


def test_resume_checkpoint_path_accepts_explicit_last(tmp_path):
    base_dir = tmp_path / "run"
    base_dir.mkdir()
    last = base_dir / "last.ckpt"
    last.write_bytes(b"checkpoint")

    cfg = normalize_config({
        "base_dir": str(base_dir),
        "training": {"resume_from_checkpoint": "last"},
    })

    assert resume_checkpoint_path(cfg, base_dir) == last


def test_train_resumes_from_lightning_last_checkpoint(tmp_path, monkeypatch):
    base_dir = tmp_path / "run"
    work_dir = tmp_path / "work"
    base_dir.mkdir()
    last = base_dir / "last.ckpt"
    torch.save({"state_dict": {"convolution.psf_coords": torch.zeros(3, 3, 2)}}, last)
    images = np.random.default_rng(12).random((4, 4, 1, 1)).astype("float32")
    cfg = normalize_config({
        "base_dir": str(base_dir),
        "work_dir": str(work_dir),
        "logging": {"wandb": False},
        "training": {"epochs": 1, "accelerator": "cpu", "devices": 1},
        "outputs": {"save_validation_figures": False},
    })
    calls = {}

    class TrainerStub:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs
            calls["callbacks"] = [type(callback).__name__ for callback in kwargs["callbacks"]]

        def fit(self, module, datamodule, ckpt_path=None):
            calls["ckpt_path"] = ckpt_path
            calls["psf_size"] = module.convolution.psf_size

    monkeypatch.setattr("neuralbd.train.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("neuralbd.train.cli.load_burst_from_config", lambda _cfg: images)
    monkeypatch.setattr("neuralbd.train.cli.Trainer", TrainerStub)
    monkeypatch.setattr("neuralbd.train.cli.fit_pretraining_stage", lambda *args, **kwargs: calls.setdefault("pretrain", True))

    train("config.yaml")

    assert calls["ckpt_path"] == str(last)
    assert calls["psf_size"] == (3, 3)
    assert calls["trainer_kwargs"]["reload_dataloaders_every_n_epochs"] == 1
    assert calls["callbacks"].index("ProgressiveTrainingCallback") < calls["callbacks"].index("NeuralBDOutputCallback")
    assert "pretrain" not in calls


def test_checkpoint_psf_size_reads_lightning_state_dict(tmp_path):
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"convolution.psf_coords": torch.zeros(31, 31, 2)}}, path)

    assert checkpoint_psf_size(path) == (31, 31)


@pytest.mark.parametrize("method", ["standard", "spatial"])
@pytest.mark.parametrize("channel_mode", ["shared", "per_channel"])
@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_training_step_supports_arbitrary_channels(method, channel_mode, n_channels):
    images = np.random.default_rng(2).random((8, 8, 2, n_channels)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=16, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "method": method,
        "model": {
            "image": {"n_channels": n_channels, "dim": 16, "n_layers": 2},
            "psf": {"size": 5, "channel_mode": channel_mode, "dim": 16, "n_layers": 2},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    batch = train[0]
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    out = module.validation_step(batch, 0)
    assert torch.isfinite(out)
    assert module.validation_batches[0][0]["convolved_pred"].shape == (16, 2, n_channels)


def test_pretrain_image_model_runs():
    images = np.random.default_rng(3).random((8, 8, 2, 2)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=16, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"n_channels": 2, "dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "pretraining": {"enabled": True, "epochs": 1, "learning_rate": 1e-4, "target": "first_frame"},
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    history = pretrain_image_model(module, dm, cfg)
    assert len(history) == 1
    assert np.isfinite(history[0])


def test_fit_pretraining_stage_runs():
    images = np.random.default_rng(4).random((4, 4, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=8, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "pretraining": {"enabled": True, "epochs": 1, "learning_rate": 1e-4},
        "training": {
            "accelerator": "cpu",
            "devices": 1,
            "learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10},
            "log_every_n_steps": 1,
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    trainer = fit_pretraining_stage(module, dm, cfg, logger=False)
    assert trainer is not None


def test_restore_validation_outputs_uses_indices():
    images_shape = (2, 2, 1, 1)
    array = np.array([[[3.0]], [[1.0]], [[0.0]], [[2.0]]], dtype="float32")
    indices = np.array([3, 1, 0, 2])
    restored = restore_image_order(array, indices, images_shape, "convolved_pred")
    assert restored[:, :, 0, 0].tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_validation_outputs_are_named_and_sorted():
    images = np.random.default_rng(7).random((4, 4, 2, 1)).astype("float32")
    valid = BurstDataset.from_numpy(images, batch_size=5, shuffle=False)
    dm = BurstDataModule(valid)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    module.on_validation_epoch_start()
    for idx, batch in enumerate(valid):
        out = module.validation_step(batch, idx)
        module.on_validation_batch_end(out, batch, idx)
    module.on_validation_epoch_end()

    assert "validation" in module.validation_outputs
    outputs = module.validation_outputs["validation"]
    assert tuple(outputs["images_shape"].tolist()) == dm.images_shape
    assert outputs["indices"].tolist() == sorted(outputs["indices"].tolist())


def test_validation_step_collects_batches_without_batch_end_hook():
    images = np.random.default_rng(7).random((4, 4, 2, 1)).astype("float32")
    valid = BurstDataset.from_numpy(images, batch_size=5, shuffle=False)
    dm = BurstDataModule(valid)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    module.on_validation_epoch_start()
    for idx, batch in enumerate(valid):
        module.validation_step(batch, idx)
    module.on_validation_epoch_end()

    outputs = module.validation_outputs["validation"]
    assert outputs["convolved_true"].shape[0] == 16


def test_validation_merge_does_not_drop_duplicate_dataset_idx():
    images_shape = (2, 2, 1, 1)
    module = NeuralBDModule(images_shape=images_shape, image_config={"dim": 8, "n_layers": 1}, psf_config={"size": [3, 3]})
    outputs_by_loader = {
        0: [
            {
                "dataset_idx": torch.tensor([0]),
                "indices": torch.tensor([0, 1]),
                "convolved_true": torch.zeros(2, 1, 1),
                "convolved_pred": torch.zeros(2, 1, 1),
                "image_pred": torch.zeros(2, 1),
                "loss": torch.zeros(1),
                "images_shape": torch.tensor(images_shape),
            },
            {
                "dataset_idx": torch.tensor([0]),
                "indices": torch.tensor([2, 3]),
                "convolved_true": torch.ones(2, 1, 1),
                "convolved_pred": torch.ones(2, 1, 1),
                "image_pred": torch.ones(2, 1),
                "loss": torch.ones(1),
                "images_shape": torch.tensor(images_shape),
            },
        ]
    }

    merged = module.merge_validation_batches(outputs_by_loader)["validation"]

    assert merged["convolved_true"].shape[0] == 4
    assert merged["indices"].tolist() == [0, 1, 2, 3]


def test_validation_merge_drops_exact_duplicate_indexed_batches():
    images_shape = (2, 2, 1, 1)
    module = NeuralBDModule(images_shape=images_shape, image_config={"dim": 8, "n_layers": 1}, psf_config={"size": [3, 3]})
    batch = {
        "dataset_idx": torch.tensor([0]),
        "indices": torch.tensor([0, 1]),
        "convolved_true": torch.zeros(2, 1, 1),
        "convolved_pred": torch.zeros(2, 1, 1),
        "image_pred": torch.zeros(2, 1),
        "loss": torch.zeros(1),
        "images_shape": torch.tensor(images_shape),
    }
    outputs_by_loader = {0: [batch, {key: value.clone() for key, value in batch.items()}]}

    merged = module.merge_validation_batches(outputs_by_loader)["validation"]

    assert merged["convolved_true"].shape[0] == 2
    assert merged["indices"].tolist() == [0, 1]


def test_output_callback_logs_validation_figures_with_wandb_figure(tmp_path, monkeypatch):
    calls = []
    figure = object()

    def image(value):
        calls.append(("image", value))
        return ("wandb-image", value)

    monkeypatch.setitem(__import__("sys").modules, "wandb", types.SimpleNamespace(Image=image))

    class ExperimentStub:
        def log(self, data, step=None):
            calls.append(("log", data, step))

    class LoggerStub:
        experiment = ExperimentStub()

    class TrainerStub:
        global_step = 12
        logger = LoggerStub()

    callback = NeuralBDOutputCallback()
    logged = callback._log_wandb(TrainerStub(), "input_vs_reconstruction", figure)

    assert calls == [
        ("image", figure),
        ("log", {"validation/input_vs_reconstruction": ("wandb-image", figure)}, 12),
    ]
    assert logged is True


def test_output_callback_logs_wandb_only_on_primary_process(tmp_path, monkeypatch):
    calls = []
    figure = object()

    def image(value):
        calls.append(("image", value))
        return ("wandb-image", value)

    monkeypatch.setitem(__import__("sys").modules, "wandb", types.SimpleNamespace(Image=image))

    class ExperimentStub:
        def log(self, data, step=None):
            calls.append(("log", data, step))

    class LoggerStub:
        experiment = ExperimentStub()

    class TrainerStub:
        global_step = 12
        logger = LoggerStub()

    callback = NeuralBDOutputCallback()
    original_rank = getattr(rank_zero_only, "rank", 0)
    try:
        rank_zero_only.rank = 1
        logged = callback._log_wandb(TrainerStub(), "input_vs_reconstruction", figure)
    finally:
        rank_zero_only.rank = original_rank

    assert calls == []
    assert logged is None


def test_validation_plot_axes_use_pixel_units():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    callback = NeuralBDOutputCallback(sample_count=1)
    observed = np.zeros((4, 6, 2, 1), dtype="float32")
    reconstructed = np.zeros((4, 6, 1), dtype="float32")
    fig = callback._plot_input_vs_reconstruction(
        plt,
        observed,
        reconstructed,
        sample_channels=[0],
        pixel_per_ds=2.0,
    )

    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "x [px]"
        assert ax.get_ylabel() == "y [px]"
        assert ax.images[0].get_extent() == [-3.0, 3.0, -2.0, 2.0]
        assert ax.get_title() == "first frame c0"
        assert ax.axison
        assert len(fig.axes) == 3
        assert fig.axes[2].get_xlabel() == "intensity [normalized]"
    finally:
        plt.close(fig)


def test_validation_colorbar_uses_displayed_panel_min_max():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    callback = NeuralBDOutputCallback(sample_count=1)
    reference = np.ones((2, 2, 2, 2), dtype="float32")
    predicted = np.full((2, 2, 2, 2), 2.0, dtype="float32")
    reference[:, :, 1, 1] = 999.0
    predicted[:, :, 1, 1] = 1000.0
    fig = callback._plot_predicted_vs_reference(
        plt,
        reference,
        predicted,
        sample_frames=[0, 1],
        sample_channels=[0, 1],
        pixel_per_ds=1.0,
    )

    try:
        assert fig.axes[0].images[0].get_clim() == (1.0, 2.0)
    finally:
        plt.close(fig)


def test_validation_colorbar_ignores_isolated_displayed_outlier():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    callback = NeuralBDOutputCallback(sample_count=1)
    observed = np.zeros((100, 100, 1, 1), dtype="float32")
    reconstructed = np.ones((100, 100, 1), dtype="float32")
    reconstructed[12, 34, 0] = 2.5
    fig = callback._plot_input_vs_reconstruction(
        plt,
        observed,
        reconstructed,
        sample_channels=[0],
        pixel_per_ds=1.0,
    )

    try:
        assert fig.axes[0].images[0].get_clim() == (0.0, 1.0)
    finally:
        plt.close(fig)


def test_validation_figures_reject_partial_pixel_outputs():
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback()
    module = types.SimpleNamespace(images_shape=(4, 4, 1, 1))
    outputs = {
        "images_shape": torch.tensor([4, 4, 1, 1]),
        "convolved_true": torch.zeros(8, 1, 1),
    }

    with pytest.raises(RuntimeError, match="require all 16 pixels"):
        callback._build_figures(module, outputs)


def test_validation_figures_use_center_subregion(monkeypatch):
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback(sample_count=1, figure_subregion_size=4)
    images_shape = (6, 8, 2, 1)
    n_pixels = images_shape[0] * images_shape[1]
    outputs = {
        "images_shape": torch.tensor(images_shape),
        "indices": torch.arange(n_pixels),
        "convolved_true": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "convolved_pred": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "image_pred": torch.zeros(n_pixels, images_shape[3]),
    }
    seen = {}

    def plot_input(_plt, observed, reconstructed, sample_channels, pixel_per_ds, image_extent=None):
        seen["input"] = (observed.shape, reconstructed.shape, image_extent)
        return "input"

    def plot_predicted(
        _plt,
        reference,
        predicted,
        sample_frames,
        sample_channels,
        pixel_per_ds,
        image_extent=None,
    ):
        seen["predicted"] = (reference.shape, predicted.shape, image_extent)
        return "predicted"

    monkeypatch.setattr(callback, "_plot_psfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(callback, "_plot_input_vs_reconstruction", plot_input)
    monkeypatch.setattr(callback, "_plot_predicted_vs_reference", plot_predicted)

    pl_module = types.SimpleNamespace(images_shape=images_shape, pixel_per_ds=1.0)
    figures = callback._build_figures(pl_module, outputs)

    assert figures == {"input_vs_reconstruction": "input", "predicted_vs_reference": "predicted"}
    assert seen["input"] == ((4, 4, 1, 1), (4, 4, 1), (-2.0, 2.0, -2.0, 2.0))
    assert seen["predicted"] == ((4, 4, 1, 1), (4, 4, 1, 1), (-2.0, 2.0, -2.0, 2.0))


def test_validation_figures_use_configured_centered_subregion(monkeypatch):
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback(
        sample_count=1,
        figure_subregion={"x": [1, 3], "y": [-2, 0]},
    )
    images_shape = (6, 8, 2, 1)
    n_pixels = images_shape[0] * images_shape[1]
    outputs = {
        "images_shape": torch.tensor(images_shape),
        "indices": torch.arange(n_pixels),
        "convolved_true": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "convolved_pred": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "image_pred": torch.zeros(n_pixels, images_shape[3]),
    }
    seen = {}

    def plot_input(_plt, observed, reconstructed, sample_channels, pixel_per_ds, image_extent=None):
        seen["input"] = (observed.shape, reconstructed.shape, image_extent)
        return "input"

    monkeypatch.setattr(callback, "_plot_psfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(callback, "_plot_input_vs_reconstruction", plot_input)
    monkeypatch.setattr(callback, "_plot_predicted_vs_reference", lambda *args, **kwargs: None)

    pl_module = types.SimpleNamespace(images_shape=images_shape, pixel_per_ds=1.0)
    figures = callback._build_figures(pl_module, outputs)

    assert figures == {"input_vs_reconstruction": "input"}
    assert seen["input"] == ((2, 2, 1, 1), (2, 2, 1), (1.0, 3.0, -2.0, 0.0))


def test_validation_configured_subregion_is_plotted_with_lower_origin(monkeypatch):
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback(
        sample_count=1,
        figure_subregion={"x": [-1, 1], "y": [-1, 1]},
    )
    images_shape = (4, 4, 1, 1)
    n_pixels = images_shape[0] * images_shape[1]
    true = torch.arange(n_pixels, dtype=torch.float32).reshape(n_pixels, 1, 1)
    outputs = {
        "images_shape": torch.tensor(images_shape),
        "indices": torch.arange(n_pixels),
        "convolved_true": true,
        "convolved_pred": true,
        "image_pred": torch.zeros(n_pixels, images_shape[3]),
    }
    seen = {}

    def plot_input(_plt, observed, reconstructed, sample_channels, pixel_per_ds, image_extent=None):
        seen["observed"] = observed[:, :, 0, 0].copy()
        seen["extent"] = image_extent
        return "input"

    monkeypatch.setattr(callback, "_plot_psfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(callback, "_plot_input_vs_reconstruction", plot_input)
    monkeypatch.setattr(callback, "_plot_predicted_vs_reference", lambda *args, **kwargs: None)

    pl_module = types.SimpleNamespace(images_shape=images_shape, pixel_per_ds=1.0)
    figures = callback._build_figures(pl_module, outputs)

    assert figures == {"input_vs_reconstruction": "input"}
    assert seen["extent"] == (-1.0, 1.0, -1.0, 1.0)
    assert seen["observed"].tolist() == [[9.0, 10.0], [5.0, 6.0]]


def test_validation_figures_skip_invalid_x_subregion(monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback(
        sample_count=1,
        figure_subregion={"x": [-10, 2], "y": [-1, 1]},
    )
    images_shape = (6, 8, 2, 1)
    n_pixels = images_shape[0] * images_shape[1]
    outputs = {
        "images_shape": torch.tensor(images_shape),
        "indices": torch.arange(n_pixels),
        "convolved_true": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "convolved_pred": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "image_pred": torch.zeros(n_pixels, images_shape[3]),
    }
    monkeypatch.setattr(callback, "_plot_psfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(callback, "_plot_input_vs_reconstruction", lambda *args, **kwargs: "input")
    monkeypatch.setattr(callback, "_plot_predicted_vs_reference", lambda *args, **kwargs: None)

    pl_module = types.SimpleNamespace(images_shape=images_shape, pixel_per_ds=1.0)
    figures = callback._build_figures(pl_module, outputs)

    assert figures == {}
    message = capsys.readouterr().out
    assert "Invalid outputs.figure_subregion.x" in message
    assert "valid range is [-4.0, 4.0]" in message


def test_validation_figures_skip_invalid_y_subregion(monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    callback = NeuralBDOutputCallback(
        sample_count=1,
        figure_subregion={"x": [100, 200], "y": [300, 400]},
    )
    images_shape = (600, 600, 2, 1)
    n_pixels = images_shape[0] * images_shape[1]
    outputs = {
        "images_shape": torch.tensor(images_shape),
        "indices": torch.arange(n_pixels),
        "convolved_true": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "convolved_pred": torch.zeros(n_pixels, images_shape[2], images_shape[3]),
        "image_pred": torch.zeros(n_pixels, images_shape[3]),
    }
    monkeypatch.setattr(callback, "_plot_psfs", lambda *args, **kwargs: None)
    monkeypatch.setattr(callback, "_plot_input_vs_reconstruction", lambda *args, **kwargs: "input")
    monkeypatch.setattr(callback, "_plot_predicted_vs_reference", lambda *args, **kwargs: None)

    pl_module = types.SimpleNamespace(images_shape=images_shape, pixel_per_ds=1.0)
    figures = callback._build_figures(pl_module, outputs)

    assert figures == {}
    message = capsys.readouterr().out
    assert "Invalid outputs.figure_subregion.y" in message
    assert "valid range is [-300.0, 300.0]" in message


def test_validation_psf_plot_has_axes_and_colorbar():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = np.random.default_rng(11).random((4, 4, 1, 1)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=8, shuffle=False)
    dm = BurstDataModule(dataset)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = NeuralBDOutputCallback(sample_count=1)
    fig = callback._plot_psfs(plt, module, sample_frames=[0], sample_channels=[0])

    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "dx [px]"
        assert ax.get_ylabel() == "dy [px]"
        assert ax.images[0].get_extent() == [-2.5, 2.5, -2.5, 2.5]
        assert len(fig.axes) == 2
        assert fig.axes[1].get_xlabel() == "PSF density [px^-2]"
    finally:
        plt.close(fig)


def test_validation_psf_extent_matches_pixel_grid_units():
    assert NeuralBDOutputCallback._centered_extent((5, 5), pixel_per_ds=2.0) == (-2.5, 2.5, -2.5, 2.5)
    assert NeuralBDOutputCallback._centered_extent((4, 4), pixel_per_ds=2.0) == pytest.approx(
        (-2.6666666667, 2.6666666667, -2.6666666667, 2.6666666667)
    )
    assert NeuralBDOutputCallback._psf_density_per_pixel(np.ones((2, 2)), pixel_per_ds=2.0)[0, 0] == 0.25


def test_checkpoint_stores_state_dict_and_reloads(tmp_path):
    images = np.random.default_rng(5).random((4, 4, 2, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=8, shuffle=False)
    dm = BurstDataModule(dataset)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5, "channel_mode": "per_channel"},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    path = tmp_path / "model.nbd"
    save_model_state(path, module, config=cfg)

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in state
    assert "image_model" not in state
    assert "psf_model" not in state
    assert state["metadata"]["images_shape"] == dm.images_shape
    assert state["metadata"]["coordinate_order"] == "x,y"
    assert state["metadata"]["coordinate_origin"] == "center"
    assert state["metadata"]["psf_size"] == (5, 5)
    assert state["metadata"]["active_psf_size"] == (5, 5)
    assert state["metadata"]["learned_psf_size"] == (5, 5)

    reconstruction = NeuralBDOutput(path, device="cpu").reconstruct(batch_size=8)
    assert reconstruction.shape == (4, 4, 2)


def test_checkpoint_restores_active_psf_size(tmp_path):
    images = np.random.default_rng(6).random((4, 4, 2, 1)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=8, shuffle=False)
    dm = BurstDataModule(dataset)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {"learning_rate": {"start": 1e-4, "end": 1e-4, "iterations": 10}},
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    module.set_psf_size((3, 3))
    path = tmp_path / "model.nbd"
    save_model_state(path, module, config=cfg)

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["metadata"]["psf_size"] == (3, 3)
    assert state["metadata"]["active_psf_size"] == (3, 3)
    assert state["metadata"]["learned_psf_size"] == (7, 7)

    restored = NeuralBDOutput(path, device="cpu").module
    assert restored.convolution.psf_size == (3, 3)
    assert restored.psf_model.active_psf_size == (3, 3)


def test_progressive_training_callback_updates_psf_and_batch_size():
    images = np.random.default_rng(6).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    valid = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train, valid)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {
            "epochs": 3,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 3,
                "target_psf_size": 7,
                "increase_every_n_epochs": 1,
                "sampling_points": 32,
                "epoch_iterations": 4,
                "learning_rate_start": 1e-3,
                "learning_rate_end": 1e-4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)

    class TrainerStub:
        max_epochs = 3
        current_epoch = 0
        datamodule = dm
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-3)]

    trainer = TrainerStub()
    callback.setup(trainer, module)
    assert module.convolution.psf_size == (3, 3)
    assert dm.train_dataset.batch_size == 32
    assert dm.valid_dataset.batch_size == 16384
    assert len(dm.train_dataset) == 4

    callback.on_train_epoch_start(trainer, module)
    assert module.convolution.psf_size == (3, 3)
    assert dm.train_dataset.batch_size == 32
    assert dm.valid_dataset.batch_size == 16384
    assert len(dm.train_dataset) == 4
    original_psf_parameter = module.psf_model.log_psfs

    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, module)
    assert module.convolution.psf_size == (5, 5)
    assert dm.train_dataset.batch_size == 12
    assert dm.valid_dataset.batch_size == 5898
    assert len(dm.train_dataset) == 4
    assert 32 * 3 * 3 == pytest.approx(dm.train_dataset.batch_size * 5 * 5, rel=0.15)
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx((1e-3 * 1e-4) ** 0.5)
    trainer.current_epoch = 2
    callback.on_train_epoch_start(trainer, module)
    assert module.convolution.psf_size == (7, 7)
    assert dm.train_dataset.batch_size == 6
    assert dm.valid_dataset.batch_size == 3009
    assert module.psf_model.log_psfs is original_psf_parameter
    optimizer_params = {param for group in trainer.optimizers[0].param_groups for param in group["params"]}
    assert module.psf_model.log_psfs in optimizer_params


def test_progressive_sampling_points_decrease_as_psf_grows():
    cfg = normalize_config({
        "model": {"psf": {"size": 65}},
        "training": {
            "epochs": 33,
            "progressive": {
                "enabled": True,
                "start_psf_size": 1,
                "target_psf_size": 65,
                "increase_every_n_epochs": 1,
                "sampling_points": 524288,
                "validation_sampling_points": 16384,
            },
        },
    })
    callback = ProgressiveTrainingCallback(cfg)
    stages = callback._build_stages(max_epochs=33)
    psf_sizes = [stage["psf_size"][0] for stage in stages]
    train_points = [stage["training_points"] for stage in stages]
    val_points = [
        progressive_validation_sampling_points(cfg, psf_size=stage["psf_size"])
        for stage in stages
    ]

    assert psf_sizes == list(range(1, 66, 2))
    assert train_points == sorted(train_points, reverse=True)
    assert val_points == sorted(val_points, reverse=True)
    assert train_points[0] == 524288
    assert train_points[-1] == 124
    assert val_points[0] == 16384
    assert val_points[-1] == 4
    assert train_points[0] * psf_sizes[0] ** 2 == pytest.approx(
        train_points[-1] * psf_sizes[-1] ** 2,
        rel=0.01,
    )


def test_progressive_training_callback_updates_validation_stage_directly():
    images = np.random.default_rng(10).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    valid = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train, valid)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {
            "epochs": 3,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 3,
                "target_psf_size": 7,
                "increase_every_n_epochs": 1,
                "sampling_points": 32,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)

    class TrainerStub:
        max_epochs = 3
        current_epoch = 0
        datamodule = dm
        optimizers = []

    trainer = TrainerStub()
    callback.setup(trainer, module)
    assert module.convolution.psf_size == (3, 3)
    assert dm.valid_dataset.batch_size == 16384

    trainer.current_epoch = 1
    callback.on_validation_epoch_start(trainer, module)
    assert module.convolution.psf_size == (5, 5)
    assert dm.valid_dataset.batch_size == 5898


def test_progressive_training_callback_uses_explicit_validation_sampling_points():
    images = np.random.default_rng(10).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    valid = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train, valid)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {
            "epochs": 3,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 3,
                "target_psf_size": 7,
                "increase_every_n_epochs": 1,
                "sampling_points": 32,
                "validation_sampling_points": 16,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)

    class TrainerStub:
        max_epochs = 3
        current_epoch = 1
        datamodule = dm
        optimizers = []

    callback.setup(TrainerStub(), module)
    callback.on_validation_epoch_start(TrainerStub(), module)

    assert dm.valid_dataset.batch_size == 6


def test_progressive_training_callback_does_not_reset_psf_during_resume_setup():
    images = np.random.default_rng(10).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {
            "epochs": 3,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 1,
                "target_psf_size": 7,
                "increase_every_n_epochs": 1,
                "sampling_points": 32,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    module.set_psf_size((5, 5))
    callback = ProgressiveTrainingCallback(cfg, resume=True)

    class TrainerStub:
        max_epochs = 3
        current_epoch = 0
        datamodule = dm
        optimizers = []

    callback.setup(TrainerStub(), module)

    assert module.convolution.psf_size == (5, 5)


def test_progressive_training_callback_logs_only_on_primary_process(monkeypatch, capsys):
    images = np.random.default_rng(6).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 3},
        },
        "training": {
            "epochs": 1,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 1,
                "target_psf_size": 3,
                "sampling_points": 32,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)
    log_calls = []
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: log_calls.append((args, kwargs)))

    class TrainerStub:
        max_epochs = 1
        current_epoch = 0
        datamodule = dm
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-3)]

    trainer = TrainerStub()
    callback.setup(trainer, module)
    original_rank = getattr(rank_zero_only, "rank", 0)
    try:
        rank_zero_only.rank = 1
        callback.on_train_epoch_start(trainer, module)
    finally:
        rank_zero_only.rank = original_rank

    assert module.convolution.psf_size == (1, 1)
    assert log_calls == []
    assert "Progressive PSF stage" not in capsys.readouterr().out


def test_progressive_training_callback_defaults_to_unit_psf():
    images = np.random.default_rng(7).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 7},
        },
        "training": {
            "epochs": 2,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "target_psf_size": 7,
                "sampling_points": 32,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)

    class TrainerStub:
        max_epochs = 2
        current_epoch = 0
        datamodule = dm
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-3)]

    trainer = TrainerStub()
    callback.setup(trainer, module)
    callback.on_train_epoch_start(trainer, module)

    assert module.convolution.psf_size == (1, 1)
    assert dm.train_dataset.batch_size == 32
    assert len(dm.train_dataset) == 4


def test_progressive_training_callback_continues_final_psf_after_growth():
    images = np.random.default_rng(8).random((8, 8, 2, 1)).astype("float32")
    train = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dm = BurstDataModule(train)
    cfg = normalize_config({
        "model": {
            "image": {"dim": 16, "n_layers": 2},
            "psf": {"size": 5},
        },
        "training": {
            "epochs": 5,
            "learning_rate": {"start": 1e-3, "end": 1e-4, "iterations": 10},
            "progressive": {
                "enabled": True,
                "start_psf_size": 1,
                "target_psf_size": 5,
                "increase_every_n_epochs": 1,
                "sampling_points": 32,
                "epoch_iterations": 4,
            },
        },
    })
    module = NeuralBDModule.from_config(cfg, images_shape=dm.images_shape)
    callback = ProgressiveTrainingCallback(cfg)

    class TrainerStub:
        max_epochs = 5
        current_epoch = 0
        datamodule = dm
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-3)]

    trainer = TrainerStub()
    callback.setup(trainer, module)
    assert callback.stages[-1]["end_epoch"] == 5

    for epoch in range(5):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, module)

    assert module.convolution.psf_size == (5, 5)
    assert dm.train_dataset.batch_size == 1
    assert len(dm.train_dataset) == 4
