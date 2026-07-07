import numpy as np
import pytest
import torch

from nbd.data import BurstDataModule, BurstDataset
from nbd.evaluation import NeuralBDOutput
from nbd.io import save_model_state
from nbd.io.outputs import restore_image_order
from nbd.train.callbacks import ProgressiveTrainingCallback
from nbd.train import NeuralBDModule, normalize_config
from nbd.train.pretrain import fit_pretraining_stage, pretrain_image_model


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
    assert out["convolved_pred"].shape == (16, 2, 1)


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
    assert out["convolved_pred"].shape == (16, 2, n_channels)


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

    reconstruction = NeuralBDOutput(path, device="cpu").reconstruct(batch_size=8)
    assert reconstruction.shape == (4, 4, 2)


def test_progressive_training_callback_updates_psf_and_batch_size():
    images = np.random.default_rng(6).random((8, 8, 2, 1)).astype("float32")
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
                "start_psf_size": 3,
                "target_psf_size": 7,
                "n_stages": 2,
                "training_points_start": 32,
                "training_points_end": 8,
                "learning_rate_start": 1e-3,
                "learning_rate_end": 1e-4,
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
    assert module.convolution.psf_size == (3, 3)
    assert dm.train_dataset.batch_size == 32

    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, module)
    assert module.convolution.psf_size == (7, 7)
    assert dm.train_dataset.batch_size == 8
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-4)
    optimizer_params = {param for group in trainer.optimizers[0].param_groups for param in group["params"]}
    assert module.psf_model.log_psfs in optimizer_params
