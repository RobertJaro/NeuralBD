import numpy as np

from neuralbd.data import BurstDataset, load_burst_from_config


def test_burst_dataset_from_numpy():
    images = np.random.default_rng(0).random((8, 8, 3)).astype("float32")
    dataset = BurstDataset.from_numpy(images, pixel_per_ds=8.0, batch_size=16, shuffle=False)
    batch = dataset[0]
    batch_images = batch["images"]
    batch_coords = batch["coords"]
    assert batch_images.shape == (16, 3, 1)
    assert batch_coords.shape == (16, 2)
    assert batch["indices"].shape == (16,)
    assert dataset.images_shape == (8, 8, 3, 1)


def test_burst_dataset_keeps_partial_batch():
    images = np.random.default_rng(0).random((7, 7, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=16, shuffle=False)
    batch_images = dataset[-1]["images"]
    assert len(dataset) == 4
    assert batch_images.shape == (1, 2, 1)


def test_load_burst_from_config_applies_subframe_frames_and_channels(tmp_path):
    path = tmp_path / "burst.npz"
    images = np.arange(4 * 6 * 7 * 3, dtype="float32").reshape(4, 6, 7, 3)
    np.savez(path, cobs=images)
    loaded = load_burst_from_config({
        "type": "dkist",
        "path": str(path),
        "array_key": "cobs",
        "frame_indices": [1, 3],
        "channels": [0, 2],
        "subframe": {"x": 3, "y": 3, "size": 4},
    })
    assert loaded.shape == (4, 4, 2, 2)
