import json
from types import SimpleNamespace

import numpy as np
import torch

from neuralbd.data import BurstDataModule, BurstDataset, load_burst_from_config, to_channels_last
from neuralbd.data.loaders import group_interleaved_channels
from neuralbd.cli.prepare_gregor import (
    build_alignment_loader_config,
    build_loader_config,
    build_metadata,
    save_prepared_burst,
    trim_alignment_buffer,
)
from neuralbd.processing import align_stack, bounded_cross_correlation_shift, image_coordinates, normalize


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


def test_burst_dataset_can_sample_frames_per_coordinate():
    images = np.random.default_rng(1).random((8, 8, 3, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=16, shuffle=False, sample_frames=True)
    batch = dataset[0]

    assert batch["images"].shape == (16, 2)
    assert batch["coords"].shape == (16, 2)
    assert batch["indices"].shape == (16,)
    assert batch["frame_indices"].shape == (16,)
    assert torch.all((batch["frame_indices"] >= 0) & (batch["frame_indices"] < 3))


def test_image_coordinates_are_centered_xy_pixels():
    coords = image_coordinates((4, 6, 1), pixel_per_ds=1.0)

    assert coords[0, 0].tolist() == [-2.5, 1.5]
    assert coords[-1, -1].tolist() == [2.5, -1.5]


def test_image_coordinates_scale_by_pixel_per_ds():
    coords = image_coordinates((4, 6, 1), pixel_per_ds=2.0)

    assert coords[0, 0].tolist() == [-1.25, 0.75]
    assert coords[-1, -1].tolist() == [1.25, -0.75]


def test_minmax_normalization_maps_finite_burst_to_unit_range():
    images = np.array([[[[2.0], [4.0]], [[6.0], [10.0]]]], dtype="float32")
    normalized = normalize(images, "minmax")

    assert normalized.dtype == np.float32
    assert np.nanmin(normalized) == 0.0
    assert np.nanmax(normalized) == 1.0


def test_align_stack_registers_shifted_frames():
    reference = np.zeros((8, 8), dtype="float32")
    reference[2:5, 3:6] = 1.0
    shifted = np.zeros_like(reference)
    shifted[3:6, 1:4] = reference[2:5, 3:6]
    images = np.stack([reference, shifted], axis=2)

    aligned, shifts = align_stack(images, max_shift=3)

    assert shifts == [[0, 0], [-1, 2]]
    assert np.allclose(aligned[:, :, 1], reference)


def test_bounded_cross_correlation_shift_selects_best_shift():
    reference = np.zeros((8, 8), dtype="float32")
    reference[2:5, 3:6] = 1.0
    shifted = np.zeros_like(reference)
    shifted[3:6, 1:4] = reference[2:5, 3:6]

    shift_x, shift_y, score = bounded_cross_correlation_shift(reference, shifted, max_shift=3)

    assert (shift_x, shift_y) == (-1, 2)
    assert score > 0.99


def test_align_stack_can_use_multiple_workers():
    reference = np.zeros((8, 8), dtype="float32")
    reference[2:5, 3:6] = 1.0
    shifted = np.zeros_like(reference)
    shifted[3:6, 1:4] = reference[2:5, 3:6]
    images = np.stack([reference, shifted], axis=2)

    aligned, shifts = align_stack(images, max_shift=3, num_workers=2)

    assert shifts == [[0, 0], [-1, 2]]
    assert np.allclose(aligned[:, :, 1], reference)


def test_burst_dataset_keeps_partial_batch():
    images = np.random.default_rng(0).random((7, 7, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=16, shuffle=False)
    batch_images = dataset[-1]["images"]
    assert len(dataset) == 4
    assert batch_images.shape == (1, 2, 1)


def test_burst_dataset_iterates_after_batch_size_change():
    images = np.random.default_rng(0).random((8, 8, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=4, shuffle=False)
    assert len(dataset) == 16

    dataset.set_batch_size(32)
    batches = list(dataset)

    assert len(dataset) == 2
    assert len(batches) == 2
    assert batches[0]["images"].shape == (32, 2, 1)


def test_burst_dataset_can_keep_fixed_epoch_size_after_batch_size_change():
    images = np.random.default_rng(0).random((8, 8, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=32, shuffle=False)
    dataset.set_epoch_size(len(dataset))
    assert len(dataset) == 2

    dataset.set_batch_size(6)
    batches = list(dataset)

    assert len(dataset) == 2
    assert len(batches) == 2
    assert batches[0]["images"].shape == (6, 2, 1)


def test_validation_dataloader_wraps_batches_with_dataset_index():
    images = np.random.default_rng(2).random((4, 4, 2)).astype("float32")
    dataset = BurstDataset.from_numpy(images, batch_size=10, shuffle=False)
    dm = BurstDataModule(dataset, validation_batch_size=10)

    batch = next(iter(dm.val_dataloader()))

    assert len(dm.valid_dataset) == 16
    assert len(dm.val_dataloader()) == 2
    assert batch["dataset_idx"].tolist() == list(range(10))
    assert batch["indices"].tolist() == list(range(10))

    dm.set_validation_sampling_points(4)
    batches = list(dm.val_dataloader())
    assert len(dm.valid_dataset) == 16
    assert len(batches) == 4
    assert torch.cat([batch["indices"] for batch in batches]).tolist() == list(range(16))


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


def test_npz_loader_prefers_unified_images_key(tmp_path):
    path = tmp_path / "prepared.npz"
    images = np.ones((3, 4, 2, 1), dtype="float32")
    np.savez_compressed(path, metadata=np.array("{}"), images=images)

    loaded = load_burst_from_config({"type": "npz", "path": str(path)})

    assert loaded.shape == images.shape
    assert np.all(loaded == images)


def test_load_burst_from_config_reports_progress(tmp_path):
    path = tmp_path / "prepared.npz"
    images = np.ones((3, 4, 2, 1), dtype="float32")
    np.savez_compressed(path, images=images)
    events = []

    loaded = load_burst_from_config(
        {"type": "npz", "path": str(path), "frame_indices": [0], "channels": [0]},
        progress_callback=lambda *event: events.append(event),
    )

    assert loaded.shape == (3, 4, 1, 1)
    assert events[0] == ("load_start", None, None, str(path))
    assert ("load_done", None, None, images.shape) in events
    assert ("select_frames_done", None, None, (3, 4, 1, 1)) in events
    assert events[-1] == ("subframe_done", None, None, (3, 4, 1, 1))


def test_alignment_loader_config_expands_size_subframe():
    loader_config = {"type": "gregor", "subframe": {"x": 20, "y": 30, "size": 16}}

    alignment_config, trim_buffer = build_alignment_loader_config(loader_config, max_shift=3)

    assert alignment_config["subframe"] == {"x": 20, "y": 30, "size": 22}
    assert trim_buffer == (3, 3)
    assert loader_config["subframe"] == {"x": 20, "y": 30, "size": 16}


def test_alignment_loader_config_expands_range_subframe():
    loader_config = {"type": "gregor", "subframe": {"x_range": [10, 30], "y_range": [40, 70]}}

    alignment_config, trim_buffer = build_alignment_loader_config(loader_config, max_shift=4)

    assert alignment_config["subframe"] == {"x_range": [6, 34], "y_range": [36, 74]}
    assert trim_buffer == (4, 4)


def test_trim_alignment_buffer_removes_buffer_region():
    images = np.arange(8 * 10 * 2, dtype="float32").reshape(8, 10, 2)

    trimmed = trim_alignment_buffer(images, (2, 3))

    assert trimmed.shape == (4, 4, 2)
    assert np.all(trimmed == images[2:-2, 3:-3, :])


def test_align_stack_reports_frame_progress():
    images = np.zeros((4, 4, 3), dtype="float32")
    events = []

    align_stack(images, max_shift=0, progress_callback=lambda *event: events.append(event))

    assert events[0] == (
        "align_start",
        0,
        3,
        {"max_shift": 0, "candidate_count": 1, "num_workers": 1},
    )
    assert events[1] == ("align_frame_start", 1, 3, {"frame_index": 0})
    assert events[2][0:3] == ("align_frame_done", 1, 3)
    assert events[2][3]["frame_index"] == 0
    assert events[2][3]["shift"] == [0, 0]
    assert events[-1] == ("align_done", 3, 3, {"shifts": [[0, 0], [0, 0], [0, 0]]})


def test_to_channels_last_converts_gregor_cube_layouts():
    fyx = np.zeros((5, 8, 9), dtype="float32")
    yxf = np.zeros((8, 9, 5), dtype="float32")
    fyxc = np.zeros((5, 8, 9, 2), dtype="float32")

    assert to_channels_last(fyx, axis_order="fyx").shape == (8, 9, 5, 1)
    assert to_channels_last(yxf, axis_order="yxf").shape == (8, 9, 5, 1)
    assert to_channels_last(fyxc, axis_order="fyxc").shape == (8, 9, 5, 2)


def test_gregor_interleaved_hdu_stack_keeps_source_channels():
    images = np.arange(2 * 3 * 6, dtype="float32").reshape(2, 3, 6)

    grouped = group_interleaved_channels(images, 2)

    assert grouped.shape == (2, 3, 3, 2)
    np.testing.assert_array_equal(grouped[:, :, :, 0], images[:, :, 0::2])
    np.testing.assert_array_equal(grouped[:, :, :, 1], images[:, :, 1::2])


def test_save_prepared_burst_writes_images_and_metadata(tmp_path):
    args = SimpleNamespace(
        input="/source/hifi.fts",
        axis_order="fyx",
        hdu_indices=None,
        hdu_start=0,
        hdu_stop=None,
        source_channels=2,
        source_channel=None,
        frame_indices="0,1",
        frame_start=None,
        frame_stop=None,
        frame_step=None,
        n_images=None,
        channels=None,
        subframe_x=None,
        subframe_y=None,
        subframe_size=8,
        x_range=None,
        y_range=None,
    )
    images = np.ones((8, 8, 2, 1), dtype="float32")
    loader_config = build_loader_config(args)
    metadata = build_metadata(args, loader_config, images, [[0, 0], [1, -1]])
    path = tmp_path / "prepared.npz"

    save_prepared_burst(path, images, metadata)

    with np.load(path) as data:
        assert sorted(data.files) == ["images", "metadata"]
        assert data["images"].shape == (8, 8, 2, 1)
        loaded_metadata = json.loads(str(data["metadata"]))
    assert loaded_metadata["layout"] == "height,width,frames,channels"
    assert loaded_metadata["shape"] == [8, 8, 2, 1]
    assert loaded_metadata["preparation"]["subframe"] == {"size": 8}
    assert loaded_metadata["alignment"]["method"] == "bounded-normalized-cross-correlation"
    assert loaded_metadata["alignment"]["max_shift"] == 0
    assert loaded_metadata["alignment"]["shifts"] == [[0, 0], [1, -1]]
