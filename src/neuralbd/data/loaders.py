from pathlib import Path

import numpy as np

from neuralbd.processing import subframe


def _optional_astropy_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("FITS loading requires the optional dependency astropy.") from exc
    return fits


def _dkist_to_channels_last(array):
    array = np.asarray(array, dtype="float32")
    if array.ndim == 3:
        return np.moveaxis(array, 0, 2)
    if array.ndim == 4:
        return np.moveaxis(array, 0, 2)
    return array


def load_numpy_burst(path, array_key=None):
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        key = array_key or ("images" if "images" in data.files else data.files[0])
        return data[key]
    return data


def load_npz_burst(path, array_key=None):
    return load_numpy_burst(path, array_key=array_key)


def to_channels_last(array, axis_order="auto"):
    array = np.asarray(array, dtype="float32").squeeze()
    if array.ndim == 2:
        return array[:, :, None]
    if axis_order == "auto":
        axis_order = _infer_axis_order(array.shape)
    axis_order = axis_order.lower()
    if len(axis_order) != array.ndim or sorted(axis_order) != sorted(set(axis_order)):
        raise ValueError("axis_order must name each input axis once, for example 'fyx' or 'fyxc'")
    if not set(axis_order).issubset({"f", "y", "x", "c"}):
        raise ValueError("axis_order may only contain f, y, x, and optional c")
    if "y" not in axis_order or "x" not in axis_order or "f" not in axis_order:
        raise ValueError("axis_order must include y, x, and f axes")
    order = [axis_order.index(axis) for axis in "yxf" if axis in axis_order]
    if "c" in axis_order:
        order.append(axis_order.index("c"))
    converted = np.transpose(array, order)
    if converted.ndim == 3:
        converted = converted[..., None]
    return converted


def group_interleaved_channels(images, channel_count):
    if channel_count is None:
        return images
    channel_count = int(channel_count)
    if channel_count <= 1:
        return images[..., None] if images.ndim == 3 else images
    if images.ndim != 3:
        return images
    if images.shape[2] % channel_count != 0:
        raise ValueError(
            f"Cannot split {images.shape[2]} GREGOR HDU images into {channel_count} interleaved channels"
        )
    return images.reshape(images.shape[0], images.shape[1], images.shape[2] // channel_count, channel_count)


def _infer_axis_order(shape):
    if len(shape) == 3:
        first_is_frame = shape[0] <= min(shape[1:])
        last_is_frame = shape[-1] <= min(shape[:2])
        return "fyx" if first_is_frame and not last_is_frame else "yxf"
    if len(shape) == 4:
        spatial_axes = sorted(range(4), key=lambda axis: shape[axis], reverse=True)[:2]
        non_spatial_axes = [axis for axis in range(4) if axis not in spatial_axes]
        frame_axis = non_spatial_axes[0] if shape[non_spatial_axes[0]] >= shape[non_spatial_axes[1]] else non_spatial_axes[1]
        channel_axis = non_spatial_axes[1] if frame_axis == non_spatial_axes[0] else non_spatial_axes[0]
        labels = [""] * 4
        labels[min(spatial_axes)] = "y"
        labels[max(spatial_axes)] = "x"
        labels[frame_axis] = "f"
        labels[channel_axis] = "c"
        return "".join(labels)
    raise ValueError("FITS image data must be 2D, 3D, or 4D")


def load_fits_burst(path, hdu_indices=None, first_hdu=0, axis_order="auto", progress_callback=None):
    fits = _optional_astropy_fits()
    with fits.open(path) as hdul:
        if hdu_indices is None:
            hdu_indices = list(range(first_hdu, len(hdul)))
        else:
            hdu_indices = list(hdu_indices)
        arrays = []
        for count, idx in enumerate(hdu_indices, start=1):
            if progress_callback is not None:
                progress_callback("load_hdu", count, len(hdu_indices), idx)
            if hdul[idx].data is not None:
                arrays.append(to_channels_last(hdul[idx].data, axis_order=axis_order))
    if not arrays:
        raise ValueError(f"No image HDUs found in {path}")
    return np.concatenate(arrays, axis=2)


def load_gregor_burst(
    path,
    hdu_indices=None,
    hdu_start=0,
    hdu_stop=None,
    channel=None,
    source_channels=2,
    axis_order="auto",
    progress_callback=None,
):
    fits = _optional_astropy_fits()
    with fits.open(path) as hdul:
        stop = len(hdul) if hdu_stop is None else hdu_stop
        indices = list(hdu_indices) if hdu_indices is not None else list(range(hdu_start, stop))
        arrays = []
        for count, idx in enumerate(indices, start=1):
            if progress_callback is not None:
                progress_callback("load_hdu", count, len(indices), idx)
            if hdul[idx].data is not None:
                arrays.append(to_channels_last(hdul[idx].data, axis_order=axis_order))
    if not arrays:
        raise ValueError(f"No GREGOR image HDUs found in {path}")
    images = np.concatenate(arrays, axis=2)
    images = group_interleaved_channels(images, source_channels)
    if channel is not None:
        images = images[..., int(channel) : int(channel) + 1]
    return images


def load_dkist_burst(path, array_key="cobs"):
    return _dkist_to_channels_last(load_npz_burst(path, array_key=array_key))


def load_kso_burst(path, array_key=None):
    return load_numpy_burst(path, array_key=array_key)


def select_frames(images, frame_slice=None, frame_indices=None, n_images=None):
    if frame_indices is not None:
        images = images[:, :, [int(idx) for idx in frame_indices], ...]
    elif frame_slice:
        start = frame_slice.get("start")
        stop = frame_slice.get("stop")
        step = frame_slice.get("step")
        images = images[:, :, slice(start, stop, step), ...]
    if n_images is not None:
        images = images[:, :, : int(n_images), ...]
    return images


def select_channels(images, channels=None):
    if channels is None:
        return images
    if images.ndim == 3:
        images = images[..., None]
    if isinstance(channels, int):
        channels = [channels]
    return images[..., [int(channel) for channel in channels]]


def load_burst_from_config(data_config, progress_callback=None):
    data_type = data_config["type"].lower()
    path = data_config.get("path")
    if path is None:
        raise ValueError("data.path is required")
    path = Path(path)
    if progress_callback is not None:
        progress_callback("load_start", None, None, str(path))

    if data_type == "numpy":
        images = load_numpy_burst(path, array_key=data_config.get("array_key"))
    elif data_type == "npz":
        images = load_npz_burst(path, array_key=data_config.get("array_key"))
    elif data_type == "fits":
        images = load_fits_burst(
            path,
            hdu_indices=data_config.get("hdu_indices"),
            first_hdu=data_config.get("first_hdu", 0),
            axis_order=data_config.get("axis_order", "auto"),
            progress_callback=progress_callback,
        )
    elif data_type == "gregor":
        images = load_gregor_burst(
            path,
            hdu_indices=data_config.get("hdu_indices"),
            hdu_start=data_config.get("hdu_start", 0),
            hdu_stop=data_config.get("hdu_stop"),
            channel=data_config.get("source_channel"),
            source_channels=data_config.get("source_channels", 2),
            axis_order=data_config.get("axis_order", "auto"),
            progress_callback=progress_callback,
        )
    elif data_type == "dkist":
        images = load_dkist_burst(path, array_key=data_config.get("array_key") or "cobs")
    elif data_type == "kso":
        images = load_kso_burst(path, array_key=data_config.get("array_key"))
    else:
        raise ValueError(f"Unknown data.type: {data_config['type']}")

    if progress_callback is not None:
        progress_callback("load_done", None, None, tuple(images.shape))
    images = select_frames(
        images,
        frame_slice=data_config.get("frame_slice"),
        frame_indices=data_config.get("frame_indices"),
        n_images=data_config.get("n_images"),
    )
    if progress_callback is not None:
        progress_callback("select_frames_done", None, None, tuple(images.shape))
    images = select_channels(images, data_config.get("channels"))
    if progress_callback is not None:
        progress_callback("select_channels_done", None, None, tuple(images.shape))
    frame_cfg = data_config.get("subframe") or {}
    images = subframe(
        images,
        x=frame_cfg.get("x"),
        y=frame_cfg.get("y"),
        size=frame_cfg.get("size"),
        x_range=frame_cfg.get("x_range"),
        y_range=frame_cfg.get("y_range"),
    )
    if progress_callback is not None:
        progress_callback("subframe_done", None, None, tuple(images.shape))
    return images
