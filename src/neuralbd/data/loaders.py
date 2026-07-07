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
        key = array_key or data.files[0]
        return data[key]
    return data


def load_npz_burst(path, array_key=None):
    return load_numpy_burst(path, array_key=array_key)


def load_fits_burst(path, hdu_indices=None, first_hdu=0):
    fits = _optional_astropy_fits()
    with fits.open(path) as hdul:
        if hdu_indices is None:
            hdu_indices = range(first_hdu, len(hdul))
        frames = [np.asarray(hdul[idx].data, dtype="float32") for idx in hdu_indices if hdul[idx].data is not None]
    if not frames:
        raise ValueError(f"No image HDUs found in {path}")
    return np.stack(frames, axis=2)


def load_gregor_burst(path, hdu_indices=None, hdu_start=1, hdu_stop=None, channel=None):
    fits = _optional_astropy_fits()
    with fits.open(path) as hdul:
        stop = len(hdul) if hdu_stop is None else hdu_stop
        indices = hdu_indices or range(hdu_start, stop)
        frames = [np.asarray(hdul[idx].data, dtype="float32") for idx in indices if hdul[idx].data is not None]
    if not frames:
        raise ValueError(f"No GREGOR image HDUs found in {path}")
    images = np.stack(frames, axis=2)
    if channel is not None and images.ndim == 4:
        images = images[..., int(channel)]
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


def load_burst_from_config(data_config):
    data_type = data_config["type"].lower()
    path = data_config.get("path")
    if path is None:
        raise ValueError("data.path is required")
    path = Path(path)

    if data_type == "numpy":
        images = load_numpy_burst(path, array_key=data_config.get("array_key"))
    elif data_type == "npz":
        images = load_npz_burst(path, array_key=data_config.get("array_key"))
    elif data_type == "fits":
        images = load_fits_burst(
            path,
            hdu_indices=data_config.get("hdu_indices"),
            first_hdu=data_config.get("first_hdu", 0),
        )
    elif data_type == "gregor":
        images = load_gregor_burst(
            path,
            hdu_indices=data_config.get("hdu_indices"),
            hdu_start=data_config.get("hdu_start", 1),
            hdu_stop=data_config.get("hdu_stop"),
            channel=data_config.get("source_channel"),
        )
    elif data_type == "dkist":
        images = load_dkist_burst(path, array_key=data_config.get("array_key") or "cobs")
    elif data_type == "kso":
        images = load_kso_burst(path, array_key=data_config.get("array_key"))
    else:
        raise ValueError(f"Unknown data.type: {data_config['type']}")

    images = select_frames(
        images,
        frame_slice=data_config.get("frame_slice"),
        frame_indices=data_config.get("frame_indices"),
        n_images=data_config.get("n_images"),
    )
    images = select_channels(images, data_config.get("channels"))
    frame_cfg = data_config.get("subframe") or {}
    images = subframe(
        images,
        x=frame_cfg.get("x"),
        y=frame_cfg.get("y"),
        size=frame_cfg.get("size"),
        x_range=frame_cfg.get("x_range"),
        y_range=frame_cfg.get("y_range"),
    )
    return images
